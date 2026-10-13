# Fusion géométrique avec pseudo-étiquettes par barycentres de Wasserstein

Ce document décrit l'algorithme exact qu'exécute le code de ce dossier dans la configuration `lr_z256_pseudo`.
C'est la configuration mesurée sur HAC le 12/10 (ABLATIONS, section 20) : 87.44 sur human, 87.69 sur animal,
58.41 sur cartoon, au dernier point, en moyenne sur 3 graines.

Elle reprend `config_exemple.yaml` avec trois changements :

```yaml
fusion: {dim: 256}                              # espace fusionné en 256 dimensions au lieu de 64
phase1: {optimiseur: {lr_extracteur: 5.0e-5}}   # au lieu de 2e-6
pseudo: {actif: true}                           # pseudo-étiquettes pendant l'entraînement
```

**Dans ce dépôt**, la configuration de la méthode est `configs/methode.yaml`, avec **10 époques par défaut** ;
les résultats cités ici ont été obtenus avec 3 époques (`epoques: 3`).

Tout le reste garde sa valeur par défaut : la partition de la cible est **spectrale**, les classes sources sont
résumées par leurs **barycentres de Wasserstein**, et l'appariement est **hongrois**. La perte de maximisation de
l'information existe dans le code mais reste désactivée (poids 0).

Le code se lit dans cet ordre :

| fichier | contenu |
|---|---|
| `phase1.py` | boucle d'entraînement ; `pseudo_etiquettes` (section 5) |
| `pertes.py` | les quatre fonctions de coût (section 4) |
| `phase2.py` | `etiqueter` : barycentres, partition, transport, appariement (section 6) |
| `ot_outils.py` | coût cosinus, Sinkhorn, barycentre de Wasserstein (section 3) |
| `modele.py`, `fusion.py`, `encodeurs.py` | le modèle (section 2) |
| `donnees.py` | lecture des caractéristiques et tirage des lots |

---

## 1. Problème et notations

On dispose de $N$ domaines sources $\mathcal{S}_1, \dots, \mathcal{S}_N$, étiquetés, et d'un domaine cible
$\mathcal{T}$ sans étiquette. Chaque exemple a $M$ modalités, ordonnées, et appartient à l'une de $K$ classes.

Le **train** cible sert à l'entraînement, sans ses étiquettes. Le **test** cible sert à l'évaluation. Ses
étiquettes ne servent qu'à mesurer l'exactitude.

Sur HAC : $N = 2$ (les deux domaines autres que la cible), $M = 2$ (audio, puis vidéo), $K = 7$ actions.

## 2. Modèle

**Encodeur de la modalité $m$.** Une dorsale, puis un extracteur $\mathrm{Linear}(d_m, 256)$ suivi de ReLU, puis
une normalisation L2 :

$$e_m = \frac{\mathrm{ReLU}(W_m\, b_m(x_m))}{\lVert \mathrm{ReLU}(W_m\, b_m(x_m)) \rVert} \in \mathbb{R}^{256}$$

Sur HAC, la dorsale $b_m$ est l'identité : les entrées sont des caractéristiques déjà extraites (AST pour l'audio,
768 dimensions ; SlowFast pour la vidéo, 2304 dimensions). L'extracteur est donc la première couche entraînée.

**Tête unimodale de la modalité $m$.** Un perceptron $256 \to 128 \to 64 \to K$, avec ReLU après la première
couche, et sa prédiction $p_m = \mathrm{softmax}(g_m(e_m))$.

**Fusion MLB** (*Multimodal Low-rank Bilinear*) vers l'espace fusionné, ici en 256 dimensions :

$$z = P\big(\sigma(U_0 e_0 \odot U_1 e_1)\big) + h_0(e_0) + h_1(e_1)$$

avec $U_m : 256 \to 128$, $P : 128 \to 256$, $h_m : 256 \to 256$, $\sigma$ la sigmoïde et $\odot$ le produit terme
à terme. Avec $M \geq 3$, le produit porte sur les $M$ projections et la somme sur les $M$ termes $h_m$.

$z$ n'est pas normalisé. **Aucun classifieur n'est entraîné sur $z$** : les classes de la cible y seront lues par
la géométrie (section 6).

## 3. Outils de transport optimal

**Coût cosinus.** Pour deux vecteurs, $c(u, v) = \lVert \tfrac{u}{\lVert u \rVert} - \tfrac{v}{\lVert v \rVert} \rVert^2 = 2 - 2\cos(u, v) \in [0, 4]$.

**Sinkhorn.** Pour une matrice de coût $C$ et deux mesures $a$ et $b$, le plan de transport régularisé par
l'entropie

$$\Gamma^\star = \arg\min_{\Gamma \mathbf{1} = a,\ \Gamma^\top \mathbf{1} = b} \langle \Gamma, C \rangle - \varepsilon H(\Gamma)$$

est calculé par les itérations de Sinkhorn en domaine logarithmique, plus stables.

**Coût de transport entre deux nuages** $X$ et $Y$ de poids uniformes : $W_\varepsilon(X, Y) = \langle \Gamma^\star, C \rangle$,
avec $C_{ij} = c(x_i, y_j)$, $\varepsilon = 0.02$ et 60 itérations.

**Barycentre de Wasserstein à support libre** de $N$ nuages $X_1, \dots, X_N$. On cherche 16 points
$B = (b_1, \dots, b_{16})$, de poids $\tfrac{1}{16}$, qui minimisent $\sum_k W_\varepsilon(B, X_k)$. Point fixe :

1. initialiser $B$ par 16 points tirés au hasard dans l'union des nuages ;
2. pour chaque nuage $k$, calculer le plan $\Gamma_k$ entre $B$ et $X_k$ (Sinkhorn, $\varepsilon = 0.02$, 60 itérations) ;
3. remplacer chaque $b_s$ par la moyenne, sur les nuages, de sa projection barycentrique :
   $b_s \leftarrow \frac{1}{N} \sum_k \frac{1}{a_s} \sum_j \Gamma_k[s, j]\, x_{k,j}$, avec $a_s = \tfrac{1}{16}$ ;
4. répéter les étapes 2 et 3, 8 fois en tout.

Le barycentre de la classe $h$ est un nuage de 16 points placé « au milieu » des nuages de cette classe dans
chaque source, au sens du transport optimal. Il représente la classe telle que les sources la partagent.

## 4. Phase 1 : fonction de coût

À chaque pas, on tire 4 exemples dans chaque domaine : les sources, puis la cible. Chaque exemple a une vue faible
$x$ et une vue augmentée $\tilde{x}$. La perte est

$$\mathcal{L} = \mathcal{L}_{\text{têtes}} + \mathcal{L}_{\text{CMCFA}} + \mathcal{L}_{\text{CDCFA}} + \mathcal{L}_{\text{SC}}^{\text{fus}}$$

avec des poids tous égaux à 1.

**Têtes unimodales**, sur les sources seulement :

$$\mathcal{L}_{\text{têtes}} = \frac{1}{MN} \sum_{i=1}^{N} \sum_{m} \mathrm{CE}\big(g_m(e_m^{(i)}), y^{(i)}\big)$$

**CMCFA** (*cross-modal contrastive feature alignment*, alignement contrastif entre modalités), sur tous les
domaines, cible comprise, sans étiquette. Pour un exemple $x$ d'un lot, avec $a = e_0$ et $b = e_1$, l'InfoNCE de
M2CAN, sans température, sur les quatre combinaisons de vues :

$$\ell(x) = -\log \frac{\sum_{(u, v)} \exp \langle u_x, v_x \rangle}{\sum_{(u, v)} \sum_{x'} \exp \langle u_x, v_{x'} \rangle}, \qquad (u, v) \in \{a, \tilde{a}\} \times \{b, \tilde{b}\}$$

où $x'$ parcourt le lot. La cohérence entre les deux têtes vaut $V(x) = \mathrm{KL}(p_1 \,\Vert\, p_0)$. Les exemples
où les deux modalités se contredisent pèsent moins :

$$\mathcal{L}_{\text{CMCFA}} = \sum_{d=1}^{N+1} \Big[ \frac{1}{4} \sum_{x \in \text{lot}_d} \ell(x)\, e^{-V(x)} + \frac{1}{4} \sum_{x \in \text{lot}_d} V(x) \Big]$$

**CDCFA** (*cross-domain contrastive feature alignment*, alignement contrastif entre domaines), une perte
contrastive supervisée (SupCon) par modalité, sur les exemples sources réunis :

$$\mathcal{L}_{\text{CDCFA}} = \sum_m \mathrm{SupCon}\Big(\big\{e_m^{(i)}\big\}_{i \leq N},\ \big\{y^{(i)}\big\}_{i \leq N},\ \tau\Big), \qquad \tau = 0.07$$

$$\mathrm{SupCon}(Z, Y, \tau) = -\frac{1}{|A|} \sum_{a \in A} \frac{1}{|P(a)|} \sum_{p \in P(a)} \log \frac{\exp(z_a \cdot z_p / \tau)}{\sum_{b \neq a} \exp(z_a \cdot z_b / \tau)}$$

où $P(a)$ contient les autres exemples de même étiquette que $a$, et $A$ les exemples qui en ont au moins un.
Elle rapproche les exemples de même classe venus de sources différentes.

**SupCon sur l'espace fusionné**, sur les sources **et** les exemples cibles du lot qui ont une pseudo-étiquette :

$$\mathcal{L}_{\text{SC}}^{\text{fus}} = \mathrm{SupCon}\Big(\big\{z^{(i)}\big\}_{i \leq N} \cup \big\{z^{\mathcal{T}}_j\big\}_{j \in \mathcal{P}},\ \big\{y^{(i)}\big\}_{i \leq N} \cup \big\{\hat{y}_j\big\}_{j \in \mathcal{P}},\ \tau\Big)$$

$\mathcal{P}$ est l'ensemble des exemples cibles du lot qui figurent dans la table des pseudo-étiquettes
(section 5). Les $z$ entrent tels quels, sans normalisation. C'est le seul terme où les pseudo-étiquettes agissent :
un point cible étiqueté $h$ est tiré vers les exemples sources de la classe $h$ dans l'espace fusionné.

**Optimisation.** Adam, $\beta = (0.9, 0.999)$. Taux d'apprentissage : extracteurs $5 \times 10^{-5}$, fusion
$5 \times 10^{-5}$, têtes $5 \times 10^{-5}$ (la dorsale, identité, n'a pas de paramètre). Décroissance
polynomiale $\big(1 - t / T\big)^{0.9}$ sur $T = 3 \times 3750 = 11\,250$ pas.

## 5. Pseudo-étiquettes (tous les 750 pas)

Au début des pas $t = 750, 1500, \dots, 11\,250$, soit 15 tours numérotés $r = 0, \dots, 14$, le modèle passe en
évaluation, sans gradient :

1. **Représentations.** $z$ de tous les exemples sources du train, avec leurs étiquettes. Pour au plus 3000
   exemples du train cible : $z_j$ et la prédiction moyenne des têtes $\bar{p}_j = \tfrac{1}{M} \sum_m p_{m,j}$.
2. **Étiquetage géométrique** de ces exemples cibles par la section 6, qui donne pour chacun une classe
   $\hat{y}_j$, son groupe et sa marge.
3. **Filtre d'accord.** On ne garde que les points où la prédiction moyenne des têtes désigne la même classe que
   la géométrie : $\arg\max_h \bar{p}_{j,h} = \hat{y}_j$. Une tête peut se tromper si l'autre est assez sûre.
4. **Quota par classe.** Avec $\rho_r = (0.3, 0.6, 0.9, 1.0)[\min(r, 3)]$ et $n$ le nombre de points cibles
   étiquetés, on garde dans chaque classe au plus $\lfloor \rho_r\, n / K \rfloor$ points, ceux de plus grande marge.
5. **Mise à jour.** La table $\{\text{identifiant} \mapsto \hat{y}_j\}$ des points retenus **remplace** celle du tour
   précédent et sert jusqu'au tour suivant.

Les vraies étiquettes du train cible ne servent qu'à journaliser la justesse des pseudo-étiquettes.

## 6. Étiquetage géométrique de la cible (cœur de la méthode)

Entrées : les $z$ et étiquettes des sources, les $z$ de $n$ points cibles. Cette procédure sert aux
pseudo-étiquettes (section 5, sur le train cible) et à l'évaluation (section 7, sur le test cible).

**Étape 1 : barycentres de Wasserstein des classes sources.** Pour chaque classe $h$ et chaque source $k$, on tire
au plus 400 points $z$ de classe $h$. Le barycentre de Wasserstein de ces $N$ nuages (section 3) donne
$B_h$, 16 points.

**Étape 2 : partition spectrale de la cible.**

1. Distances $C_{ij} = c(z_i, z_j)$ entre tous les points cibles.
2. Ressemblances $A_{ij} = \exp(-C_{ij} / \sigma)$, avec $\sigma$ la médiane des $C_{ij}$ non nuls, et $A_{ii} = 1$.
3. Partition spectrale en $K$ groupes sur ce graphe : vecteurs propres du laplacien normalisé
   $I - D^{-1/2} A D^{-1/2}$, puis k-moyennes sur ces nouvelles coordonnées.

On obtient des groupes $G_1, \dots, G_K$ dont les numéros ne correspondent encore à aucune classe.

**Étape 3 : coût entre classes et groupes.** $W_{hl} = W_\varepsilon(B_h, G_l)$, le coût de transport entre le
barycentre de la classe $h$ et les points du groupe $l$ ($\varepsilon = 0.02$, 60 itérations).

**Étape 4 : transport entre structures et appariement.** Sinkhorn sur la matrice $W$ ($K \times K$), marges uniformes
$\tfrac{1}{K}$, $\varepsilon = 0.02$, 300 itérations, donne un plan $G$. L'algorithme hongrois appliqué à $-G$
apparie chaque groupe à une classe distincte : $\pi : l \mapsto h$.

**Étape 5 : étiquettes.** Chaque point du groupe $l$ reçoit la classe $\pi(l)$ : $\hat{y}_j = \pi(\text{groupe}(j))$.

**Marge géométrique** (utilisée par le quota de la section 5). Avec $c_l$ le centre du groupe $l$ :

$$\mathrm{marge}_j = \frac{\min_{l \neq \text{groupe}(j)} c(z_j, c_l)}{c(z_j, c_{\text{groupe}(j)})}$$

Plus elle est grande, plus le point est central dans son groupe.

C'est une adaptation de HOT-DA (*hierarchical optimal transport for domain adaptation*, El Hamri et al.) : le
transport se fait entre structures, des barycentres de classe vers des groupes cibles, et non point à point. La
différence est l'usage, à l'étape 1, du barycentre de Wasserstein des sources plutôt que d'un seul domaine source.

## 7. Évaluation

Après les pas 750, 1500, …, 11 250, la section 6 est appliquée au **test** cible. On journalise :

- l'**exactitude** des étiquettes $\hat{y}$ ;
- l'**oracle** : l'exactitude obtenue avec le meilleur appariement possible des groupes aux classes, qui mesure la
  qualité de la seule partition ;
- le **ratio intra sur inter** des distances, critère d'arrêt sans étiquette.

Le résultat rapporté est celui du **dernier point**. Le meilleur point, choisi avec les étiquettes du test, n'est
qu'un plafond optimiste.

## 8. Algorithme

```
Entrées : sources S_1..S_N étiquetées, cible T (train sans étiquette, test), M modalités, K classes
Initialiser encodeurs f_m (extracteur 256-d), têtes g_m, fusion MLB (z en 256-d)
table ← {} ; r ← 0
pour t = 1 .. 11 250 :
    si t mod 750 = 0 :                                                   # section 5
        Z_s, Y_s ← z des sources ; Z_t, P_t, ids ← z et têtes moyennes du train cible (≤ 3000)
        ŷ, groupes, marge ← ETIQUETER(Z_s, Y_s, Z_t)
        garder ← { j : argmax P_t[j] = ŷ[j] }
        garder ← par classe, les ⌊ρ_r n / K⌋ points de garder de plus grande marge
        table ← { ids[j] : ŷ[j] , j ∈ garder } ; r ← r + 1
    tirer 4 exemples par domaine (sources puis cible), vues faible et augmentée
    e_m ← f_m(x_m) ; p_m ← softmax(g_m(e_m)) ; z ← MLB(e_0, e_1)
    L ← L_têtes + L_CMCFA + L_CDCFA + L_SC_fus(sources ∪ cibles du lot présentes dans table)
    pas d'Adam sur L
    si t mod 750 = 0 : ETIQUETER(z des sources, z du test cible), journaliser exactitude, oracle, intra/inter
Sortie : étiquettes du test cible au dernier point

ETIQUETER(Z_s, Y_s, Z_t) :                                               # section 6
    pour chaque classe h : B_h ← barycentre de Wasserstein des z de classe h de chaque source (16 points)
    A ← exp(-coût cosinus(Z_t, Z_t) / médiane) ; groupes ← partition spectrale de A en K
    W[h, l] ← coût de Sinkhorn entre B_h et le groupe l
    G ← Sinkhorn(W, marges 1/K, ε = 0.02) ; π ← hongrois sur -G
    ŷ[j] ← π(groupes[j]) ; marge ← marge géométrique
    renvoyer ŷ, groupes, marge
```

## 9. Hyperparamètres

| élément | valeur |
|---|---|
| époques, pas par époque, lot par domaine | 3, 3750, 4 |
| dimension de $e_m$, de $z$ | 256, 256 |
| taux d'apprentissage (extracteur, fusion, têtes) | $5 \times 10^{-5}$ |
| décroissance | polynomiale, puissance 0.9 |
| température SupCon $\tau$ | 0.07 |
| poids des quatre pertes | 1 |
| barycentre : support, $\varepsilon$, itérations de Sinkhorn, itérations de point fixe, points par classe et source | 16, 0.02, 60, 8, 400 |
| transport entre structures : $\varepsilon$, itérations | 0.02, 300 |
| pseudo-étiquettes : période, points cibles max, taux $\rho$, filtre | 750 pas, 3000, (0.3, 0.6, 0.9, 1.0), accord |
| évaluation | tous les 750 pas, dernier point rapporté |

## 10. Résultats et conditions

HAC, audio AST + vidéo SlowFast, dernier point, moyenne sur 3 graines (123456, 234567, 345678) :

| configuration | human | animal | cartoon | moyenne |
|---|---|---|---|---|
| base (z 64-d, extracteur $2 \times 10^{-6}$, sans pseudo-étiquettes) | 68.60 | 66.48 | 52.91 | 62.66 |
| z 256-d, extracteur $5 \times 10^{-5}$ | 89.37 | 76.89 | 55.81 | 74.02 |
| **z 256-d, extracteur $5 \times 10^{-5}$, pseudo-étiquettes (ce document)** | **87.44** | **87.69** | **58.41** | **77.85** |

Contre la base : +15.19 points, IC95 [+9.26, +21.11], 9 paires sur 9 positives. L'apport propre des
pseudo-étiquettes (+3.82, IC95 [−1.41, +9.06]) n'est pas significatif : +10.8 sur animal, −1.9 sur human.
Justesse des pseudo-étiquettes retenues sur animal : 88 à 96 % au premier tour, 87 à 91 % en fin d'entraînement.

Repères, non appariés (autre extraction, vues augmentées fortes, meilleur point) : M2CAN 85.99 / 84.85 / 72.94 ;
plancher sans adaptation 76.59 / 80.68 / 50.92.

**Conditions à rappeler dans le mémoire.**

- Caractéristiques gelées : seuls les extracteurs, la fusion et les têtes apprennent. Le taux de $5 \times 10^{-5}$
  est réglé pour ce cas ; avec BERT et ResNet entraînés (sentiment), il reste à régler.
- Les caractéristiques HAC utilisées n'ont pas de vue augmentée : $\tilde{x} = x$ dans CMCFA.
- Le test cible de HAC sert aussi à choisir le meilleur point, d'où le choix du dernier point comme mesure.
- Méthode transductive : le train cible, sans étiquette, est vu pendant l'entraînement.
- Non mesuré sur le sentiment (Twitter, Yelp, TumEmo).
