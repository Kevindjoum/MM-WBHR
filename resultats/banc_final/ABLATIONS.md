# Registre des versions, ablations et résultats

Document de travail pour la rédaction du mémoire. Chaque entrée décrit **ce qui a été fait et
pourquoi**, puis **les résultats**, puis **ce qu'on en conclut**. Les entrées sont chronologiques.
Un gabarit en fin de document sert à ajouter les prochaines.

Dernière mise à jour : 2026-10-09.

---

## 0. Protocole commun

Tout ce qui suit partage ce cadre, sauf mention contraire.

| élément | valeur |
|---|---|
| sources étiquetées | TumEmo, Yelp |
| cible non étiquetée | Twitter (T4SA), sauf section 9 |
| nombre de classes | 3 (polarité) |
| encodeurs | BERT base + ResNet50, affinés de bout en bout |
| fusion | MLB, sortie dans R^64 |
| taille de lot | 4 par domaine, soit 12 échantillons par pas |
| pas par époque | 3750 (15000 échantillons cibles / 4) |
| cadence | 2.56 s/pas, soit 2.67 h par époque |
| validation cible | 1500 points, répartition exacte 500/500/500 sur les 3 cibles |
| matériel | Tesla P100, sessions Kaggle de 12 h maximum |

**Bruit de mesure.** Sur 1500 points à une accuracy proche de 0.70, l'erreur-type d'une mesure vaut
1.18 point. L'erreur-type de la **différence** entre deux mesures indépendantes vaut 1.67 point.
Tout écart inférieur à environ 3.3 points n'est donc pas significatif sur une seule graine.

**Répartition équilibrée.** Les 3 classes cibles comptent exactement 500 points chacune. L'accuracy
globale égale donc l'accuracy équilibrée, et les marginales uniformes du transport optimal sont
correctement spécifiées. Il n'y a pas de décalage marginal d'étiquettes à corriger.

**Réserve transversale, à écrire dans le mémoire.** L'inférence par HOT-DA est **transductive** :
elle partitionne les 1500 points cibles ensemble avant d'assigner les classes, donc elle exploite la
structure collective de l'ensemble évalué. Les méthodes de comparaison publiées prédisent chaque
point indépendamment. Conséquence pratique : notre méthode ne peut pas prédire un échantillon isolé.
Toute comparaison à la littérature doit porter cette étiquette.

---

## 1. Vérification des checkpoints publiés

**Pourquoi.** Avant de construire quoi que ce soit, s'assurer que le code M2CAN reproduit bien les
chiffres de ses auteurs sur notre copie des données.

**Ce qui a été fait.** Chargement des 3 checkpoints de sentiment publiés par les auteurs, évaluation
sur la validation de chaque cible.

**Résultats.**

| cible | notre mesure | annonce des auteurs | écart |
|---|---|---|---|
| TumEmo | 65.00 | 64.87 | +0.13 |
| Yelp | 61.20 | 61.20 | 0.00 |
| Twitter | 84.47 | 86.40 | −1.93 |

**Conclusion.** Le code reproduit. L'écart sur Twitter s'explique probablement par quelques images
manquantes dans notre copie du jeu, le chargeur remplaçant l'image absente par des zéros.

**Point ouvert.** Une analyse extérieure cite une Table 1 du papier donnant 84.7 pour M2CAN sur
T4SA, soit très proche de notre 84.47 et loin des 86.40 du nom de fichier. Il existe donc
probablement 2 protocoles ou 2 tables distinctes. **À trancher sur le papier avant toute phrase de
comparaison.**

**Tranché (09/10).** Le mémoire retient le Tableau 1 de l'article (Zhao et al., Information Fusion 2025) :
84.7 sur Twitter, 63.7 sur Yelp, 63.8 sur TumEmo. Les valeurs 86.40, 61.20 et 64.87 sont celles des noms
des checkpoints publiés. Les sections 11 et 13 comparent désormais aux valeurs du Tableau 1.

---

## 2. Phase 1 CMCFA seule, puis Phase 2 post-hoc

**Pourquoi.** Première tentative de méthode en 2 temps : apprendre les représentations avec
l'alignement inter-modal seul, puis adapter vers la cible par transport optimal appliqué après coup.

**Ce qui a été fait.** Entraînement sur 5 époques avec l'alignement inter-modal seul et un routage
asymétrique de la cible (elle entre dans cet alignement sans étiquette, jamais dans la tâche). Puis
export des représentations fusionnées et application hors ligne du barycentre et du transport
hiérarchique.

**Résultats.** Plafonné autour de 61 à 62 sur Twitter. En reprenant les représentations du
checkpoint **officiel** au lieu des nôtres, la même Phase 2 remonte à 84.00 sur Twitter, 64.87 sur
TumEmo et 60.53 sur Yelp.

**Conclusion.** Notre Phase 1 était le facteur limitant, pas la Phase 2. Et les masses issues de la
cohérence inter-modale n'apportent rien : écart entre −0.20 et +0.13 selon les jeux.

---

## 3. Plancher source seule contrôlé

**Pourquoi.** Disposer d'un point de comparaison sans aucune adaptation.

**Ce qui a été fait.** Entraînement de la tâche seule sur TumEmo et Yelp, adaptation coupée, cible
jamais présentée au réseau pour éviter toute fuite par la normalisation par lot.

**Résultats.**

| méthode | accuracy sur Twitter |
|---|---|
| transfert direct du classifieur source | 59.40 |
| barycentre + transport, 1 plus proche voisin | 60.47 |
| barycentre + transport, ridge pondéré | 61.93 |

**Conclusion.** Le transport appliqué après coup sur des représentations figées ne récupère que 2.53
points sur les 27 qui séparent le plancher de la référence. Formulation retenue : le transport
optimal fabrique de l'alignement quand il est une **perte**, pas quand il est un **post-traitement**.

---

## 4. Ablation A contre B : le barycentre comme perte différentiable

**Pourquoi.** C'était l'expérience manquante du projet. Le barycentre de Wasserstein, rendu
différentiable et placé dans la boucle, apporte-t-il quelque chose ?

**Ce qui a été fait.** Deux bras identiques en tout point sauf un drapeau.

$$\mathcal{L}_A = \mathcal{L}_{\text{task}} + \mathcal{L}_{\text{CMCFA}} + \mathcal{L}_{\text{CDCFA}}
\qquad
\mathcal{L}_B = \mathcal{L}_A + \lambda_{\text{eff}} \mathcal{L}_{\text{WBHR}}$$

Le barycentre est à support libre, 16 points, obtenu par point fixe barycentrique avec 3 itérations
externes et 20 itérations de Sinkhorn en domaine logarithmique. Coût cosinus calculé sur des copies
normalisées, sans toucher au vecteur fusionné. Une banque mémoire de 64 points par couple source et
classe compense la taille de lot de 4, qui ne donne qu'un à deux échantillons par classe.

**Calibration préalable.** Les échelles mesurées sur 400 pas : tâche 3.3, inter-modal 4.0,
inter-sources 3.9, barycentre 0.002. Un poids fixe était donc inutilisable. D'où une pondération
adaptative maintenant le terme à une fraction constante de la perte de tâche.

**Résultats.**

| époque | bras A | bras B | écart |
|---|---|---|---|
| 0 | 70.00 | 70.40 | +0.40 |
| 1 | 65.87 | 66.80 | +0.93 |
| 2 | 68.53 | 68.93 | +0.40 |
| 3 | 69.13 | 68.80 | −0.33 |
| **meilleur** | **70.00** | **70.40** | **+0.40** |

**Conclusion.** Aucun effet mesurable. Et le résultat est valide : la pondération adaptative a tenu
le terme à 8.9 % de la perte de tâche en moyenne, avec les 3 classes actives en permanence, et la
perte brute est montée de 0.0024 à 0.038 à mesure que l'espace fusionné se dépliait.

**Résultat secondaire.** Le bras A seul atteint 70.00 contre un plancher de 59.40. Environ 10.6
points gagnés par les pertes contrastives. Comparaison indicative seulement, les 2 protocoles
diffèrent.

---

## 5. Phase 2 complète sur le bras B

**Pourquoi.** Mesurer ce que le transport hiérarchique complet apporte au-dessus du classifieur.

**Ce qui a été fait.** Barycentres par classe sur les 2 sources, partition spectrale de la cible sur
affinité cosinus, coût de Wasserstein entre structures, Sinkhorn puis appariement hongrois.

**Résultats.**

| méthode de décision | accuracy |
|---|---|
| classifieur transféré | 70.40 |
| transport hiérarchique, masses uniformes | 65.73 |
| transport hiérarchique, masses de cohérence | 65.67 |
| même chose, implémentation indépendante | 66.20 |
| **appariement oracle, plafond de la partition** | **66.20** |
| plus proche barycentre, sans partition | 50.87 |
| classe majoritaire | 33.33 |

**Conclusion, et c'est la plus importante du projet.** L'appariement atteint **exactement** son
plafond oracle. Le transport trouve donc toujours la meilleure bijection possible. Le goulot n'est
pas le transport, c'est la **qualité de la partition de la cible**. Et ce plafond était alors sous le
classifieur, ce qui rendait le transport structurellement perdant.

Ce constat s'est vérifié sur toutes les représentations entraînées de Twitter. **Il a 2 exceptions,
découvertes ensuite** (section 9) : au tout début de l'entraînement, quand l'espace fusionné est
encore quasi aléatoire, et quand la partition dégénère avec un groupe presque vide. Dans ces 2 cas
les marginales uniformes du transport sont mal spécifiées et l'appariement rate le plafond, de 0.13
à 4.67 points selon les mesures.

---

## 6. Filtre de marge géométrique

**Pourquoi.** Le transport se trompe en moyenne. Se trompe-t-il uniformément, ou concentre-t-il ses
erreurs sur les points frontières ?

**Ce qui a été fait.** Pour chaque point cible, calcul de la marge

$$\text{marge}_j = \frac{d(e_j,\ \text{centre du groupe voisin le plus proche})}{d(e_j,\ \text{centre de son groupe})}$$

puis mesure de l'accuracy des pseudo-étiquettes sur les points de plus forte marge.

**Résultats.**

| fraction gardée | accuracy | équilibre des classes |
|---|---|---|
| 10 % | 80.67 | 74, 26, 50 |
| 20 % | 81.67 | 136, 51, 113 |
| **30 %** | **82.22** | 189, 89, 172 |
| 50 % | 80.40 | 288, 189, 273 |
| 80 % | 72.75 | 367, 460, 373 |
| 100 % | 66.20 | 380, 725, 395 |

**Conclusion.** Les erreurs sont concentrées sur les frontières. Sur les 30 % les plus centraux, le
transport est juste à 82.22 contre 70.40 pour le classifieur. Un apprentissage auto-supervisé guidé
par le transport devient donc viable, à condition de n'apprendre que sur ce sous-ensemble.

**Deux réserves.** La sélection déséquilibre les classes, il faut des quotas par classe. Et il reste
à vérifier si le classifieur n'est pas lui aussi à 82 sur ces mêmes points, auquel cas rien de
nouveau n'est appris.

**Amendement (08/10, section 15).** La seconde réserve est tranchée. Sur les 30 % les plus sûrs, le
classifieur est juste à 89.8 %, contre 82.0 % pour ce filtre. La conclusion « apprentissage guidé par
le transport viable » ne tient pas.

---

## 7. Transport puis réentraînement

**Pourquoi.** C'est l'étape 8 de la formulation initiale : transporter les points sources vers la
géométrie de la cible, puis réentraîner un classifieur sur ces points transportés.

**Résultats.**

| classifieur réentraîné | accuracy | écart au transfert direct |
|---|---|---|
| 1 plus proche voisin | 66.40 | −4.00 |
| 5 plus proches voisins | 66.40 | −4.00 |
| régression logistique | 66.93 | −3.47 |
| ridge, alpha = 10 | 67.53 | −2.87 |

**Conclusion.** Toutes les variantes perdent. L'étape 8 est à abandonner telle quelle. Ce qui sauve
la voie géométrique est la sélection par marge, pas le choix du classifieur.

---

## 8. Bras C, D et E : retirer le classifieur de l'espace fusionné

**Pourquoi.** Le classifieur de tâche trace des frontières rigides dans l'espace même que la
partition doit lire. Peut-être abîme-t-il la géométrie qu'on cherche ensuite à exploiter.

**Ce qui a été fait.** Trois bras avec la perte de tâche à zéro, donc aucun classifieur entraîné sur
la fusion. Un contrastif supervisé a été **ajouté sur l'espace fusionné**, sans quoi plus rien
n'entraînerait la couche MLB et le barycentre, qui n'a aucune répulsion entre classes, l'écraserait
sur un point unique.

| bras | têtes unimodales | alignement inter-modal | barycentre |
|---|---|---|---|
| C | supervisées | pondéré par la cohérence | éteint |
| D | supervisées | pondéré par la cohérence | actif |
| E | supprimées | InfoNCE nu | actif |

**Précision de vocabulaire, importante pour la rédaction.** Le bras C n'est pas « sans classifieur ».
Ses têtes unimodales sont bien des classifieurs, supervisés par entropie croisée. Ce qui est absent,
c'est un classifieur sur l'**espace fusionné**, c'est-à-dire précisément l'espace que l'on partitionne.

**Résultats.**

| époque | bras C | bras D | bras E |
|---|---|---|---|
| 0 | 68.27 | 69.40 | 68.53 au pas 3000 |
| 1 | **73.60** | **72.47** | 65.40 en fin |
| 2 | 70.67 | 70.47 | — |
| 3 | 69.27 | 70.40 | — |

Trajectoire fine du bras E, mesurée toutes les 750 pas : 39.60, 44.87, 65.27, **68.53**, 67.53, puis
67.53, 65.73, 67.27, 66.40, 65.40.

**Plan factoriel.**

| | barycentre éteint | barycentre actif |
|---|---|---|
| têtes actives | C = 73.60 | D = 72.47 |
| têtes retirées | non mesuré | E = 68.53 |

- **Effet du barycentre**, à têtes fixées : $D - C = -1.13$. Nul, dans le bruit.
- **Effet des têtes et de la pondération**, à barycentre fixé : $D - E = +3.94$.
- Sous additivité, la case manquante vaudrait $C - D + E = 69.66$.

**Conclusions.**

1. Retirer le classifieur de l'espace fusionné fait passer le transport de 66.20 à 73.60.
2. Le barycentre différentiable est neutre, 3e mesure concordante.
3. Les têtes unimodales et leur pondération valent environ 4 à 5 points. **Première contribution
   positive isolée du projet.**
4. La règle qui en sort : superviser en amont au niveau des modalités, garder la fusion purement
   géométrique, ne jamais classifier l'espace que l'on partitionne. Les 3 chiffres l'appuient dans
   l'ordre : 66.20, 68.53, 73.60.
5. Le profil pic puis chute persiste dans le bras E, qui n'a **aucune** entropie croisée nulle part.
   L'explication par la saturation des logits est donc réfutée.

**Biais connu, à documenter.** Les bras A et B n'avaient pas le contrastif sur la fusion, qui
n'existait pas encore. L'écart de 66.20 à 73.60 confond donc le retrait de la perte de tâche avec
l'ajout de ce terme. Un bras avec tâche active **et** contrastif fusionné actif isolerait
proprement la perte de tâche.

---

## 9. Généralisation du bras C à Yelp et TumEmo

**Pourquoi.** Tout reposait sur Twitter, qui est de loin la cible la plus facile : la référence
M2CAN y vaut 86.40, contre 61.20 sur Yelp et 64.87 sur TumEmo. Il fallait savoir si le 73.60 du
bras C relevait de la méthode ou d'une particularité de Twitter.

**Ce qui a été fait.** Configuration strictement identique au bras C : perte de tâche à 0, têtes
unimodales supervisées, alignement inter-modal pondéré, contrastifs inter-sources et sur la fusion,
barycentre éteint. Seule la cible change. Comme le pic est inconnu sur une cible nouvelle : 3 époques
et une évaluation toutes les 750 pas, soit 15 points de mesure par run. Chaque évaluation loggue en
plus 2 indicateurs de pureté calculés **sans étiquette** sur la partition spectrale : le coefficient
de silhouette et le ratio des distances moyennes intra-groupe sur inter-groupe.

| run | sources | cible | durée |
|---|---|---|---|
| C-Yelp | TumEmo, Twitter | Yelp | 9.27 h |
| C-TumEmo | Twitter, Yelp | TumEmo | 9.04 h |

**Résultats, cible Yelp.**

| moment | accuracy | plafond oracle | silhouette | intra / inter | plus petit groupe |
|---|---|---|---|---|---|
| ép. 0 + 750 | 31.60 | 36.27 | 0.1130 | 0.8188 | 361 |
| ép. 0 + 1500 | 56.07 | 56.07 | 0.4268 | 0.2500 | 291 |
| **ép. 0 + 2250** | **60.07** | 60.07 | 0.5233 | **0.1727** | 354 |
| ép. 0 + 3000 | 58.67 | 58.67 | 0.5029 | 0.2115 | 204 |
| ép. 0 + 3750 | 58.27 | 58.27 | 0.5259 | 0.2101 | 261 |
| ép. 1 + 750 | 57.53 | 57.53 | 0.5609 | 0.2015 | 176 |
| ép. 1 + 1500 | 51.40 | 52.47 | 0.5737 | 0.1980 | 37 |
| ép. 1 + 2250 | 51.40 | 52.80 | **0.5862** | 0.1924 | 33 |
| ép. 1 + 3000 | 52.00 | 52.00 | 0.5605 | 0.2042 | 63 |
| ép. 1 + 3750 | 52.47 | 52.47 | 0.5630 | 0.2137 | 63 |
| ép. 2 + 750 | 52.33 | 52.33 | 0.5458 | 0.2401 | 64 |
| ép. 2 + 1500 | 51.87 | 51.87 | 0.5359 | 0.2544 | 58 |
| ép. 2 + 2250 | 50.07 | 51.47 | 0.5118 | 0.2750 | 37 |
| ép. 2 + 3000 | 50.47 | 50.47 | 0.4872 | 0.3139 | 53 |
| ép. 2 + 3750 | 50.87 | 51.00 | 0.4785 | 0.3331 | 53 |

**Résultats, cible TumEmo.**

| moment | accuracy | plafond oracle | silhouette | intra / inter | plus petit groupe |
|---|---|---|---|---|---|
| ép. 0 + 750 | 36.80 | 39.73 | 0.1150 | 0.8207 | 366 |
| ép. 0 + 1500 | 51.67 | 51.67 | 0.1589 | 0.6922 | 301 |
| ép. 0 + 2250 | 49.07 | 49.07 | 0.4893 | 0.2134 | 229 |
| ép. 0 + 3000 | 52.93 | 52.93 | 0.5573 | 0.1742 | 267 |
| ép. 0 + 3750 | 56.20 | 56.20 | 0.6192 | 0.1593 | 276 |
| ép. 1 + 750 | 55.73 | 55.73 | 0.6786 | 0.1604 | 291 |
| ép. 1 + 1500 | 56.20 | 56.20 | 0.6827 | 0.1692 | 297 |
| ép. 1 + 2250 | 57.07 | 57.07 | 0.6931 | 0.1653 | 317 |
| **ép. 1 + 3000** | **57.87** | 57.87 | 0.7167 | **0.1537** | 321 |
| ép. 1 + 3750 | 56.93 | 56.93 | 0.7175 | 0.1598 | 319 |
| ép. 2 + 750 | 57.67 | 57.67 | 0.7178 | 0.1583 | 322 |
| ép. 2 + 1500 | 56.53 | 56.53 | **0.7291** | 0.1551 | 315 |
| ép. 2 + 2250 | 57.60 | 57.60 | 0.7161 | 0.1600 | 318 |
| ép. 2 + 3000 | 57.67 | 57.67 | 0.7115 | 0.1646 | 319 |
| ép. 2 + 3750 | 57.13 | 57.13 | 0.7178 | 0.1640 | 317 |

**Comparaison aux références.**

| cible | bras C, meilleur point | référence M2CAN | écart |
|---|---|---|---|
| Twitter, 4 mesures | 73.60 | 86.40 | −12.80 |
| Yelp, 15 mesures | 60.07 | 61.20 | −1.13 |
| TumEmo, 15 mesures | 57.87 | 64.87 | −7.00 |

**Conclusions.**

1. **Le résultat n'est pas une particularité de Twitter.** Sur Yelp, le pipeline sans classifieur
   sur la fusion arrive à 1.13 point de la référence M2CAN, écart inférieur au bruit d'une
   différence. Sur TumEmo il reste 7.00 points dessous.
2. **Un critère d'arrêt non supervisé fonctionne.** Le minimum du ratio intra sur inter tombe
   **exactement** sur le meilleur point d'accuracy, sur les 2 cibles : 0.1727 sur Yelp au pas 2250
   de l'époque 0, et 0.1537 sur TumEmo au pas 3000 de l'époque 1. La silhouette, elle, ne convient
   **pas** : son maximum désigne un point à 51.40 sur Yelp, soit 8.67 points sous le meilleur, parce
   qu'elle continue de monter pendant l'effondrement d'un groupe. Sur TumEmo elle perd 1.34 point.
3. **Conséquence méthodologique.** Sur ces 2 runs, le meilleur point rapporté est aussi celui que
   choisirait le critère sans étiquette. Le chiffre ne dépend donc pas d'un coup d'œil aux vraies
   étiquettes pour choisir l'époque.
4. **L'optimalité de l'appariement a 2 exceptions** : le tout premier point, sur un espace encore
   aléatoire, et les partitions dégénérées. Voir l'amendement de la section 5.
5. **Le moment du pic dépend de la cible** : 60 % de l'époque 0 sur Yelp, 80 % de l'époque 1 sur
   TumEmo, fin de l'époque 1 sur Twitter. Évaluer finement était nécessaire.
6. **Deux modes d'échec différents.** Sur Yelp, un groupe s'effondre de 354 à 33 points après le pic
   et l'accuracy perd 9.20 points. Sur TumEmo, le plus petit groupe se stabilise vers 320 points alors
   que chaque classe en compte 500 : une classe reste durablement mal séparée, ce qui bloque le
   plafond vers 58.

**Réserves.**

- Le transfert direct du classifieur n'a **pas** été mesuré sur Yelp et TumEmo, faute de bras A ou B
  sur ces cibles. On ne sait donc pas si le bras C bat le classifieur hors de Twitter.
- Une seule graine. Inférence transductive, comme partout ailleurs.
- Le bras C sur Twitter n'a que 4 mesures, sa résolution n'est pas comparable aux 15 mesures ici.
- Le critère intra sur inter est validé sur 2 cibles seulement, et n'a pas été loggué pour Twitter.

---

## 10. HAC (vidéo + audio) : le cadre M2CAN et le bras C hors sentiment

**Pourquoi.** L'encadrant a fourni HAC (Human-Animal-Cartoon, jeu de SimMMDG, NeurIPS 2023) : 3 domaines,
7 actions, vidéo + audio + flot optique. Question : le cadre M2CAN, et le bras C posé sur ce cadre,
tiennent-ils sur un benchmark multimodal à 3 domaines qui n'est pas du sentiment ?

**Ce qui a été fait.**
- Protocole leave-one-domain-out : 2 sources étiquetées → 1 cible, train cible sans étiquette, évaluation
  sur le test officiel de la cible (176 à 276 clips). K = 7.
- Paire **vidéo + audio** : le cadre est bimodal par construction (MLB bilinéaire, ExpVar entre 2 têtes).
  Slot image ← vidéo, slot texte ← audio.
- **Backbones gelés**, features extraites une fois. Vidéo : ResNet-50 ImageNet de M2CAN sur 8 frames moyennées
  (variante fidèle au cadre) **ou** SlowFast R101 Kinetics-400 (l'encodeur de SimMMDG ; 2304 → 2048 par une
  ACP fixe ajustée sur les seules sources). Audio : AST AudioSet (768). Une vue faible et 4 vues fortes par
  clip, avec les transformations de M2CAN.
- **Aucune ligne du code de référence modifiée** : adaptateur `hac/` injecté à l'exécution (entrée des
  modèles et chargeurs). Deux contournements côté entrée seulement : vue dédoublonnée pour `validate_pseudo`
  (la cible est répétée pour remplir 15 000 exemples par époque), pas de lot singleton en évaluation
  (`.squeeze()` de `forward_test`).
- M2CAN : hyperparamètres publiés (`config_Twitter.yaml`), 10 époques de 3 750 pas. Bras C : drapeaux
  canoniques (task_rate 0, têtes modales, CMCFA pondéré ExpVar, CDCFA + SupCon fusion, sans WBHR),
  3 époques, évaluation HOT-DA tous les 750 pas.
- 3 graines (123456, 234567, 345678), 36 runs, kernel `hac-full`. Plancher : régression logistique
  source-only sur les mêmes features (2 sources → test cible), sans adaptation.

**Résultats.** Accuracy sur le test cible, moyenne ± écart-type sur 3 graines.

| encodeur | cible | plancher source-only | M2CAN best | M2CAN last | bras C best | bras C last | bras C critère intra/inter |
|---|---|---|---|---|---|---|---|
| ResNet-50 | human | 61.90 | **68.72 ± 1.37** | 68.36 ± 1.37 | 49.03 ± 2.18 | 47.71 ± 2.93 | 47.71 ± 3.12 |
| ResNet-50 | animal | 70.45 | **80.49 ± 0.87** | 80.49 ± 0.87 | 67.80 ± 6.73 | 66.67 ± 5.25 | 66.10 ± 5.91 |
| ResNet-50 | cartoon | 49.08 | **63.61 ± 1.40** | 62.84 ± 1.21 | 49.54 ± 2.43 | 46.94 ± 3.71 | 44.80 ± 1.61 |
| SlowFast | human | 76.59 | **85.99 ± 3.48** | 85.75 ± 3.29 | 69.32 ± 1.46 | 64.98 ± 2.41 | 63.89 ± 2.72 |
| SlowFast | animal | 80.68 | **84.85 ± 0.87** | 84.47 ± 1.43 | 67.23 ± 2.30 | 65.53 ± 1.43 | 65.72 ± 0.87 |
| SlowFast | cartoon | 50.92 | **72.94 ± 2.00** | 71.25 ± 2.17 | 52.75 ± 2.00 | 50.46 ± 2.55 | 49.08 ± 4.66 |

Diagnostic du bras C (11 runs) : HOT-DA ≈ oracle (écart moyen 2.9 points, max 10.5), oracle entre 50 et
76 %, partitions déséquilibrées (human, classes équilibrées à ~39 clips : groupes de 97, 79, 72).
Courbes (`results/hac_m2can_brasc/`) : M2CAN monte sur ses 10 époques, son dernier point ≈ son meilleur ;
le bras C plafonne dès ~5 000 pas. CMCFA reste plate (~4) pour les deux méthodes.

**Conclusion.**
- M2CAN bat le bras C partout (+12 à +20 points) et dépasse le plancher source-only (+4 à +22 points) :
  son adaptation fonctionne aussi hors sentiment.
- Le bras C passe **sous le plancher source-only** dans 4 cas sur 6 : sa lecture par HOT-DA ne tient pas à
  7 classes. Le goulot principal reste le regroupement de la cible : même avec un appariement parfait,
  l'oracle ne dépasse pas 50 à 76 %. À la différence du sentiment, où il est optimal au pic,
  l'appariement perd ici en plus 2.9 points en moyenne, jusqu'à 10.5. Le tout est aggravé par K = 7
  et ~200 points cibles.
- Un encodeur vidéo dédié aide nettement les deux méthodes (SlowFast : M2CAN +17 points sur human,
  +9 sur cartoon).

**Réserves.**
- Backbones gelés : les LR publiés (extracteurs à 2e-6, prévus pour un fine-tuning) figent quasiment les
  extracteurs, d'où CMCFA plate. Ça peut pénaliser davantage le bras C, méthode contrastive. À trancher par
  un run end-to-end ou des LR d'extracteurs relevés.
- Drapeaux du bras C calibrés pour K = 3 : `hotda_src_batches 40` ne donne que ~23 exemples par classe et
  par source pour K = 7.
- « best » est choisi sur le test cible (HAC n'a pas de split de validation) : optimiste pour les deux
  méthodes, d'où les colonnes last et critère.
- Tests cibles petits (176 à 276 clips) : environ ±3 points de bruit par graine.
- Le split officiel human contient 24 clips présents à la fois en train et en test (1 387 lignes,
  1 363 clips distincts).

---

## 11. Tableau de synthèse

| configuration | mode de décision | accuracy |
|---|---|---|
| référence M2CAN publiée (Tableau 1 de l'article) | classifieur | 84.7 |
| notre vérification du checkpoint officiel | classifieur | 84.47 |
| bras C, graine 123456 | transport hiérarchique | 73.60 |
| bras D | transport hiérarchique | 72.47 |
| bras B | classifieur transféré | 70.40 |
| **bras C, 3 graines (section 13)** | transport hiérarchique | **70.22 ± 2.97** |
| bras A | classifieur transféré | 70.00 |
| bras E | transport hiérarchique | 68.53 |
| bras B, transport après coup | transport hiérarchique | 66.20 |
| plancher source seule | classifieur transféré | 59.40 |
| classe majoritaire | — | 33.33 |

Seul le bras C a été répété sur 3 graines. Les autres lignes reposent sur une seule graine, avec
environ ±3.3 points de bruit sur un écart (section 0).

**Par cible, bras C.** Moyenne ± écart-type sur 3 graines (section 13).

| cible | graine 123456 | meilleur point | point choisi par le critère intra / inter | référence M2CAN publiée | notre vérification du checkpoint |
|---|---|---|---|---|---|
| Twitter | 73.60 | 70.22 ± 2.97 | 66.13 (2 graines, non loggué pour 123456) | 84.7 | 84.47 |
| Yelp | 60.07 | 58.80 ± 2.73 | 58.49 | 63.7 | 61.20 |
| TumEmo | 57.87 | 55.31 ± 2.21 | 54.84 | 63.8 | 65.00 |

**HAC, vidéo + audio, backbones gelés (section 10).** Meilleur point, moyenne de 3 graines.

| encodeur | cible | plancher source-only | M2CAN | bras C |
|---|---|---|---|---|
| ResNet-50 | human / animal / cartoon | 61.90 / 70.45 / 49.08 | 68.72 / 80.49 / 63.61 | 49.03 / 67.80 / 49.54 |
| SlowFast | human / animal / cartoon | 76.59 / 80.68 / 50.92 | 85.99 / 84.85 / 72.94 | 69.32 / 67.23 / 52.75 |

L'auto-apprentissage sur caractéristiques gelées, avec ou sans OT, est en section 14. Son protocole
diffère (10 sous-échantillons, dernier tour), ses chiffres ne se comparent pas case à case à ceux-ci.

**Ce que l'ensemble établit, au 09/10.** Synthèse des interprétations des sections 1 à 16.

1. **Le goulot est la partition de la cible, pas le transport.** Sur le sentiment, l'appariement
   atteint son plafond oracle au pic (sections 9 et 13). Sur HAC à K = 7, l'oracle lui-même plafonne
   entre 50 et 76 %, et l'appariement perd en plus 2.9 points en moyenne (section 10). Remplacer la
   partition spectrale par un OT équilibré vers les barycentres sources récupère 2.87 points sur
   Twitter, mais ne fait que rejoindre le classifieur (section 15).
2. **Les têtes unimodales supervisées et leur pondération valent environ 4 points** (D − E = +3.94,
   section 8). C'est le seul effet positif isolé de la lignée des bras A à E, mesuré sur une graine,
   à répliquer.
3. **Le bras C ne bat ni le classifieur transféré ni M2CAN.** Sur 3 graines, il vaut 70.22 ± 2.97 sur
   Twitter, le niveau du bras B (section 13). Sur HAC, il passe sous le plancher source seule dans
   4 cas sur 6 (section 10).
4. **Utilisé comme étiqueteur, l'OT ne fait pas mieux que la confiance du classifieur** à nombre égal
   de points, sur HAC (60 runs) comme sur Twitter (sections 14 et 15). Son seul effet mesuré passe
   par les effectifs retenus dans chaque classe. Cet effet change de signe selon la configuration, et
   il est plus favorable avec l'encodeur le plus faible (ResNet-50).
5. **L'auto-apprentissage simple est une ligne de base forte.** Sur HAC, il gagne 8.11 points sur le
   plancher en moyenne sur 10 graines, et approche M2CAN avec SlowFast (section 14). Toute méthode doit
   s'y comparer.
6. **EM et BBSE échouent à estimer les proportions cibles sous décalage de domaine** (L1 0.56 et 0.53,
   contre 0.18 en reprenant les proportions sources, section 14). Ils supposent un pur décalage
   d'étiquettes, ce qui est faux ici.
7. **Méthodologie.** Une graine ne suffit pas : 73.60 est devenu 70.22. Choisir le meilleur point sur
   la cible gonfle les chiffres. Comparer à couverture inégale peut inverser une conclusion (section 6
   amendée). Le critère intra / inter est un substitut utilisable, à 0.89 point du meilleur point en
   moyenne (section 13).
8. **Aucune règle de sélection ne bat la confiance du classifieur fusionné.** Quatre ont été testées à
   nombre égal de points : OT sur les postérieurs, OT géométrique, accord entre sources (l'UACR de M2CAN),
   accord entre modalités (sections 14 à 16). Sur HAC, l'accord entre modalités coûte même 8 points de
   précision : les erreurs vidéo et audio sont presque indépendantes, mais chaque modalité seule est trop
   faible. Les modalités y sont complémentaires, pas redondantes.

---

## 12. Points ouverts

1. **Graines.** Résolu pour le bras C (section 13) : 70.22 ± 2.97 sur Twitter, ce qui efface son
   avance sur le bras B. Les autres bras restent sur une seule graine, y compris l'effet du barycentre.
2. **Autres cibles.** Résolu en partie (section 9) : Yelp et TumEmo mesurés pour le bras C. Les bras
   avec classifieur transféré n'y ont pas été rejoués, donc on ignore si le bras C bat le classifieur
   hors de Twitter.
3. **Provenance de la référence.** Tranché : Tableau 1 de l'article, 84.7 / 63.7 / 63.8 (section 1).
4. **Même découpage de test ?** Nos 1500 points viennent de `split_full`. À confirmer.
5. **Décomposition du gain contrastif.** Les 10.6 points attribués aux pertes contrastives reposent
   sur une comparaison à un plancher obtenu dans un autre protocole.
6. **Courbe côté source.** Jamais mesurée. Sans elle, impossible de distinguer un sur-apprentissage
   des sources d'une simple instabilité.
7. **Aucun terme cible dans l'objectif.** Les 4 termes optimisés ne concernent que les sources, ou la
   cible sans étiquette et sans notion de classe. Le résultat obtenu est un produit dérivé.
8. **Effondrement d'un groupe sur Yelp.** Le plus petit groupe cible passe de 354 à 33 points après
   l'époque 0, et l'accuracy perd 9.20 points. Cause inconnue.
9. **Critère d'arrêt non supervisé.** Testé sur 8 runs (section 13) : 3 fois exactement le meilleur
   point, 4 fois à moins de 1.4 point, 1 fois à 4.20 points. Utilisable, pas exact.

10. **HAC et le gel des backbones.** Vérifier que l'écart M2CAN / bras C ne vient pas du gel : run
   end-to-end (ResNet-50 + AST fine-tunés) ou LR d'extracteurs relevés en mode gelé.
11. **Bras C à K = 7.** La partition spectrale de la cible plafonne (oracle 50 à 76 % sur HAC). Pistes :
   affectation OT équilibrée vers les barycentres sources (§0bis de A_FAIRE), plus d'exemples sources
   dans les barycentres. L'affectation OT équilibrée a été testée sur Twitter (section 15) : +2.87
   points sur la partition spectrale, au niveau du classifieur. Pas encore testée sur HAC.
12. **OT et auto-apprentissage.** L'effet de l'OT change de signe selon la configuration (section 14).
   Hypothèse à tester : il protège l'auto-apprentissage quand le classifieur de départ est faible, et
   gêne quand il est bon.
13. **OT comme étiqueteur.** À couverture égale, il ne bat la confiance du classifieur ni sur HAC
   (60 runs, section 14) ni sur Twitter (section 15). Sa seule contribution mesurée passe par les
   effectifs retenus dans chaque classe.
14. **Préparé ou proposé, non lancé.**
   - Bras C sur l'esthétique (AVA, PCCD, RPCD) : kernels prêts, bloqués par le quota de Google Drive,
     puis mis en pause avec le changement de voie.
   - HAC de bout en bout, et HAC à 3 modalités avec le flot optique.
   - EPIC-Kitchens, avec les découpes de MM-SADA.
15. **Lecture du survey de Dong et al.** (TPAMI, arXiv 2501.18592) faite le 09/10. Elle a conduit à la
   proposition C (section 16).
16. **L'UACR sans son facteur d'incertitude.** Dans la simulation de la section 16, l'accord entre
   classifieurs de sources dégrade la précision des pseudo-étiquettes, sans gain final. Ablation à faire dans
   M2CAN entraîné de bout en bout.
17. **La proposition C sur le sentiment.** Les exports du bras B contiennent déjà, pour chaque point, le poids
   w = exp(−KL) entre les têtes texte et image. Le test C1 y est faisable dans un kernel CPU.

---

## 13. Bras C sur 3 graines (sentiment)

**Pourquoi.** L'avance du bras C sur le bras B (73.60 contre 70.40) et son écart à M2CAN sur Yelp
(−1.13) reposaient sur la seule graine 123456, sous le seuil de 3.3 points de la section 0.

**Ce qui a été fait.** Drapeaux canoniques du bras C, inchangés. Graines 234567 et 345678 sur les
3 cibles, soit 6 runs répartis sur 3 comptes Kaggle (kernels `brasc-<cible>-<graine>`, données
téléchargées par l'API Kaggle depuis le kernel). 2 époques, une évaluation toutes les 750 pas, soit
10 mesures par run. 5.4 à 6.5 h par run.

**Résultats.** Meilleur point, choisi sur la cible.

| cible | 123456 | 234567 | 345678 | moyenne ± écart-type | M2CAN publié | écart | M2CAN remesuré | écart |
|---|---|---|---|---|---|---|---|---|
| Twitter | 73.60 | 68.00 | 69.07 | **70.22 ± 2.97** | 84.7 | −14.48 | 84.47 | −14.25 |
| Yelp | 60.07 | 55.67 | 60.67 | **58.80 ± 2.73** | 63.7 | −4.90 | 61.20 | −2.40 |
| TumEmo | 57.87 | 54.07 | 54.00 | **55.31 ± 2.21** | 63.8 | −8.49 | 65.00 | −9.69 |

Point choisi par le minimum du ratio intra sur inter, sans étiquette. Entre parenthèses, l'écart au
meilleur point du même run.

| cible | 234567 | 345678 |
|---|---|---|
| Twitter | 63.80 (−4.20) | 68.47 (−0.60) |
| Yelp | 55.67 (0) | 59.73 (−0.93) |
| TumEmo | 54.00 (−0.07) | 52.67 (−1.33) |

Dans les 6 runs, le plafond oracle égale le meilleur point : l'appariement hongrois est optimal au pic.

**Conclusions.**

1. **Le 73.60 était une graine favorable.** Sur 3 graines, le bras C vaut 70.22 ± 2.97 sur Twitter,
   le niveau du bras B (70.40, une graine). Son avance sur le classifieur transféré n'est plus établie.
2. **Yelp :** 58.80 ± 2.73, à 4.90 points de la valeur publiée de M2CAN et à 2.40 points de son
   checkpoint remesuré. Le « 1.13 point » de la section 9 tenait à la graine.
3. **TumEmo :** 55.31 ± 2.21, à 8.49 points de la valeur publiée de M2CAN (9.69 de son checkpoint
   remesuré).
4. **Critère d'arrêt sans étiquette.** Sur les 8 runs où il est loggué (2 de la section 9, 6 ici) :
   3 fois exactement le meilleur point, 4 fois à moins de 1.4 point, 1 fois à 4.20 points. Écart
   moyen : 0.89 point.

**Réserves.**

- Le meilleur point est choisi sur la cible, donc optimiste. Ces runs ont 10 mesures, la graine
  123456 n'en avait que 4 sur Twitter. Le biais joue en faveur des nouvelles graines, qui sont
  pourtant plus basses.
- 2 époques ici contre 3 pour la graine 123456 sur Yelp et TumEmo. Sans effet sur le meilleur point :
  tous les pics de la graine 123456 tombent avant la fin de l'époque 1.
- Le bras B n'a qu'une graine. La comparaison avec le bras C n'est pas symétrique.

---

## 14. L'OT comme étiqueteur de la cible pour l'auto-apprentissage (HAC, caractéristiques gelées)

**Pourquoi.** Changement de voie décidé le 07/10, en restant dans l'OT. Le bras C lit la classe dans
une partition de la cible. Ici, l'OT étiquette la cible et un classifieur se réentraîne sur ces
pseudo-étiquettes. Deux questions :

1. L'OT donne-t-il des pseudo-étiquettes plus justes que la confiance du classifieur, à nombre égal ?
2. Améliore-t-il l'auto-apprentissage ?

**Ce qui a été fait.**

- **Tour exploratoire, 07/10.** Une graine, 2 scripts (`hac/ot_selftrain.py` puis
  `hac/ot_selftrain_v2.py`, résultats dans `results/hac_ot_selftrain/`). Régression logistique sur
  vidéo + audio gelés, standardisés sur les sources. 5 tours au calendrier de l'UACR : 30, 60, 90,
  100 et 100 % de la cible, avec des quotas par classe. Dernier tour rapporté. Résultats et lecture
  détaillés juste après cette liste.
- **Protocole figé, 08/10.** `hac/ot_protocole_10graines.py`, résultats dans `results/hac_ot_protocole/`.
  Critère écrit dans `protocole.json` avant tout résultat, et aucune nouvelle variante :
  - Critère primaire : la moyenne des écarts appariés `ot_mix_em − c0_naive` est positive et son
    IC95 % exclut 0. L'IC vient d'un bootstrap des graines dans chaque configuration, 10 000 tirages.
  - Critère secondaire : le taux de runs sous le plancher source seule.
  - Graine : sous-échantillon de 90 % de chaque train source et du train cible. Le test cible reste
    complet.
  - 2 encodeurs vidéo × 3 cibles = 6 configurations, × 10 graines = 60 runs. Mesure au dernier tour.

| variante | sélection des pseudo-étiquettes |
|---|---|
| `c0_naive` | argmax et confiance du classifieur courant |
| `c0_em` | idem, postérieurs corrigés par EM (Saerens 2002) : témoin de l'estimation des proportions sans OT |
| `ot_mix_em` | plan de Sinkhorn, coût ½ postérieurs + ½ distance aux prototypes de classe, marges cibles estimées par EM |

**Résultats du tour exploratoire, une graine.** Moyennes sur les 6 configurations.

Premier script : l'OT lit-il ou fabrique-t-il de bonnes étiquettes ?

| méthode | accuracy moyenne |
|---|---|
| plancher source seule | 65.04 |
| lecteur OT : test cible affecté aux prototypes sources par Sinkhorn, sans apprentissage | 65.16 |
| OT unique : train cible étiqueté par l'OT, un seul réentraînement | 69.19 |
| auto-apprentissage naïf, sans OT (`c0_naive`) | **72.88** |
| auto-apprentissage guidé par l'OT, marges uniformes (`c_equi`) | 71.74 |
| idem, marges relâchées (`c_relax`) | 71.52 |
| repères : M2CAN et bras C, meilleur point (section 10) | 76.10 et 59.28 |

Écart `c_equi − c0_naive` par configuration, dans l'ordre SlowFast human, animal, cartoon, puis
ResNet-50 human, animal, cartoon : −1.8, −5.7, −6.4, +2.2, +6.2, −1.4.

Second script : 8 variantes, chacune avec son témoin sans OT. Écart à `c0_naive` : moyenne ± écart-type
sur les 6 configurations.

| variante | OT | marges | étiquettes | coût | accuracy | écart à `c0_naive` | configs > 0 |
|---|---|---|---|---|---|---|---|
| `c0_naive` | non | — | dures | — | 72.88 | — | — |
| `c0_em` | non | EM | dures | — | 71.77 | −1.11 ± 4.37 | 2/6 |
| `c0_soft` | non | EM | douces | — | 66.14 | −6.74 ± 9.93 | 1/6 |
| `ot_equi` | oui | uniformes | dures | postérieurs | 71.74 | −1.15 ± 4.79 | 2/6 |
| `ot_em` | oui | EM | dures | postérieurs | 72.90 | +0.01 ± 3.98 | 2/6 |
| `ot_em_soft` | oui | EM | douces | postérieurs | 68.48 | −4.40 ± 7.29 | 2/6 |
| `ot_mix_em` | oui | EM | dures | mixte | **74.36** | +1.48 ± 4.97 | 3/6 |
| `ot_full` | oui | EM | douces | mixte | 49.96 | −22.92 ± 17.00 | 1/6 |

Lecture du tour exploratoire :

1. **L'auto-apprentissage seul fait l'essentiel** : +7.84 points sur le plancher, sans aucun OT.
2. **Lire la cible par OT, sans apprendre, ne rapporte rien** (65.16 contre 65.04). Étiqueter une fois
   par OT puis réentraîner rapporte +4.15 points. Itérer sans OT fait mieux.
3. **L'équipartition, façon SeLa, impose de fausses proportions de classes.** Elle coûte 5.7 et
   6.4 points sur SlowFast animal et cartoon. Elle ne rapporte que là où l'auto-apprentissage naïf
   dérive : ResNet-50 animal, +6.2.
4. **Les étiquettes douces nuisent à chaque fois.** La proposition complète (`ot_full`) s'effondre,
   jusqu'à 24.31 % sur SlowFast cartoon.
5. **Corriger les proportions par EM sans OT n'aide pas** (−1.11). EM estime des proportions fausses :
   certaines classes de cartoon sont estimées à 0 dès le premier tour.
6. **`ot_mix_em` arrive en tête** (+1.48), mais avec 4.97 d'écart-type sur 6 configurations, soit ±2.0
   d'erreur-type, et après un choix parmi 6 variantes OT fait en regardant le test. Il gagne sur les
   3 configurations ResNet-50 et perd ou égalise sur les 3 SlowFast. D'où le protocole figé.

**Résultats du protocole figé.** Moyenne sur 10 graines, test cible, dernier tour. Écart = `ot_mix_em − c0_naive`, IC95 %
par bootstrap des graines de la configuration.

| encodeur / cible | plancher | `c0_naive` | `c0_em` | `ot_mix_em` | écart [IC95 %] | graines où l'OT gagne |
|---|---|---|---|---|---|---|
| ResNet-50 / animal | 68.69 | 69.49 | 70.68 | 74.49 | **+5.00** [+3.64, +6.53] | 10/10 |
| ResNet-50 / cartoon | 50.09 | 58.30 | 59.17 | 60.14 | **+1.83** [+0.96, +2.94] | 9/10 |
| ResNet-50 / human | 62.28 | 72.86 | 71.59 | 72.21 | −0.65 [−1.20, 0.00] | 2/10 |
| SlowFast / animal | 80.11 | 84.03 | 82.27 | 83.13 | −0.91 [−3.18, +1.59] | 3/10 |
| SlowFast / cartoon | 52.11 | 70.73 | 70.28 | 68.90 | **−1.83** [−2.75, −0.92] | 1/10 |
| SlowFast / human | 76.59 | 83.15 | 82.75 | 84.64 | **+1.49** [+0.33, +2.50] | 7/10 |

| mesure globale | valeur |
|---|---|
| `ot_mix_em − c0_naive`, critère primaire | **+0.82** [+0.28, +1.39] : **satisfait** |
| `c0_em − c0_naive` | −0.30 [−0.81, +0.23] |
| `ot_mix_em − c0_em` | +1.12 [+0.79, +1.47] |
| runs sous le plancher, sur 60 | `c0_naive` 3, `c0_em` 2, `ot_mix_em` 1 |
| précision des pseudo-étiquettes à nombre égal par classe, OT − confiance | +0.21 point, OT meilleur dans 57 % des 300 cas |
| erreur L1 sur les proportions cibles | proportions sources 0.18, EM 0.56, BBSE 0.53, Sinkhorn semi-relâché 0.20 |

**Analyse de sensibilité, non pré-enregistrée.**

- 3 configurations sur 6 sont positives, 1 significativement négative (SlowFast / cartoon) et 1 à la
  limite (ResNet-50 / human, borne haute à 0.00).
- Au niveau des configurations (n = 6), t = 0.81 : non significatif.
- Sans ResNet-50 / animal, l'écart moyen tombe à −0.02.
- Précision à nombre égal par tour : −0.65, +0.61, +0.64, +0.40 et +0.04 point.

**Conclusions.**

1. **Le critère pré-enregistré est satisfait, mais il ne mesure que le bruit des graines à
   configurations fixées.** L'effet change de signe selon la configuration. Énoncé défendable : sur
   ces 6 configurations, l'OT améliore l'auto-apprentissage de 0.82 point en moyenne, et ce gain
   vient pour l'essentiel de ResNet-50 / animal. « L'OT aide l'auto-apprentissage » en général
   n'est pas établi.
2. **L'OT n'est pas un meilleur étiqueteur point par point.** À effectifs par classe égaux, ses
   pseudo-étiquettes ne sont pas plus justes que celles de la confiance (+0.21 point). Quand il aide,
   c'est par le nombre de points retenus dans chaque classe, pas par le choix de ces points.
3. **EM et BBSE estiment mal les proportions cibles sur HAC** (L1 0.56 et 0.53, contre 0.18 en
   reprenant simplement les proportions sources). Ils supposent un pur décalage d'étiquettes, que le
   décalage de domaine viole. Le Sinkhorn semi-relâché ramène l'erreur à 0.20, soit le niveau des
   proportions sources, sans faire mieux.
4. **Les gains vont aux configurations où l'auto-apprentissage naïf est fragile.** ResNet-50 / animal
   compte 2 runs naïfs sous le plancher, aucun avec l'OT, et c'est là que l'OT gagne 5 points. Avec
   SlowFast, où le classifieur de départ est bon, l'OT est neutre ou nuisible. C'est une hypothèse
   formée après coup : l'OT servirait de garde-fou contre la dérive de confirmation quand le
   classifieur est faible. Il faut la tester sur d'autres données avant de l'écrire.
5. **Ordre de grandeur.** L'auto-apprentissage naïf sur caractéristiques gelées atteint, avec
   SlowFast, 83.15 / 84.03 / 70.73 (human / animal / cartoon), contre 85.99 / 84.85 / 72.94 pour
   M2CAN et 69.32 / 67.23 / 52.75 pour le bras C (section 10). Avec ResNet-50 sur human, il dépasse
   M2CAN : 72.86 contre 68.72. Les protocoles diffèrent : M2CAN prend son meilleur point sur le test
   avec 3 graines, ici c'est le dernier tour avec 10 sous-échantillons. Cette ligne de base simple
   est forte et toute méthode future doit s'y comparer.

**Réserves.**

- Caractéristiques gelées et tête linéaire : rien ne dit que l'effet survit à un entraînement de bout
  en bout.
- Les 10 graines se partagent 90 % des mêmes données et le même test : même à configuration fixée,
  l'IC est sans doute trop étroit.
- Tests cibles petits : 176 à 276 clips.

---

## 15. L'OT comme étiqueteur sur Twitter (bras B, représentations figées)

**Pourquoi.** Refaire le test de la section 14 dans le cadre sentiment, sur l'espace fusionné
réellement appris, et trancher la seconde réserve de la section 6 : le classifieur n'est-il pas lui
aussi à 82 % sur les points que garde le filtre de marge ?

**Ce qui a été fait.** `analyses/ot_etiqueteur_twitter.py`, exécuté dans un kernel Kaggle CPU
(`ot-etiqueteur-twitter`, construit par `kaggle/ot_etiqueteur/build.py`). Le kernel récupère la
sortie de `m2can-diff-b-twitter` : les z (64-d) de l'état final du bras B (époque 3) et les poids des
2 classifieurs de source de ce même état. Sources TumEmo + Yelp (train, 30 000 points), cible Twitter
(train 15 000 points, val 1500 points à 500 par classe). Résultats dans `results/twitter_ot_etiqueteur/`.

**Contrôle.** Le classifieur recalculé hors GPU donne 69.00 sur Twitter val, contre 68.80 loggués
pendant l'entraînement, soit 3 prédictions sur 1500. Les z ont été exportés dans une autre passe que
la validation (lots de 32), l'écart est probablement numérique. Toutes les comparaisons ci-dessous
portent sur les mêmes z.

**Test 1. Précision des pseudo-étiquettes sur Twitter val, à couverture égale, quotas par classe.**

| étiqueteur | 10 % | 20 % | 30 % | 50 % | 100 % |
|---|---|---|---|---|---|
| classifieur | 85.33 | 82.33 | 83.56 | 80.80 | 69.00 |
| classifieur + EM | 85.33 | 82.33 | 84.00 | 80.67 | 69.20 |
| OT sur les postérieurs, marges uniformes | 86.67 | 84.00 | 83.78 | 80.80 | 69.07 |
| OT géométrique vers les barycentres sources, marges uniformes | 81.33 | 82.67 | 84.67 | 81.07 | 69.07 |
| OT géométrique + filtre de marge (section 6) | 81.33 | 81.33 | 79.78 | 79.33 | 69.07 |
| OT mixte + EM | 86.00 | 84.33 | 84.67 | 81.47 | 68.93 |
| HOT-DA spectral + filtre de marge (section 6) | 80.00 | 79.33 | 79.78 | 78.53 | 66.20 |

Sans quotas, à 30 % : classifieur 89.8, classifieur + EM 89.3, OT sur les postérieurs 89.1, OT
géométrique 90.2, OT géométrique + marge 84.4, OT mixte + EM 89.8, HOT-DA + marge 82.0.

**Test 0bis.1. Affectation complète de Twitter val.**

| affectation | espace brut | recentré + blanchi |
|---|---|---|
| partition spectrale + appariement (HOT-DA) | 66.20, groupes 380 / 725 / 395 | 44.47 |
| OT équilibré vers les barycentres sources | 69.07, groupes 498 / 500 / 502 | 58.73 |
| k-means ancré sur les sources + OT | 68.27 | 68.53 |

**Bonus. Auto-apprentissage sur z gelé, tête linéaire, calendrier de l'UACR.** Accuracy sur Twitter
val à chaque tour, précision des pseudo-étiquettes entre parenthèses.

| étiqueteur | tour 1 | tour 2 | tour 3 | tour 4 | tour 5 |
|---|---|---|---|---|---|
| classifieur seul | 68.60 (82.5) | 69.07 (77.4) | 69.40 (71.0) | 69.27 (68.5) | 69.07 (68.7) |
| OT sur les postérieurs | 68.60 (82.4) | 69.07 (77.7) | 69.47 (70.9) | 69.27 (68.6) | 68.93 (68.5) |
| OT mixte + EM | 68.87 (83.3) | 69.20 (77.7) | 69.40 (70.9) | 69.40 (68.9) | 69.40 (68.8) |

**Conclusions.**

1. **La réserve de la section 6 est tranchée : à couverture égale, le classifieur fait mieux.** À 30 %
   sans quotas, ses pseudo-étiquettes sont justes à 89.8 %, contre 82.0 % pour HOT-DA filtré par la
   marge (82.22 en section 6). Avec quotas : 83.56 contre 79.78. Le « 82.22 contre 70.40 » de la
   section 6 comparait un sous-ensemble choisi à l'ensemble complet.
2. **Aucun étiqueteur OT ne bat nettement le classifieur.** Les meilleurs, l'OT sur les postérieurs et
   l'OT mixte + EM, font de 0 à +2 points entre 10 et 50 %. C'est l'ordre de l'erreur-type d'une
   précision sur 150 à 750 points, soit 1.4 à 2.9 points. Même conclusion que sur HAC (section 14).
3. **Le filtre de marge géométrique nuit** une fois comparé à couverture égale : −4.89 points à 30 %
   pour l'OT géométrique.
4. **Le goulot de HOT-DA est la partition spectrale.** Remplacée par un OT équilibré vers les
   barycentres sources, l'affectation complète passe de 66.20 à 69.07. La contrainte de marges
   uniformes résorbe le groupe de 725 points, mais on ne fait que rejoindre le classifieur (69.00).
5. **Recentrer et blanchir la cible nuit** à la partition spectrale (−21.73) et à l'OT vers les
   barycentres (−10.33). Seul le k-means ancré sur les sources y résiste.
6. **L'auto-apprentissage sur z gelé est plat**, de 68.60 à 69.47 quel que soit l'étiqueteur. Sur un
   espace figé, réentraîner une tête linéaire sur ses propres pseudo-étiquettes ne déplace presque
   rien. Un gain éventuel ne peut venir que d'un réentraînement des encodeurs.

**Réserves.**

- Une graine, un état (fin de l'époque 3 du bras B), une cible.
- Twitter val sert à la fois d'ensemble étiqueté et de mesure.
- Pas d'IC apparié : le JSON ne garde que les précisions agrégées.

---

## 16. Proposition C1 : l'accord entre modalités pour choisir les pseudo-étiquettes (HAC, caractéristiques gelées)

**Pourquoi.** Dans l'UACR de M2CAN, le score d'un point cible vaut : confiance du classifieur fusionné ×
exp(−KL) entre les classifieurs des deux sources. Ces deux classifieurs lisent le même vecteur fusionné. La
proposition C1 les remplace par les têtes de modalités, qui lisent des entrées différentes. Elle s'appuie sur
le co-training (Blum et Mitchell, 1998) : des vues aux erreurs décorrélées peuvent se corriger mutuellement.

**Ce qui a été fait.** `hac/c_accord_protocole.py`, résultats dans `results/hac_c_accord/`. Critère écrit dans
`protocole.json` avant tout résultat, avec l'empreinte sha256 du script. Mêmes graines, mêmes sous-échantillons
et même calendrier que la section 14 : 6 configurations × 10 graines = 60 runs, régressions logistiques.
Contrôle : la règle « confiance » redonne exactement l'auto-apprentissage naïf de la section 14, avec un écart
maximal de 0 sur les 60 runs.

Les trois règles partagent la même étiquette (argmax du classifieur fusionné) et le même quota par classe,
donc le même nombre de points par classe. Seul le classement change. Les têtes auxiliaires sont
réentraînées à chaque tour, comme le classifieur fusionné.

| règle | score de classement |
|---|---|
| `confiance` | confiance du classifieur fusionné |
| `accord_sources` (UACR de M2CAN) | confiance × exp(−KL symétrique entre les classifieurs des 2 sources) |
| `accord_modalites` (C1) | confiance × exp(−KL symétrique entre la tête vidéo et la tête audio) |

Critère primaire : au tour 1, à 30 % de couverture, l'écart de précision `accord_modalites − confiance` est
positif, son IC95 % exclut 0, et il est positif dans au moins 5 configurations sur 6.

**Diagnostic : la condition du co-training.** Classifieurs source seule, moyenne sur 10 graines. Accuracy
sur le test cible ; corrélation phi des erreurs et accord mesurés sur le train cible.

| encodeur / cible | vidéo seule | audio seule | fusion (plancher) | phi, modalités | phi, sources | accord vidéo / audio : part, précision |
|---|---|---|---|---|---|---|
| ResNet-50 / animal | 60.2 | 52.8 | 68.7 | 0.07 | 0.37 | 34.2 %, 87.1 |
| ResNet-50 / cartoon | 37.1 | 34.3 | 50.1 | 0.11 | 0.38 | 23.9 %, 59.1 |
| ResNet-50 / human | 62.5 | 39.4 | 62.3 | 0.07 | 0.46 | 28.9 %, 84.4 |
| SlowFast / animal | 68.5 | 52.8 | 80.1 | 0.16 | 0.36 | 40.5 %, 90.4 |
| SlowFast / cartoon | 46.8 | 34.3 | 52.1 | 0.09 | 0.31 | 23.4 %, 58.4 |
| SlowFast / human | 77.9 | 39.4 | 76.6 | 0.04 | 0.40 | 32.9 %, 89.2 |

Erreur jointe observée sur erreur attendue sous indépendance : 1.05 à 1.22 pour les modalités, 1.21 à 2.03
pour les sources.

**Précision des pseudo-étiquettes au tour 1, 30 %, nombre égal par classe.**

| encodeur / cible | `confiance` | `accord_sources` | `accord_modalites` |
|---|---|---|---|
| ResNet-50 / animal | 92.0 | 90.9 | 86.4 |
| ResNet-50 / cartoon | 80.4 | 69.9 | 62.3 |
| ResNet-50 / human | 83.4 | 81.4 | 76.8 |
| SlowFast / animal | 95.6 | 91.7 | 92.4 |
| SlowFast / cartoon | 80.0 | 75.7 | 69.3 |
| SlowFast / human | 96.5 | 94.6 | 91.7 |

| mesure | valeur [IC95 %], configurations positives |
|---|---|
| précision tour 1, `accord_modalites − confiance`, critère primaire | **−8.17** [−8.62, −7.74], 0/6 : **non soutenu** |
| précision tour 1, `accord_modalites − accord_sources` | −4.24 [−4.69, −3.81], 1/6 |
| précision tour 1, `accord_sources − confiance` | −3.93 [−4.26, −3.60], 0/6 |
| accuracy finale, `accord_modalites − confiance` | −1.96 [−2.58, −1.32], 1/6 |
| accuracy finale, `accord_modalites − accord_sources` | −2.29 [−2.91, −1.68], 0/6 |
| accuracy finale moyenne | `confiance` 73.10, `accord_sources` 73.42, `accord_modalites` 71.14, plancher 64.98 |
| runs sous le plancher, sur 60 | `confiance` 3, `accord_sources` 3, `accord_modalites` 4 |
| précision accord − confiance au fil des tours, même modèle | −8.17, −6.61, −3.69, −2.65, −2.66 |
| non pré-enregistré : accuracy finale `accord_sources − confiance` | +0.33 [−0.29, +0.96], 3/6 |

La précision suit le même ordre à 10, 20 et 50 % de couverture.

**Conclusions.**

1. **Le critère primaire est rejeté, et l'effet inverse est net.** Classer par l'accord entre modalités fait
   perdre 8.17 points de précision, dans les 6 configurations, à toutes les couvertures et à tous les tours.
   L'auto-apprentissage final perd 1.96 point.
2. **La moitié du raisonnement est confirmée.** Les erreurs de la vidéo et de l'audio sont presque
   indépendantes (phi de 0.04 à 0.16), bien moins corrélées que celles des deux classifieurs de sources (phi
   de 0.31 à 0.46).
3. **L'autre moitié échoue : chaque modalité seule est faible.** L'audio seul fait 34 à 53 %, la vidéo seule
   37 à 78 %, nettement sous la fusion (50 à 80 %). Le co-training suppose des erreurs décorrélées ET des vues
   suffisantes chacune. La seconde condition est fausse ici. Exiger l'accord d'une modalité faible injecte
   son bruit : c'est le « veto de la modalité faible ».
4. **Sur HAC, les modalités sont complémentaires, pas redondantes.** La fusion tire sa force des cas où une
   modalité compense l'autre. Exiger leur accord écarte justement ces cas. La conclusion ne tient pas à la
   mesure d'accord : même le simple accord des argmax (23 à 40 % des points) est moins juste que les 30 % les
   plus sûrs selon la confiance, dans 5 configurations sur 6.
5. **Sous-produit sur M2CAN : le facteur d'incertitude de l'UACR n'aide pas non plus ici.** L'accord entre
   classifieurs de sources fait perdre 3.93 points de précision par rapport à la confiance seule, dans les
   6 configurations, sans gain d'accuracy finale. Ce qui fait marcher l'UACR serait l'auto-apprentissage
   lui-même, pas son facteur d'incertitude. À vérifier dans M2CAN entraîné de bout en bout.

**Réserves.**

- Caractéristiques gelées et régressions logistiques. Dans M2CAN, les têtes de modalités s'entraînent avec
  le reste du réseau, et leur qualité peut différer.
- La forme exp(−KL), reprise de M2CAN, est sévère : une modalité faible mais sûre d'elle suffit à exclure un
  point. L'accord des argmax échoue toutefois aussi (conclusion 4).
- HAC uniquement. Sur le sentiment, tout dépend de la force de chaque modalité seule, non mesurée ici.
- L'enseignement croisé (C2) n'a pas été testé.

---

## 17. Fusion géométrique + pseudo-étiquettes WBHR dans la boucle (Twitter, 3 graines)

**Pourquoi.** Ne pas abandonner le WBHR : s'en servir pour étiqueter la cible pendant l'entraînement de la
fusion géométrique, au lieu de le réserver à la phase 2. Le test figé sur HAC (`results/hac_wbhr_pl`) était
négatif (−1.67 point avec le filtre d'accord, −6.76 sans), mais il gelait le modèle ; ici l'encodeur apprend.

**Ce qui a été fait.** `phase1_pl.py` (ne modifie aucun fichier du bras C). Drapeaux canoniques de la fusion
géométrique, inchangés. Tous les 750 pas, sur au plus 3000 exemples d'entraînement de la cible : barycentres
sources, partition spectrale, transport et appariement hongrois, chaque point reçoit l'étiquette de son groupe.
Filtre d'accord (argmax de la moyenne des têtes texte et image égal à l'étiquette WBHR), puis quota par classe
selon la marge cosinus, taux 0.3, 0.6, 0.9 puis 1.0. Les exemples cibles étiquetés entrent dans la SupCon
fusionnée. Graines 123456, 234567, 345678, une par compte Kaggle (`geo-pl-twitter-<graine>`), 2 époques,
6.8 à 7.0 h par run. Journaux et résumé : `results/twitter_geo_pl/`. Les fichiers de sortie du kernel
(checkpoints, `history_C.json`, `pseudo_labels_log.json`) n'ont pas été conservés par Kaggle ; tout ce qui
suit vient des journaux.

**Résultats.** Meilleur point, choisi sur la cible, apparié par graine à la section 13.

| graine | fusion géométrique | + pseudo-étiquettes WBHR | écart |
|---|---|---|---|
| 123456 | 73.60 | 68.27 (ep0+3000) | −5.33 |
| 234567 | 68.00 | 69.40 (ep0+3750) | +1.40 |
| 345678 | 69.07 | 64.60 (ep0+3750) | −4.47 |
| moyenne | 70.22 ± 2.97 | **67.42 ± 2.51** | **−2.80 ± 3.66** |

Dernier point : 66.80 / 67.93 / 63.07 (moyenne 65.93). Point choisi par le minimum du ratio intra sur inter :
65.53 / 67.13 / 61.33 (moyenne 64.66). Le plafond oracle égale le meilleur point dans les 3 runs.

Justesse des pseudo-étiquettes retenues, par tour (graine 123456 / 234567 / 345678) :

| tour | taux | retenues (sur 3000) | justesse des retenues | justesse WBHR complète |
|---|---|---|---|---|
| 0 (750 pas) | 0.3 | 300 | 0.35 / 0.26 / 0.32 | 0.34 / 0.32 / 0.35 |
| 1 | 0.6 | 623 / 1105 / 878 | 0.49 / 0.65 / 0.71 | 0.47 / 0.60 / 0.46 |
| 2 | 0.9 | 2183 / 2369 / 1893 | 0.73 / 0.77 / 0.67 | 0.63 / 0.71 / 0.60 |
| 3 | 1.0 | 2666 / 2585 / 2590 | 0.70 / 0.74 / 0.66 | 0.68 / 0.70 / 0.61 |
| 4 à 9 | 1.0 | 2429 à 2781 | 0.67 à 0.69 / 0.70 à 0.72 / 0.65 | 0.64 à 0.65 / 0.67 à 0.70 / 0.60 à 0.63 |

**Conclusions.**

1. **Non soutenu.** La variante perd 2.80 points en moyenne, 2 graines sur 3 en baisse ; l'écart-type des
   écarts (3.66) interdit de conclure à mieux qu'une absence de gain.
2. **Le premier tour arrive trop tôt.** À 750 pas, le WBHR est au hasard (26 à 35 % de justesse) : on injecte
   300 étiquettes aléatoires dans la SupCon fusionnée.
3. **Le filtre d'accord ne filtre presque plus.** À partir du taux 1.0, 81 à 93 % des points sont retenus, avec
   une justesse de 65 à 74 % : la SupCon s'entraîne sur environ 30 % d'étiquettes fausses.
4. **Biais de confirmation.** Le pic arrive à la fin de l'époque 0 (3000 à 3750 pas), puis l'exactitude HOT-DA
   et la justesse des pseudo-étiquettes stagnent ou baissent pendant toute l'époque 1. Les erreurs du WBHR sont
   apprises, puis reproduites.
5. Avec la section 14 (HAC) et le test figé, c'est le troisième résultat où les étiquettes du transport ne font
   pas mieux qu'une sélection simple : la justesse du WBHR plafonne au niveau de son exactitude.

**Réserves.** 3 graines, une cible. Un seul réglage (`--pl_every 750`, taux 0.3 à 1.0, filtre d'accord) ; un
démarrage plus tardif (après la montée de l'époque 0) et un taux plafonné à 0.3 ou 0.5 n'ont pas été testés.

---

## 18. Options de la fusion géométrique sur HAC (paquet `fusion_geometrique_m`, protocole pré-enregistré)

**Pourquoi.** Une critique externe (12/10) proposait : une perte sur la cible, un transport moins contraint, un z
plus grand, plus d'exemples sources pour les barycentres. On teste chaque piste seule, puis ensemble.

**Ce qui a été fait.** Paquet `C:\Users\KEVIN\Downloads\fusion_geometrique_m` (M modalités, autonome), trois options
ajoutées, défauts inchangés. HAC, audio AST + vidéo SlowFast, caractéristiques gelées, sans vues augmentées fortes.
Base : `config_exemple.yaml` (3 époques × 3750 pas, lot 4, lr extracteur 2e-6, z 64-d, partition spectrale).
7 configurations × 3 cibles × 3 graines = 63 runs, 3 kernels CPU Kaggle `fgm-hac-options-<cible>`, 8 min par run.
Protocole écrit avant les résultats (`results/hac_fgm_options/protocole.json`, empreinte du paquet vérifiée à
l'envoi) : métrique principale = dernier point ; SOUTENU si IC95 de l'écart moyen apparié (9 paires) > 0 et
au moins 7/9 positifs. Analyse : `analyse.py`, `analyse.json`.

**Résultats.** Dernier point, moyenne sur 3 graines.

| configuration | human | animal | cartoon | écart moyen à la base [IC95] | positifs | verdict |
|---|---|---|---|---|---|---|
| base | 68.60 | 66.48 | 52.91 | | | |
| lr extracteur 5e-5 | 83.82 | 72.92 | 54.28 | +7.68 [+1.83, +13.52] | 7/9 | **SOUTENU** |
| partition équilibrée | 76.09 | 72.16 | 58.26 | +6.17 [+3.54, +8.81] | 8/9 | **SOUTENU** |
| z 256-d | 71.98 | 69.32 | 54.13 | +2.48 [−0.54, +5.50] | 7/9 | inconclusif |
| maximisation de l'information | 68.60 | 67.80 | 53.98 | +0.80 [−0.52, +2.11] | 4/9 | inconclusif |
| sources limitées à 160 | 68.60 | 66.48 | 52.91 | 0.00 | 0/9 | non soutenu |
| **les quatre premières ensemble** | **85.02** | **79.36** | **61.62** | **+12.67 [+8.67, +16.68]** | **9/9** | **SOUTENU** |

Repères non appariés (code d'origine, avec augmentations fortes, meilleur point) : plancher source-only
76.59 / 80.68 / 50.92 ; M2CAN 85.99 / 84.85 / 72.94.

Oracle de la partition au dernier point : 62.72 (base), 70.80 (lr), 68.83 (équilibrée), 75.33 (ensemble). La perte
due à l'appariement reste sous 0.5 point partout.

**Conclusions.**

1. **Le taux de l'extracteur était le premier verrou.** À 2e-6 (taux publié, prévu pour un réglage fin de BERT et
   ResNet), l'extracteur posé sur des caractéristiques gelées ne bouge pas. À 5e-5 : +15.2 points sur human.
2. **La partition équilibrée bat la partition spectrale** sur les 3 cibles (+5.4 à +7.5), comme sur Twitter
   (section 15, 69.07 contre 66.20). La critique proposait l'inverse (relâcher les marges) ; c'est la contrainte
   d'équilibre, appliquée à la partition, qui aide.
3. **Ensemble, +12.67 points, 9 paires sur 9.** Sur human (85.02), la fusion géométrique arrive au niveau de M2CAN
   (85.99, non apparié) et passe le plancher source-only de 8.4 points ; sur cartoon, +10.7 au-dessus du plancher.
   Sur animal (79.36), elle reste 1.3 point sous le plancher.
4. **Plus d'exemples sources ne change rien.** 160 ou tous : mêmes étiquettes au bit près, car l'appariement est
   déjà à l'oracle. Le point 4 de la critique est réfuté.
5. **La perte sur la cible (SHOT) n'apporte presque rien ici** (+0.80, 4/9).

**Réserves.**
- Caractéristiques gelées et sans vues augmentées fortes : les repères M2CAN et plancher ne sont pas appariés.
  La comparaison qui compte est interne au paquet.
- Le dernier point est la métrique principale, mais la durée d'entraînement (3 époques) vient de M2CAN, non réglée.
- 3 graines, petits tests cibles (176 à 276 clips, environ ±3 points de bruit par graine).
- À reporter sur le sentiment de bout en bout (BERT et ResNet entraînés) : le taux de l'extracteur n'y a pas le
  même rôle, mais la partition équilibrée s'y transpose directement.

---

## 19. Pseudo-étiquettes WBHR sur AVA (esthétique, 1 graine)

**Ce qui a été fait.** Cible AVA (2 classes), graine 123456, 3 époques, évaluation tous les 169 pas, code d'origine
(`phase1_diff.py` 798ec4aafaf5). `geo-pl-ava-123456` (compte djoumessi237, lancé par une autre session) :
`phase1_pl.py`, un tour tous les 338 pas. `brasc-ava-123456` : la même chose sans pseudo-étiquettes. Journaux :
`results/ava_geo_pl/`.

| | fusion géométrique | + pseudo-étiquettes WBHR | écart |
|---|---|---|---|
| meilleur point | 71.43 (ep0+845) | 75.80 (ep2+676) | +4.37 |
| dernier point | 63.40 | 75.32 | +11.92 |
| critère intra/inter | 60.68 | 73.55 | +12.87 |

**Lecture.** Sans pseudo-étiquettes, le pic arrive à la fin de l'époque 0, puis la partition se déséquilibre
(groupes 673 / 174) et l'exactitude retombe à 63. Avec, les groupes restent près de 514 / 333 et l'exactitude monte
jusqu'à la fin. La justesse des pseudo-étiquettes retenues passe de 0.47 (1er tour) à 0.76. Contraire de Twitter
(section 17), où la variante perdait 2.80 points.

**Réserves.** Une seule graine. Le gain tient surtout à la stabilité de la partition : il rejoint la section 18
(partition équilibrée), qui pourrait donner le même effet sans pseudo-étiquettes. À vérifier sur 3 graines.

---

## 20. La méthode complète sur HAC : partition spectrale et barycentres de Wasserstein, avec lr extracteur et z 256-d

**Pourquoi.** La section 18 obtenait son meilleur score en remplaçant la partition spectrale et les barycentres de
Wasserstein par un transport équilibré vers les moyennes de classe. Il fallait mesurer la méthode elle-même, avec
seulement les options qui améliorent l'espace z, sans et avec les pseudo-étiquettes par barycentres de Wasserstein.

**Ce qui a été fait.** Même paquet, mêmes données, mêmes graines que la section 18. Protocole écrit avant les
résultats (`results/hac_fgm_options/protocole_2.json`). Configurations : `c0_base` (contrôle), `lr_z256`
(lr extracteur 5e-5, z 256-d, partition spectrale, barycentres), `lr_z256_pseudo` (la même avec pseudo-étiquettes,
tous les 750 pas, filtre d'accord, quotas 0.3 / 0.6 / 0.9 / 1.0). Grille complète sur CPU (`fgm-hac-options-g2-*`),
réplique GPU (`fgm-hac-options-g2gpu-*`, cartoon en cours). Analyse : `analyse_2.py`.

**Contrôle.** `c0_base` redonne au bit près la section 18 sur les 9 paires (CPU) et sur les 6 paires GPU disponibles.

**Résultats.** Dernier point, moyenne sur 3 graines (CPU).

| configuration | human | animal | cartoon | moyenne |
|---|---|---|---|---|
| base | 68.60 | 66.48 | 52.91 | 62.66 |
| méthode, lr + z256 | 89.37 | 76.89 | 55.81 | 74.02 |
| **méthode, lr + z256 + pseudo-étiquettes** | **87.44** | **87.69** | 58.41 | **77.85** |
| section 18, « tout » (sans barycentres) | 85.02 | 79.36 | **61.62** | 75.33 |
| repère : plancher source-only | 76.59 | 80.68 | 50.92 | 69.40 |
| repère : M2CAN | 85.99 | 84.85 | 72.94 | 81.26 |

| comparaison (9 paires) | écart [IC95] | positifs | verdict |
|---|---|---|---|
| lr + z256 contre base | +11.36 [+4.07, +18.66] | 9/9 | SOUTENU |
| + pseudo-étiquettes contre base | +15.19 [+9.26, +21.11] | 9/9 | SOUTENU |
| + pseudo-étiquettes contre lr + z256 | +3.82 [−1.41, +9.06] | 7/9 | inconclusif |
| + pseudo-étiquettes contre « tout » | +2.51 [−1.69, +6.72] | 5/9 | inconclusif |

Réplique GPU (human, animal) : mêmes conclusions ; + pseudo-étiquettes contre « tout » : +4.14 [+0.56, +7.72], 5/6.

Pseudo-étiquettes (animal) : justesse des points retenus 88 à 96 % au 1er tour (pas 750), 87 à 91 % en fin de run.

**Conclusions.**

1. **La méthode, avec sa partition spectrale et ses barycentres de Wasserstein, fonctionne** dès que l'espace z est
   entraînable : +11.36 points sans pseudo-étiquettes, +15.19 avec, 9 paires sur 9.
2. **Les pseudo-étiquettes par barycentres de Wasserstein aident quand le premier étiquetage est juste.** Sur animal,
   +10.8 points ; sur human, rien (−1.9). Au contraire de Twitter (section 17), le 1er tour est juste à ~90 %.
3. **Sur human et animal, la méthode complète dépasse les repères de M2CAN** (87.44 contre 85.99, 87.69 contre 84.85,
   non appariés) et le plancher source-only.
4. **Cartoon reste le point faible** : 58.41, à 14.5 points de M2CAN, et sous la version sans barycentres (61.62).
5. Elle n'est pas significativement meilleure que la version sans barycentres sur l'ensemble (+2.51, inconclusif) :
   elle gagne sur human et animal, perd sur cartoon.

**Réserves.** Repères M2CAN non appariés (anciens runs, vues fortes, meilleur point). Caractéristiques gelées.
3 graines, petits tests cibles. À mesurer : M2CAN sur ces mêmes caractéristiques ; la méthode sur Twitter.

---

## 21. La méthode contre M2CAN sur HAC, appariée (mêmes caractéristiques et vues fortes, mêmes graines)

**Ce qui a été fait.** Méthode d'ALGORITHME.md (lr extracteur 5e-5, z 256-d, partition spectrale, barycentres de
Wasserstein), avec et sans pseudo-étiquettes, sur les caractéristiques HAC **avec les 4 vues fortes** (comme M2CAN).
3 cibles × 3 graines, 3 époques, kernel GPU `fgm-hac-options-g3`. Protocole écrit avant :
`results/hac_fgm_options/protocole_3.json`. M2CAN : runs de la section 10, mêmes caractéristiques et vues fortes,
mêmes graines, dernier point (SlowFast ramené à 2048-d par ACP, 10 époques). Analyse : `analyse_3.py`, `analyse_3.txt`.

**Résultats.** Dernier point, moyenne sur 3 graines.

| | human | animal | cartoon | moyenne |
|---|---|---|---|---|
| méthode | 86.96 | 85.98 | 62.39 | 78.44 |
| méthode sans pseudo-étiquettes | 88.04 | 71.78 | 56.42 | 72.08 |
| M2CAN | 85.75 | 84.47 | 71.26 | 80.49 |

| comparaison (9 paires) | écart [IC95] | positifs | verdict |
|---|---|---|---|
| méthode − M2CAN | −2.05 [−6.60, +2.50] | 4/9 | indistinguable |
| méthode sans pseudo − M2CAN | −8.41 [−14.90, −1.92] | 2/9 | moins bien |
| méthode − méthode sans pseudo | +6.36 [+0.91, +11.81] | 7/9 | mieux |
| méthode, vues fortes − sans (section 20) | +0.60 [−2.57, +3.76] | 5/9 | indistinguable |

**Conclusions.**

1. **À entrées égales, la méthode n'est pas distinguable de M2CAN sur HAC** (−2.05 points, IC95 [−6.60, +2.50]).
   Elle fait mieux sur human (+1.21) et animal (+1.51), moins bien sur cartoon (−8.87).
2. **Les pseudo-étiquettes par barycentres de Wasserstein sont nécessaires** : sans elles, la méthode est nettement
   sous M2CAN (−8.41) ; avec, l'écart disparaît. Leur apport (+6.36) est significatif ici.
3. **Les vues fortes ne changent pas significativement la méthode** (+0.60).
4. **Cartoon reste le point faible**, à 9 points de M2CAN.

**Réserves.** M2CAN : 10 époques et ACP 2048-d de SlowFast ; méthode : 3 époques et 2304-d bruts. 3 graines,
petits tests. Sans pseudo-étiquettes, plusieurs runs donnent exactement la même exactitude d'une graine à l'autre
(88.04 trois fois sur human, 56.42 trois fois sur cartoon) : la partition finale du test semble identique. C'est
plausible si les groupes sont très séparés, mais à vérifier.

---

## 22. Hypothèses H1 et H2 du mémoire sur HAC

**Pourquoi.** Éprouver les deux hypothèses validées du mémoire. H1 : résumer chaque classe source par le
barycentre de Wasserstein des sources donne un représentant plus proche de la classe cible, et un meilleur
étiquetage, que la concaténation des sources. H2 : le transport entre structures (groupes cibles, classes
sources) étiquette mieux que le transport plat point à point.

**Ce qui a été fait.** Protocole `hac_fgm_options/protocole_hypotheses.json`, écrit à 06:13 avant tout résultat.
Base = la méthode (ALGORITHME.md, vues fortes). Une seule option de `phase2.py` change par configuration, et
elle agit à la fois sur les pseudo-étiquettes et sur l'évaluation : `representant: concatenation` (H1),
`transport: plat` (H2), les deux (transport « classique » vers les points sources). 3 cibles × 3 graines, kernels
CPU `fgm-hac-options-hyp-<cible>`, 36 runs, tous rc 0. Analyse : `analyse_hypotheses.py`, sortie dans
`analyse_hypotheses.txt`.

**Résultats.** Dernier point, moyenne sur 3 graines (%).

| configuration | human | animal | cartoon | moyenne |
|---|---|---|---|---|
| méthode (barycentre, hiérarchique) | 86.47 | 86.55 | 61.31 | 78.11 |
| H1 : concaténation, hiérarchique | 86.47 | 86.55 | 61.31 | 78.11 |
| H2 : barycentre, plat | 85.27 | 83.52 | 63.30 | 77.36 |
| concaténation, plat | 86.11 | 81.25 | 62.08 | 76.48 |

| comparaison (9 paires) | écart moyen | IC95 | positifs | verdict du protocole |
|---|---|---|---|---|
| H1 : méthode − concaténation | +0.00 | [+0.00, +0.00] | 0/9 | NON CONCLUANTE |
| H2 : méthode − plat | +0.75 | [−1.93, +3.43] | 6/9 | NON CONCLUANTE |
| méthode − concaténation et plat | +1.63 | [−1.50, +4.77] | 6/9 | NON CONCLUANTE |

Diagnostic de H1 (coût de transport entre le représentant de chaque classe et les points de cette classe dans le
test cible, vraies étiquettes, dernier point) : barycentre 0.037 à 0.068, concaténation 0.057 à 0.083. **Le
barycentre est plus proche dans 9 paires sur 9**, écart moyen +0.019.

**Conclusions.**

1. **H1, volet « représentant plus proche » : confirmé 9/9.** Le barycentre est plus près de la classe cible
   que les sources mises bout à bout, au sens du coût de transport.
2. **H1, volet « meilleur étiquetage » : non tranché, et l'expérience ne peut pas le trancher sous transport
   hiérarchique.** Les deux configurations donnent exactement les mêmes exactitudes, run par run. Ce n'est pas
   un bogue : le représentant n'intervient que dans la matrice W, donc dans l'appariement des groupes aux
   classes. Or l'appariement est déjà optimal (exactitude = oracle dans 36 runs sur 36). Un représentant plus
   proche ne peut rien changer quand l'appariement ne se trompe pas ; c'est la partition qui limite.
3. **H2 : non concluante.** Le transport hiérarchique fait mieux sur animal (+3.0 points en moyenne) et human
   (+1.2), moins bien sur cartoon (−2.0). Cela rejoint la grille 1 (section 18), où le transport équilibré vers
   les centres sources battait la partition spectrale sous la configuration de base.
4. **Le transport classique (concaténation et plat) n'est pas distinguable de la méthode** (+1.63, NC), avec
   un net retard sur animal (−5.3).

**Réserves.** 3 graines, petits tests HAC (276, 176 et 218 clips). Le diagnostic de H1 compare un nuage de 16
points (barycentre) à un nuage d'au plus 800 points (concaténation) : moyenner deux sources réduit la
dispersion du représentant, ce qui peut suffire à rapprocher le barycentre de la classe cible ; la proximité
mesurée n'est donc pas forcément un effet d'« alignement » plus fin. Pour que H1 puisse jouer sur l'exactitude,
il faudrait un étiquetage où le représentant agit point par point (par exemple le transport plat : comparer
`h2_plat` à `plat_concatenation`, +0.9 point en moyenne pour le barycentre, mieux sur 2 cibles sur 3, hors protocole).

---

## 23. Cadre A sur l'esthétique et le sentiment : la méthode contre M2CAN (résultat NON VALIDE pour M2CAN)

**Ce qui a été fait.** Protocoles `cadre_a/protocole_aesthetic.json` et `protocole_sentiment.json` (écrits avant les
mesures). Kernels `djoumessi237/bench-a-aesthetic` et `kevindjoumessiwoumpe/bench-a-sentiment`, 27 runs chacun, tous
rc 0. Méthode : 3 époques × 3750 pas ; M2CAN : configuration publiée, 10 époques. Mêmes caractéristiques gelées
(BERT pooler 768 + ResNet-50 2048, 1 vue faible + 4 fortes). Résultats : `cadre_a/sortie_<groupe>/resultats.json`.

**Résultats.** Dernier point, moyenne sur 3 graines (%).

| cible | classe majoritaire | méthode | sans pseudo-étiquettes | M2CAN |
|---|---|---|---|---|
| AVA | 54.07 | 73.12 | 74.03 | 52.42 |
| PCCD | 50.30 | 60.41 | 60.65 | 48.29 |
| RPCD | 58.56 | 54.11 | 55.57 | 54.58 |
| TumEmo | 33.33 | 43.60 | 42.76 | 38.04 |
| Twitter | 33.33 | 54.13 | 54.00 | 36.62 |
| Yelp | 33.33 | 43.53 | 44.78 | 36.93 |

| comparaison (9 paires) | esthétique | sentiment |
|---|---|---|
| méthode − sans pseudo | −0.87 [−1.68, −0.06], 3/9 : MOINS BIEN | −0.09 [−1.69, +1.51], 3/9 : INDISTINGUABLE |
| méthode − M2CAN | +10.78 [+3.65, +17.92], 7/9 | +9.89 [+5.41, +14.36], 9/9 |

**Pourquoi M2CAN n'est pas valide ici.** M2CAN n'apprend même pas ses sources. Sa perte de tâche reste au niveau du
hasard : sur AVA, 2.09 au départ et environ 1.9 à 2.2 ensuite, pour un hasard de 3 × ln 2 ≈ 2.08 ; sur Twitter,
3.40 puis 2.77, pour un hasard de 3 × ln 3 ≈ 3.30. Sur AVA, il prédit presque tout en classe 1 (rappel de 0.94),
et son exactitude cible reste sous la classe majoritaire sur AVA, PCCD et RPCD. Ses taux d'apprentissage publiés
sont faits pour un BERT et un ResNet ajustés de bout en bout (extracteur 2e-6 et classifieur 5e-5 pour le sentiment ;
valeurs par défaut 2e-5 et 5e-4 pour AVA). La méthode, elle, a reçu un taux d'extracteur réglé pour les
caractéristiques gelées (5e-5, section 18). La comparaison n'est donc pas équitable, à moins que l'adaptateur
`m2can_carac` n'ait en plus un défaut propre à ces jeux. Sur HAC, le même cadre donnait environ 86 % à M2CAN.

**Conclusions valides.**
1. Les pseudo-étiquettes n'aident pas en cadre A : légère perte sur l'esthétique, rien sur le sentiment.
   C'est le contraire de HAC (+6.36, section 21) et de l'AVA de bout en bout (section 19, 1 graine).
2. Sur RPCD, toutes les configurations restent sous la classe majoritaire.
3. Le sentiment sur caractéristiques gelées est difficile : 43 à 54 %, contre environ 70 % sur Twitter avec BERT
   ajusté (section 9). La sortie pooler de BERT non ajusté porte peu d'information de polarité.

**À faire avant toute conclusion contre M2CAN.** Un kernel de diagnostic : sonde linéaire sur les caractéristiques
gelées ; M2CAN avec un taux d'extracteur et de classifieur adapté au cadre gelé ; précision de M2CAN sur les sources ;
M2CAN sur HAC via `m2can_carac`, pour écarter un défaut de l'adaptateur.

---

## 24. Leviers sans toucher au barycentre, et encodeurs CLIP (esthétique et sentiment, cadre A)

> **Note du 20/10.** L'encadrant juge la partition équilibrée illégitime : elle suppose connues les proportions de classes de la cible (les tests de sentiment sont exactement équilibrés). Elle n'est plus présentée comme un résultat de la méthode, seulement comme une analyse sous cette hypothèse. Par ailleurs, la comparaison à M2CAN de ces sections est faite contre M2CAN à taux publiés ; voir la section 27 pour la comparaison à taux égal.

**Pourquoi.** L'appariement n'est pas le goulot (section 22) ; hors HAC, les pseudo-étiquettes du premier tour sont au
niveau du hasard et la partition se déséquilibre (section 23). Trois leviers sur la partition et les pseudo-étiquettes,
puis de meilleures caractéristiques (CLIP).

**Ce qui a été fait.** Protocoles `cadre_a/protocole_leviers_<groupe>.json` (13/10, sha 033b4a44), écrits avant les mesures.
Options de `fusion_geometrique_m`, défauts inchangés : `pseudo.debut: 3750` (pas de pseudo-étiquettes avant la fin de
l'époque 1), `phase2.partition: spectrale_equilibree` (même plongement spectral et mêmes k-moyennes, affectation finale par
transport équilibré vers les centres), `phase2.evaluation_jointe` (test cible partitionné avec 3000 points du train cible).
Kernels `bench-a-aesthetic-leviers` (compte2) et `bench-a-sentiment-leviers` (compte1), 45 runs chacun, tous rc 0. La
méthode relancée reproduit exactement bench-a (runs déterministes). CLIP ViT-B/32 (512 + 512) : `extract-clip-<groupe>` puis
`bench-a-aesthetic-clip` (compte3, protocole `protocole_clip_aesthetic.json`). Analyse : `cadre_a/analyse_leviers_clip.py`.

**Leviers** (dernier point, moyenne 3 graines ; écart à la méthode sur 9 paires)

| | AVA | PCCD | RPCD | esthétique | TumEmo | Twitter | Yelp | sentiment |
|---|---|---|---|---|---|---|---|---|
| méthode | 73.12 | 60.41 | 54.11 | | 43.60 | 54.13 | 43.53 | |
| début tardif | 73.83 | 60.88 | 54.31 | +0.46 [+0.13, +0.79] 7/9 MIEUX | 44.27 | 54.51 | 43.96 | +0.49 [−0.52, +1.50] NC |
| spectrale équilibrée | **76.51** | 60.68 | **58.44** | **+2.66 [+1.08, +4.24] 8/9 MIEUX** | **46.64** | 54.20 | **52.00** | +3.86 [−0.09, +7.81] 6/9 NC |
| évaluation jointe | 72.53 | 60.13 | 55.49 | +0.17 NC | 43.51 | 54.18 | 43.73 | +0.05 NC |
| les trois | 75.68 | 60.45 | 58.28 | +2.26 [+0.61, +3.91] 7/9 MIEUX | 46.29 | 53.62 | 50.64 | +3.10 [−0.37, +6.56] NC |

Sans contrainte, les groupes se déséquilibrent (Yelp : 1097 / 112 / 291 ; RPCD : 258 / 589) ; la partition équilibrée
les ramène à des tailles égales. La justesse des pseudo-étiquettes retenues ne bouge presque pas (63 % en esthétique,
55 à 57 % en sentiment).

**CLIP, esthétique** (méthode et M2CAN, mêmes caractéristiques CLIP)

| | AVA | PCCD | RPCD |
|---|---|---|---|
| méthode, CLIP (BERT/ResNet) | 72.06 (73.12) | 62.06 (60.41) | 61.08 (54.11) |
| sans pseudo-étiquettes, CLIP | 73.04 | 63.75 | 61.67 |
| M2CAN, CLIP (BERT/ResNet) | 62.38 (52.42) | 61.59 (48.29) | 58.24 (54.58) |

| comparaison (9 paires) | écart | verdict |
|---|---|---|
| méthode − M2CAN, CLIP | +4.33 [+1.00, +7.66], 7/9 | MIEUX |
| méthode − sans pseudo, CLIP | −1.09 [−1.75, −0.43], 0/9 | MOINS BIEN |
| méthode CLIP − méthode BERT/ResNet | +2.52 [−0.26, +5.30], 6/9 | NC (RPCD +7.0) |
| M2CAN CLIP − M2CAN BERT/ResNet | +8.97 [+5.66, +12.29], 9/9 | MIEUX |

**Conclusions.**
1. **La partition spectrale équilibrée est le levier utile** : significative sur l'esthétique (+2.66, RPCD passe au-dessus
   de la classe majoritaire), forte mais non significative sur le sentiment (+3.86, Yelp +8.5, Twitter inchangé). Elle garde
   la partition spectrale, les barycentres et le transport hiérarchique.
2. Le début tardif aide à peine (+0.46) ; l'évaluation jointe n'apporte rien.
3. **Avec CLIP, M2CAN se met à apprendre** (+9 points), et la méthode le devance de 4.33 points (significatif). C'est la
   première comparaison avec M2CAN sur l'esthétique qui ne repose pas sur un M2CAN cassé.
4. Les pseudo-étiquettes restent nuisibles hors HAC (−1.09 avec CLIP, −0.87 avec BERT/ResNet).

**Réserves.** 3 graines. M2CAN garde des taux d'apprentissage prévus pour des encodeurs ajustés ; avec CLIP, il reste
loin de ses valeurs publiées (77.9 / 69.1 / 73.8). Le levier gagnant a été choisi sur ces mêmes cibles : sa confirmation
sur CLIP est lancée (`bench-a-sentiment-clip`, `bench-a-aesthetic-clip-equilibree`, protocoles `protocole_clip_*.json`).
Le code de référence de M2CAN est désormais copié dans `results/code_ref/` (le scratchpad temporaire commençait à être purgé).

---

## 25. Confirmation sur CLIP : sentiment, et partition équilibrée sur l'esthétique

> **Note du 20/10.** L'encadrant juge la partition équilibrée illégitime : elle suppose connues les proportions de classes de la cible (les tests de sentiment sont exactement équilibrés). Elle n'est plus présentée comme un résultat de la méthode, seulement comme une analyse sous cette hypothèse. Par ailleurs, la comparaison à M2CAN de ces sections est faite contre M2CAN à taux publiés ; voir la section 27 pour la comparaison à taux égal.

**Ce qui a été fait.** Protocoles `cadre_a/protocole_clip_sentiment.json` et `protocole_clip_aesthetic_equilibree.json`
(14/10, avant les mesures, sha 033b4a44). Kernels `bench-a-sentiment-clip` (36 runs) et `bench-a-aesthetic-clip-equilibree`
(9 runs, appariés aux runs de `bench-a-aesthetic-clip`), compte3, tous rc 0.

**Résultats.** Dernier point, moyenne 3 graines (%).

| | TumEmo | Twitter | Yelp | AVA | PCCD | RPCD |
|---|---|---|---|---|---|---|
| méthode, CLIP | 46.02 | 56.36 | 39.67 | 72.06 | 62.06 | 61.08 |
| + partition équilibrée, CLIP | **53.07** | **57.49** | **45.27** | **72.96** | **62.61** | 60.57 |
| sans pseudo-étiquettes, CLIP | 44.80 | 56.38 | 44.87 | 73.04 | 63.75 | 61.67 |
| M2CAN, CLIP | 52.18 | 47.89 | 37.73 | 62.38 | 61.59 | 58.24 |
| M2CAN publié (encodeurs ajustés) | 63.8 | 84.7 | 63.7 | 77.9 | 69.1 | 73.8 |

| comparaison (9 paires) | sentiment, CLIP | esthétique, CLIP |
|---|---|---|
| partition équilibrée − méthode | **+4.59 [+2.43, +6.76], 8/9 MIEUX** | +0.31 [−0.40, +1.03], 5/9 NC |
| partition équilibrée − M2CAN | **+6.01 [+2.29, +9.72], 8/9 MIEUX** | **+4.64 [+1.06, +8.22], 8/9 MIEUX** |
| méthode − M2CAN | +1.41 [−3.86, +6.69], 5/9 NC | +4.33 [+1.00, +7.66], 7/9 MIEUX |
| méthode − sans pseudo | −1.33 [−3.70, +1.03], NC | −1.09 [−1.75, −0.43], MOINS BIEN |
| M2CAN CLIP − M2CAN BERT/ResNet | +8.73 [+3.34, +14.13], 7/9 | +8.97 [+5.66, +12.29], 9/9 |

**Conclusions.**
1. **Avec la partition équilibrée, la méthode devance M2CAN sur les deux familles image et texte, à caractéristiques CLIP
   identiques** : +6.01 en sentiment, +4.64 en esthétique, 8 paires sur 9 à chaque fois. Sans elle, l'écart n'est
   significatif qu'en esthétique.
2. **Bilan de la partition équilibrée sur les quatre bancs** : MIEUX sur l'esthétique BERT/ResNet (+2.66) et le sentiment
   CLIP (+4.59), positif non significatif sur le sentiment BERT/ResNet (+3.86), nul sur l'esthétique CLIP (+0.31). Elle
   aide là où la partition spectrale se déséquilibre (Yelp : 794 / 545 / 161 sans elle), surtout à 3 classes, et ne nuit
   nulle part.
3. CLIP n'aide pas la méthode de base en sentiment (+0.26 NC), mais M2CAN y gagne +8.7 : il apprend enfin, tout en restant
   à 11 à 37 points de ses valeurs publiées.
4. Les pseudo-étiquettes restent sans apport hors HAC (−1.33 NC en sentiment CLIP).

**Réserves.** La partition équilibrée a été retenue après les leviers (section 24) : sa confirmation sur CLIP, avec
protocoles écrits avant, porte sur de nouvelles caractéristiques mais sur les mêmes cibles. M2CAN garde des réglages
prévus pour des encodeurs ajustés. 3 graines.

---

## 26. La méthode avec BERT et ResNet-50 entraînés de bout en bout (esthétique)

**Ce qui a été fait.** Protocole `cadre_a/protocole_e2e_aesthetic.json` (15/10, avant les mesures). Kernels
`e2e-aesthetic-{ava, pccd, rpcd}` (compte3, compte1, compte2), builder `kaggle/e2e/build.py` (worktree inspiring-mahavira).
Méthode d'ALGORITHME.md inchangée (z 256, extracteur 5e-5, pseudo-étiquettes, partition spectrale, sans équilibrage),
dorsales BERT (pooler) et ResNet-50 de M2CAN entraînées à 2e-6, données brutes par les chargeurs de M2CAN. 3 époques
de 847 pas, pseudo-étiquettes tous les 169 pas, évaluation en fin d'époque. 9 runs, rc 0, 143 à 156 min chacun.

**Résultats.** Dernier point, moyenne 3 graines (%).

| | AVA | PCCD | RPCD | moyenne |
|---|---|---|---|---|
| **bout en bout** | 56.67 | 60.02 | 53.96 | 56.88 |
| bout en bout, meilleur point | 59.66 | 60.02 | 63.09 | 60.92 |
| encodeurs gelés (section 23) | 73.12 | 60.41 | 54.11 | 62.55 |
| M2CAN, article (meilleure époque) | 79.9 | 69.8 | 74.5 | 74.7 |
| classe majoritaire | 54.07 | 50.30 | 58.56 | |

Bout en bout − gelé : −5.67 [−12.03, +0.69], 3/9 (INDISTINGUABLE, AVA −16.5).

**Ce qui se passe** (graine 123456) : la silhouette de la partition cible monte de 0.10 à 0.70 au fil de
l'entraînement, mais la partition devient très déséquilibrée (AVA 711 / 136, PCCD 156 / 691, RPCD 142 / 705). Les
encodeurs entraînés rendent l'espace très compact autour d'un amas dominant, que la partition spectrale découpe en un
gros groupe et un petit. Les pseudo-étiquettes restent proches du hasard (44 à 58 % de justesse en première époque,
55 à 67 % ensuite) : la SupCon fusionnée rapproche donc des points mal étiquetés, cette fois en modifiant BERT et ResNet.
L'appariement reste à l'oracle (dernier = oracle) : le goulot est la partition.

**Conclusions.**
1. **Entraîner les encodeurs, dans cette configuration, dégrade la méthode** au lieu de la rapprocher de M2CAN :
   56.9 contre 62.6 gelé, 74.7 pour M2CAN.
2. La configuration réglée pour les caractéristiques gelées (extracteur 5e-5, z 256, pseudo-étiquettes dès le pas 169)
   ne se transpose pas. L'ancienne version de bout en bout (section 19 : z 64, extracteur 2e-6, 1 graine) atteignait
   75.32 sur AVA avec pseudo-étiquettes, et sans elles s'effondrait de la même façon (673 / 174, 63.40).

**Réserves.** 3 graines ; une seule configuration de bout en bout, non réglée ; la comparaison à l'article n'est pas
appariée (leur meilleure époque contre notre dernier point).

---

## 27. M2CAN avec le même taux d'extracteur que la méthode (CLIP gelé) — esthétique, sentiment, HAC

**Pourquoi.** La méthode a reçu un taux d'extracteur réglé pour les caractéristiques gelées (5e-5, section 18), M2CAN
gardait ses taux publiés, prévus pour des encodeurs ajustés. Objection du relecteur (correction.md, A1).

**Ce qui a été fait.** Protocoles `cadre_a/protocole_m2can_taux_{aesthetic,sentiment}.json` (20/10, avant les mesures).
Kernels `bench-a-{aesthetic,sentiment}-m2can-taux` (compte3), mêmes caractéristiques CLIP que les sections 24-25.
`m2can_carac/run.py` accepte désormais `--extractor_lr`, `--mlb_lr`, `--classifier_lr`. Deux réglages :
`extracteur_5e-5` (seul l'extracteur change) et `taux_eleves` (extracteur 5e-5, fusion et classifieur 5e-4).
Esthétique : 18 runs, rc 0.

**Résultats, esthétique.** Dernier point, moyenne 3 graines (%).

| | AVA | PCCD | RPCD | moyenne |
|---|---|---|---|---|
| méthode, CLIP (section 24) | **72.06** | 62.06 | **61.08** | **65.07** |
| M2CAN, taux publiés | 62.38 | 61.59 | 58.24 | 60.74 |
| M2CAN, extracteur 5e-5 | 69.62 | **62.42** | 59.94 | 63.99 |

| comparaison (9 paires) | écart | verdict |
|---|---|---|
| M2CAN extracteur 5e-5 − M2CAN publié | +3.25 [+0.86, +5.64], 9/9 | MIEUX |
| **méthode − M2CAN extracteur 5e-5** | **+1.08 [−0.13, +2.28], 7/9** | **INDISTINGUABLE** |

Les deux réglages de M2CAN donnent exactement les mêmes chiffres : en esthétique, la configuration publiée fixe déjà
la fusion et le classifieur à 5e-4 (valeurs par défaut de M2CAN), seul l'extracteur change réellement.

**Conclusions (esthétique).**
1. **Le « +4.33, MIEUX » de la section 24 tenait surtout à l'asymétrie de réglage.** À taux d'extracteur égal, la
   méthode et M2CAN sont indistinguables (+1.08, IC95 frôlant zéro, 7/9 en faveur de la méthode).
2. Toute comparaison à M2CAN sur caractéristiques gelées doit désormais se faire contre M2CAN à 5e-5.

**Résultats, sentiment** (18 runs, rc 0). Dernier point, moyenne 3 graines (%).

| | TumEmo | Twitter | Yelp | moyenne |
|---|---|---|---|---|
| méthode, CLIP (section 25) | 46.02 | 56.36 | 39.67 | 47.35 |
| M2CAN, taux publiés | 52.18 | 47.89 | 37.73 | 45.93 |
| **M2CAN, extracteur 5e-5** | **58.02** | **59.07** | 50.31 | **55.80** |
| M2CAN, taux élevés (+ fusion et classifieur 5e-4) | 54.24 | 58.87 | **50.36** | 54.49 |

| comparaison (9 paires) | écart | verdict |
|---|---|---|
| M2CAN extracteur 5e-5 − M2CAN publié | +9.87 [+6.40, +13.33], 9/9 | MIEUX |
| M2CAN taux élevés − M2CAN publié | +8.56 [+4.07, +13.04], 8/9 | MIEUX |
| **méthode − M2CAN extracteur 5e-5** | **−8.45 [−12.32, −4.58], 0/9** | **MOINS BIEN** |
| méthode − M2CAN taux élevés | −7.14 [−10.62, −3.67], 0/9 | MOINS BIEN |

**Conclusions (sentiment).**
3. **À taux égal, M2CAN devance nettement la méthode en sentiment** (−8.45, 9 paires sur 9 en faveur de M2CAN).
   Le « jeu égal » de la section 25 (+1.41) venait de l'asymétrie de réglage.
4. Cohérent avec les figures t-SNE (cible Yelp isolée en un nuage à part des classes sources, `cadre_a/tsne_sentiment`) :
   quand la cible ne rejoint pas les amas sources, la partition est arbitraire, alors qu'un classifieur entraîné sur les
   sources (M2CAN) garde un pouvoir de transfert.

**Résultats, HAC avec CLIP pour la vidéo** (kernel `bench-a-hac-clip`, compte1, protocole `protocole_clip_hac.json` ;
vidéo CLIP 512-d sur 8 frames, audio AST ; 36 runs, rc 0). Dernier point, moyenne 3 graines (%).

| | human | animal | cartoon | moyenne |
|---|---|---|---|---|
| **méthode, CLIP** | **93.36** | **96.40** | 76.15 | **88.64** |
| méthode sans pseudo-étiquettes | 87.32 | 95.08 | 61.93 | 81.44 |
| M2CAN, taux publiés (config Twitter) | 86.23 | 86.17 | 77.06 | 83.16 |
| **M2CAN, extracteur 5e-5** | 87.68 | 95.45 | **81.96** | 88.36 |
| *rappel SlowFast : méthode / M2CAN publié (section 21)* | *86.96 / 85.75* | *85.98 / 84.47* | *62.39 / 71.26* | *78.44 / 80.49* |

| comparaison (9 paires) | écart | verdict |
|---|---|---|
| **méthode − M2CAN extracteur 5e-5** | **+0.27 [−3.68, +4.22], 5/9** | **INDISTINGUABLE** |
| méthode − M2CAN publié | +5.48 [+1.53, +9.43], 6/9 | INDISTINGUABLE (IC95 > 0 mais 6/9 < 7) |
| méthode − sans pseudo-étiquettes | +7.19 [+2.22, +12.17], 8/9 | MIEUX |
| M2CAN extracteur 5e-5 − M2CAN publié | +5.21 [+2.41, +8.01], 8/9 | MIEUX |

**Conclusions (HAC CLIP).**
5. **À taux égal, jeu égal avec M2CAN sur HAC** (+0.27) : la méthode devance M2CAN sur human (+5.7) et animal (+1.0),
   reste derrière sur cartoon (−5.8).
6. **CLIP pour la vidéo fait gagner environ 10 points à la méthode** par rapport à SlowFast (88.64 contre 78.44), et
   environ 8 à M2CAN réglé. C'est le meilleur résultat obtenu sur HAC.
7. **Les pseudo-étiquettes restent décisives sur HAC** (+7.19, significatif), comme avec SlowFast (+6.36), et les
   figures t-SNE le montrent (cible cartoon isolée sans elles, réunie aux classes sources avec elles).

**Bilan de la comparaison à taux égal :** HAC +0.27 (jeu égal), esthétique +1.08 (jeu égal), sentiment −8.45 (moins bien).

---

## 28. Bout en bout avec les réglages de l'ancienne version (z 64, extracteur 2e-6)

**Ce qui a été fait.** Protocole `cadre_a/protocole_e2e_aesthetic_ancien.json` (20/10). Kernels
`e2e-aesthetic-{ava-pccd, rpcd}-ancien` (compte2), graine 123456, seuls `fusion.dim` (64) et `lr_extracteur` (2e-6)
changent par rapport à la section 26.

**Résultat partiel (RPCD).** 58.32 au dernier point (groupes 189 / 658), contre 54.43 avec les réglages actuels
(même graine, groupes 142 / 705) et 58.56 pour la classe majoritaire. Évaluations : 49.70 (fin d'époque 1, groupes
418 / 429), 59.03 (209 / 638), 58.32 (189 / 658). La partition se déséquilibre encore, un peu moins vite.

**AVA et PCCD** (kernel `e2e-aesthetic-ava-pccd-ancien`, rc 0) : AVA 59.74 (groupes 706 / 141, meilleur point 60.92),
PCCD 59.39 (165 / 682). Avec les réglages actuels, même graine : AVA 58.91, PCCD 59.03.

| graine 123456 | AVA | PCCD | RPCD |
|---|---|---|---|
| bout en bout, anciens réglages | 59.74 | 59.39 | 58.32 |
| bout en bout, réglages actuels (section 26) | 58.91 | 59.03 | 54.43 |
| encodeurs gelés BERT/ResNet (moyenne 3 graines) | 73.12 | 60.41 | 54.11 |

**Conclusions.**
1. **Les anciens réglages ne sauvent pas le bout en bout** : +0.4 à +3.9 point, et la partition s'effondre de la même
   façon (706 / 141 sur AVA). L'échec de la section 26 ne vient donc ni de z 256 ni du taux d'extracteur 5e-5.
2. Le 75.32 de l'ancienne version (section 19, 1 graine) n'est pas retrouvé. Il venait d'un autre pipeline
   (`phase1_pl.py` sur `phase1_diff.py` : partition et barycentres évalués autrement, évaluation tous les 169 pas,
   meilleur point) ; les différences restantes ne sont pas isolées.

---

## 29. Bloc A : planchers (sources seules, auto-apprentissage naïf), analyse des journaux, diagnostic de partition

**Ce qui a été fait.** Protocole `cadre_a/protocole_bloc_a.json` (21/10, avant les mesures). Kernels processeur
`bases-diag` (compte1, compte2, compte3), code `kaggle/bases/bases.py`. Résultats : `cadre_a/bloc_a/compte{1,2,3}/`.
A1 : régression logistique sur les deux modalités concaténées (standardisées sur les sources), entraînée sur les sources.
A2 : même modèle + 5 tours d'auto-apprentissage par confiance (quotas 30/60/90/100/100 %), dernier tour. Graines :
sous-échantillons de 90 % (comparaison à la méthode non appariée au sens strict). A3/B3 : sur les `z_dernier.npz` des
runs t-SNE (CLIP, graine 123456). A5 : `cadre_a/analyse_journaux.py` sur 198 runs existants. A4 : vérifié dans les
protocoles, la méthode des sections 21 et 27 utilise la partition spectrale (aucune partition équilibrée).

**A1 et A2.** Moyenne 3 graines (%), au regard de la méthode et de M2CAN réglé (extracteur 5e-5, section 27).

| jeu | cible | sources seules | auto-apprentissage | méthode | M2CAN 5e-5 | oracle cible |
|---|---|---|---|---|---|---|
| HAC, CLIP | human | 73.79 | 88.04 | **93.36** | 87.68 | 94.08 |
| | animal | 86.55 | 92.80 | **96.40** | 95.45 | 97.16 |
| | cartoon | 71.10 | 81.35 | 76.15 | **81.96** | 93.12 |
| | **moyenne** | 77.15 | 87.40 | **88.64** | 88.36 | |
| esthétique, CLIP | AVA | 65.80 | 66.98 | **72.06** | 69.62 | 70.41 |
| | PCCD | 59.74 | 59.35 | 62.06 | **62.42** | 63.91 |
| | RPCD | 54.55 | 59.46 | **61.08** | 59.94 | 66.94 |
| | **moyenne** | 60.03 | 61.93 | **65.07** | 63.99 | |
| sentiment, CLIP | TumEmo | 51.64 | 55.13 | 46.02 | **58.02** | 79.64 |
| | Twitter | 52.07 | 55.49 | 56.36 | **59.07** | 75.51 |
| | Yelp | 50.44 | 51.31 | 39.67 | **50.31** | 57.67 |
| | **moyenne** | 51.38 | 53.98 | 47.35 | **55.80** | |
| esthétique, BERT/ResNet | moyenne (AVA, PCCD, RPCD) | 57.25 (61.59, 57.85, 52.30) | 58.79 (63.87, 57.93, 54.58) | **62.55** (73.12, 60.41, 54.11) | (cassé) | (70.29, 63.24, 65.09) |
| sentiment, BERT/ResNet | moyenne (TumEmo, Twitter, Yelp) | 49.04 (46.27, 49.71, 51.13) | **50.17** (46.11, 51.56, 52.84) | 47.09 (43.60, 54.13, 43.53) | (cassé) | (70.38, 71.84, 60.02) |

**A3 et B3** (graine 123456, CLIP ; exactitude, NMI, sonde linéaire oracle dans z, MMD² cible–sources).

| cible | avec pseudo-étiquettes | sans pseudo-étiquettes |
|---|---|---|
| human | 93.1 · NMI 0.88 · sonde 93.5 · MMD² 0.025 | 89.1 · 0.81 · 88.4 · 0.085 |
| animal | 98.3 · 0.96 · 88.1 · 0.017 | 94.9 · 0.91 · 79.0 · 0.050 |
| cartoon | 76.6 · 0.65 · 68.9 · 0.020 | 56.9 · 0.51 · 42.7 · 0.120 |
| AVA | 71.8 · 0.15 · 73.2 · 0.044 | 73.1 · 0.16 · 71.0 · 0.014 |
| PCCD | 61.9 · 0.04 · 61.6 · 0.008 | 64.8 · 0.07 · 63.8 · 0.014 |
| RPCD | 61.7 · 0.04 · 60.8 · 0.041 | 62.1 · 0.04 · 60.3 · 0.045 |
| TumEmo | 47.2 · 0.16 · **66.7** · 0.035 | 45.8 · 0.15 · 66.4 · 0.031 |
| Twitter | 55.9 · 0.14 · 60.9 · 0.061 | 55.7 · 0.14 · 60.9 · 0.057 |
| Yelp | 43.3 · 0.05 · 50.9 · **0.249** | 47.6 · 0.07 · 52.3 · 0.237 |

**A5.** Dernier point = oracle dans 184 runs sur 198 (écart moyen 0.21 point, maximum 10.5) ; les exceptions sont
surtout en sentiment. Le critère d'arrêt sans étiquette (ratio intra/inter) perd 1.99 point en moyenne sur le dernier
point (meilleur dans 61 runs, pire dans 109), jusqu'à −7 à −8 points en sentiment.

**Conclusions.**
1. **HAC : la méthode dépasse les deux planchers** (88.64 contre 87.40 pour l'auto-apprentissage), surtout sur human et
   animal ; sur cartoon, l'auto-apprentissage naïf (81.35) et M2CAN (81.96) font mieux qu'elle (76.15).
2. **Esthétique : la méthode dépasse les deux planchers** (CLIP 65.07 contre 61.93 ; BERT/ResNet 62.55 contre 58.79).
3. **Sentiment : la méthode est sous le plancher sources seules** (CLIP 47.35 contre 51.38 ; BERT/ResNet 47.09 contre
   49.04). Seul M2CAN réglé dépasse l'auto-apprentissage. La géométrie OT n'apporte rien sur le sentiment.
4. **Les pseudo-étiquettes rapprochent la cible des sources dans z** quand elles sont justes : MMD² divisé par 3 à 6 sur
   HAC (cartoon 0.120 → 0.020) avec un NMI en hausse ; aucun effet en esthétique ni en sentiment.
5. **Le goulot dépend de la famille.** TumEmo : z sépare les classes (sonde 66.7) mais la partition échoue (47.2) →
   partition. Yelp : cible isolée (MMD² 0.249, 4 à 30 fois les autres) et sonde à 50.9 → espace z. Esthétique :
   sonde ≈ exactitude → séparabilité des classes dans z. HAC : sonde ≈ exactitude, partition quasi parfaite.
6. « Appariement = oracle » vaut sur HAC et l'esthétique, pas toujours en sentiment (A5).

**Réserves.** Planchers et méthode partagent les graines mais pas les sous-échantillons (pas d'appariement strict) ;
diagnostics sur une seule graine ; la régression logistique n'utilise pas les vues fortes.

---

## 30. Variante fidèle à WBHR (branche `claude/repo-access-lzj47d`) et H1/H2 avec support `auto`, CLIP

**Ce qui a été fait.** Code de la branche (commit `84fd73b` de MM-WBHR, poussé le 26/10 par une session Claude en ligne),
extrait tel quel. Protocoles `cadre_a/protocole_wbhr_branche_{hac,aesthetic,sentiment}.json` et
`protocole_h1h2_hac_clip_auto.json` (27/10, avant les mesures). 3 époques, mêmes caractéristiques CLIP et graines que
les sections 25 et 27. Configurations : `methode` (référence), `support_auto` (barycentre à la taille moyenne de la classe
dans les sources, 400 au plus), `fin_wbhr` (appariement argmax, représentants transportés vers leur groupe, classifieur
logistique, support 16), `methode_wbhr` (les deux). Kernels `bench-a-{hac,aesthetic,sentiment}-wbhr-branche` et
`bench-a-hac-h1h2-auto`, 144 runs, tous rc 0. La référence relancée avec le code de la branche reproduit exactement les
sections 25 et 27 (runs déterministes).

**Résultats, variante WBHR** (dernier point, moyenne 3 graines, %).

| | HAC | esthétique | sentiment |
|---|---|---|---|
| `methode` | 88.64 | 65.07 | 47.35 |
| `support_auto` | 88.64 (identique) | 65.07 (identique) | 47.35 (identique) |
| `fin_wbhr` | 88.10 | 65.18 | **50.88** |
| `methode_wbhr` | **89.34** | 65.16 | 50.80 |
| M2CAN extracteur 5e-5 (section 27) | 88.36 | 63.99 | 55.80 |

| écart (9 paires) | HAC | esthétique | sentiment |
|---|---|---|---|
| `support_auto` − `methode` | +0.00 | +0.00 | +0.00 |
| `fin_wbhr` − `methode` | −0.54 NC | +0.12 NC | **+3.53 [+1.14, +5.92], 7/9 MIEUX** |
| `methode_wbhr` − `methode` | +0.70 NC | +0.09 NC | **+3.45 [+1.28, +5.63], 9/9 MIEUX** |
| `methode_wbhr` − M2CAN 5e-5 | +0.97 [−2.58, +4.53] NC | +1.17 [−0.04, +2.37] NC | −5.00 [−7.56, −2.44] MOINS BIEN |

En sentiment, la fin WBHR dépasse l'oracle de la partition de 2.33 points : le classifieur n'est plus limité par les groupes.

**Résultats, H1 et H2 sur HAC CLIP avec support `auto`** (barycentre et concaténation de cardinalités comparables).

| | human | animal | cartoon | moyenne |
|---|---|---|---|---|
| barycentre, hiérarchique (`methode`) | 93.36 | 96.40 | 76.15 | 88.64 |
| concaténation, hiérarchique (H1) | 93.36 | 96.40 | 76.15 | 88.64 |
| barycentre, plat (H2) | 91.55 | 90.53 | 76.91 | 86.33 |
| concaténation, plat | 92.27 | 90.53 | 76.91 | 86.57 |

| comparaison (9 paires) | écart | verdict du protocole |
|---|---|---|
| H1 : barycentre − concaténation | +0.00 (étiquettes identiques 9/9) | NON CONCLUANTE |
| H1, diagnostic : coût concaténation − barycentre | +0.022 en moyenne, barycentre plus proche **9/9** | (confirmé) |
| H2 : hiérarchique − plat | +2.31 [−0.04, +4.65], 7/9 | NON CONCLUANTE (de justesse) |
| hiérarchique − transport classique | +2.06 [−0.49, +4.61], 6/9 | NON CONCLUANTE |

**Conclusions.**
1. **La taille du barycentre ne change aucune étiquette, sur les trois familles** (16 points ou taille moyenne de la
   classe). Elle n'agit que sur l'appariement, déjà optimal.
2. **H1 : même verdict qu'avec SlowFast et 16 points.** Le barycentre reste plus proche de la classe cible (9/9), et
   la réserve « un nuage de 16 points est plus resserré » ne tient plus (cardinalités comparables) ; mais l'étiquetage
   est identique.
3. **H2 : le transport hiérarchique fait mieux que le plat sur HAC CLIP** (+2.31, 7/9, surtout animal +5.9), mais
   l'IC95 frôle zéro : non concluant de justesse, alors qu'avec SlowFast l'écart était +0.75.
4. **La fin fidèle à WBHR (classifieur sur les représentants transportés) aide sur le sentiment (+3.5) et rien
   ailleurs.** Elle ne suffit pas à rattraper M2CAN réglé (−5.00) ni le plancher sources seules (51.38, section 29).

**Réserves.** 3 graines ; la variante réintroduit un classifieur (sur les représentants, pas sur la fusion).

---

## 31. Configurations de la branche (commit dcff47f) et pilote de bout en bout avec maximisation de l'information

**Pourquoi.** Mesurer les configurations `configs/methode_*.yaml` de la branche `claude/repo-access-lzj47d` (commit
dcff47f) contre `methode`, sur CLIP gelé, et tester si la maximisation de l'information (SHOT) empêche l'effondrement
de la partition de bout en bout (section 26).

**Ce qui a été fait.** Protocoles `cadre_a/protocole_branche_dcff47f_<groupe>.json` et
`cadre_a/protocole_pilote_e2e_im03.json`, écrits avant tout résultat (28/10). 6 configurations × 3 cibles × 3 graines,
3 époques, extracteur 5e-5, pseudo-étiquettes actives ; chaque configuration ne change qu'une option par rapport à
`methode` (`methode_wbhr_zn` en change quatre). Analyse : `cadre_a/analyse_dcff47f.py` (sortie `.txt` et `.json`).
`methode` reproduit exactement la section 27 (88.64 / 65.07 / 47.35).

**Résultats (dernier point, moyenne des 9 runs).**

| configuration | HAC | esthétique | sentiment |
|---|---|---|---|
| methode | 88.64 | 65.07 | 47.35 |
| methode_im03 (IM 0.3) | 89.08 | 65.14 | 48.17 |
| methode_im1 (IM 1.0) | 88.66 | 65.16 | **50.41** |
| methode_zn (z normalisé) | 88.37 | 64.31 | 46.56 |
| methode_knn (graphe k-NN, k = 15) | 87.48 | **65.70** | 48.39 |
| methode_wbhr_zn | 88.71 | 64.36 | 47.84 |
| M2CAN, extracteur 5e-5 (section 27) | 88.36 | 63.99 | 55.80 |
| plancher sources seules (section 29) | 77.15 | 60.03 | 51.38 |

| comparaison (9 paires) | HAC | esthétique | sentiment |
|---|---|---|---|
| im03 − methode | +0.45, 6/9, NS | +0.08, 4/9, NS | **+0.82 [+0.32, +1.33], 9/9, MIEUX** |
| im1 − methode | +0.03, 3/9, NS | +0.09, 4/9, NS | **+3.07 [+0.01, +6.12], 7/9, MIEUX** |
| zn − methode | −0.26, NS | −0.76, 1/9, NS | −0.79, NS |
| knn − methode | −1.15, 1/9, NS | +0.63, 6/9, NS | +1.04, NS |
| wbhr_zn − methode | +0.08, NS | −0.71, 1/9, NS | +0.50, NS |
| meilleure − M2CAN 5e-5 | im03 : +0.72, 4/9, NS | knn : +1.71 [+0.14, +3.27], 7/9 | im1 : −5.39, 2/9, MOINS_BIEN |

NS = non significatif (INDISTINGUABLE du protocole).

**Pilote de bout en bout (AVA, graine 123456, IM 0.3).** 57.85, groupes 722 / 125 (le plus petit = 14.8 % du test),
plancher logistique 58.56, 142 min. Référence sans IM : 58.91 (711 / 136). Règle du protocole : plus petit groupe
≥ 30 % ET > 73.12 pour étendre ; **aucune des deux conditions n'est remplie : arrêt du bout en bout, limite déclarée.**

**Conclusions.**
1. **La maximisation de l'information est le seul levier qui aide, et seulement sur le sentiment** (+0.8 à +3.1). Elle
   ne rattrape ni M2CAN réglé (−5.39) ni le plancher sources seules (50.41 < 51.38).
2. **Aucune configuration ne change HAC ni l'esthétique** au-delà du bruit ; `zn` et `wbhr_zn` sont plutôt en dessous.
3. **knn contre M2CAN sur l'esthétique (+1.71, IC > 0) n'est pas un résultat** : la configuration a été choisie comme
   la meilleure des six sur le même test, et knn − methode n'est pas significatif. Comparaison secondaire seulement.
4. **De bout en bout, l'IM n'empêche pas l'effondrement de la partition** (722 / 125) : l'effondrement vient de z, pas
   d'un déséquilibre des prédictions des têtes.

**Réserves.** 3 graines, 3 époques (la branche en prévoit 10) ; 5 comparaisons principales par famille sans correction
pour comparaisons multiples (im1 sur le sentiment : borne basse +0.01) ; pilote de bout en bout sur une graine.

---

## Gabarit pour une nouvelle entrée

```markdown
## N. Titre court de la version ou de l'ablation

**Pourquoi.** La question posée, en une ou deux phrases.

**Ce qui a été fait.** La configuration exacte, les drapeaux, ce qui change par rapport à quel bras
de référence. Une seule variable doit changer, sinon le dire.

**Résultats.** Un tableau. Toujours préciser le moment de la mesure et le bruit associé.

**Conclusion.** Ce que ça tranche, et ce que ça ne tranche pas.

**Réserves.** Confusions de variables, graine unique, écart dans le bruit.
```
