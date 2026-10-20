# Fusion géométrique pour l'adaptation de domaine multi-sources et multimodale

Ce dépôt compare deux méthodes d'adaptation de domaine multi-sources et multimodale (MSMMDA) sur les mêmes données :

- **la méthode** (`methode/`) : fusion géométrique avec étiquetage de la cible par barycentres de Wasserstein, partition
  spectrale et appariement hongrois, et pseudo-étiquettes pendant l'entraînement. L'algorithme complet, ses fonctions
  de coût et ses hyperparamètres sont décrits dans [`methode/ALGORITHME.md`](methode/ALGORITHME.md) ;
- **M2CAN** (`m2can/`), la référence (Zhao et al., *Information Fusion* 117, 2025), à partir du code de ses auteurs
  (voir [`m2can/MODIFICATIONS.md`](m2can/MODIFICATIONS.md)).

Les deux méthodes tournent dans le **cadre A** : les encodeurs (BERT, ResNet-50, AST, SlowFast) sont gelés, leurs
caractéristiques sont extraites une fois, puis chaque méthode entraîne ses propres couches (extracteurs, fusion, têtes)
sur ces mêmes caractéristiques. La comparaison est donc appariée : mêmes entrées, mêmes graines, mêmes découpes.

## Jeux de données

| groupe | domaines (chacun est la cible à tour de rôle) | modalités | classes | préparation |
|---|---|---|---|---|
| HAC | human, animal, cartoon | audio (AST 768-d), vidéo (SlowFast 2304-d) | 7 | `preparation/hac_extraire.py` puis `hac_convertir.py` |
| sentiment | TumEmo, Twitter (T4SA), Yelp | texte (BERT 768-d), image (ResNet-50 2048-d) | 3 | `preparation/jeux_m2can_extraire.py --groupe sentiment` |
| esthétique | AVA, PCCD, RPCD | texte (BERT 768-d), image (ResNet-50 2048-d) | 2 | `preparation/jeux_m2can_extraire.py --groupe aesthetic` |

Deux modalités par jeu. Le flot optique de HAC n'est pas utilisé pour l'instant ; la méthode accepte déjà trois
modalités ou plus, M2CAN non.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 ou plus. **GPU détecté automatiquement** : s'il existe, tout tourne dessus ; sinon, sur le processeur.
L'entraînement des deux méthodes sur caractéristiques gelées reste raisonnable sur processeur. L'**extraction** des
caractéristiques, elle, demande un GPU en pratique (plusieurs heures à plusieurs jours sans).

## 1. Préparer les données

Chaque script télécharge les données publiques, extrait les caractéristiques et les écrit au format commun
(`<dossier>/meta.json`, `<dossier>/<domaine>/{train,test}/<modalité>.npy`, vues augmentées `<modalité>_aug.npy`,
`labels.npy`, `ids.npy`).

```bash
# HAC (Hugging Face hdong51/Human-Animal-Cartoon)
python preparation/hac_extraire.py --sortie donnees_brutes/hac_feats --brut donnees_brutes/hac
python preparation/hac_convertir.py --feats donnees_brutes/hac_feats --sortie donnees/hac --video slowfast --vues_fortes

# sentiment et esthétique (images : Google Drive des auteurs de M2CAN ; découpes : leur GitHub)
python preparation/jeux_m2can_extraire.py --groupe sentiment --brut donnees_brutes --sortie donnees/sentiment
python preparation/jeux_m2can_extraire.py --groupe aesthetic --brut donnees_brutes --sortie donnees/aesthetic
```

Le Google Drive des auteurs limite parfois les téléchargements (« quota dépassé ») : il suffit de relancer plus tard,
les fichiers déjà prêts ne sont pas retéléchargés.

## 2. Lancer les deux méthodes

```bash
python lancer_banc.py --feats donnees/hac --sortie sorties/hac --parallele 3
python lancer_banc.py --feats donnees/sentiment --sortie sorties/sentiment --parallele 3
python lancer_banc.py --feats donnees/aesthetic --sortie sorties/aesthetic --parallele 3
```

Par défaut : toutes les cibles, graines 123456, 234567 et 345678, et trois runs par (cible, graine) :

- `methode / methode` : la méthode d'`ALGORITHME.md`, avec pseudo-étiquettes ;
- `methode / methode_sans_pseudo` : la même sans pseudo-étiquettes (ablation) ;
- `m2can / publie` : M2CAN avec la configuration publiée de la cible (`m2can/msmm/scripts/config_<cible>.yaml`).

Options utiles : `--methodes methode` ou `--methodes m2can`, `--cibles Twitter`, `--graines 123456`,
`--appareil cpu`, `--parallele N` (plusieurs runs tiennent sur un même GPU).

**Taux d'apprentissage de M2CAN.** Par défaut, ceux de sa configuration publiée, prévus pour des encodeurs ajustés de
bout en bout. Sur caractéristiques gelées, ils figent presque ses couches entraînables, comme ils figeaient celles de la
méthode avant le passage de son extracteur à 5e-5. Pour un traitement symétrique, `configs/m2can.yaml` accepte
`extractor_lr`, `mlb_lr` et `classifier_lr` (exemple commenté dans le fichier).

**Nombre d'époques : 10 par défaut pour les deux méthodes.** Pour le changer :
`configs/methode.yaml` et `configs/methode_sans_pseudo.yaml` (`phase1: epoques:`) pour la méthode,
`configs/m2can.yaml` (`epochs:`) pour M2CAN.

M2CAN demande au moins 10 lots dans le train cible (40 exemples en HAC, 80 en sentiment et esthétique) : en dessous,
son pseudo-étiquetage divise par zéro (`print_iter = loader_length // 10` dans `msmm/models/model_msmm.py`). Les vrais
jeux en ont toujours bien plus ; cela ne concerne que les tests sur de tout petits sous-échantillons.

Chaque run écrit son journal dans `sorties/<groupe>/<méthode>_<config>/<cible>_s<graine>/journal.txt`.

## 3. Lire les résultats

```bash
python analyser.py sorties/sentiment/resultats.json
python analyser.py sorties/sentiment/resultats.json --mesure meilleur
```

Le tableau donne l'exactitude par méthode et par cible (moyenne ± écart-type sur les graines), puis l'écart de la
méthode à M2CAN, apparié par (cible, graine), avec son intervalle de confiance à 95 %. La mesure par défaut est le
**dernier point** d'évaluation : le meilleur point est choisi avec les étiquettes du test cible, il est donc optimiste.

## 4. Visualiser l'espace fusionné (t-SNE)

Ajouter dans `configs/methode.yaml` :

```yaml
phase2:
  sauver_z: true
```

Chaque run écrit alors `z_dernier.npz` (z des sources, de la cible de test et des barycentres, au dernier point) dans
son dossier de sortie. Puis :

```bash
python methode/visualiser_tsne.py --npz sorties/hac/methode_methode/human_s123456/z_dernier.npz --sortie tsne_human.pdf --cible human
```

Une seule projection t-SNE (distance cosinus, celle de l'étiquetage) : chaque source en points de couleur différente,
la cible en astérisques, les barycentres de Wasserstein en tirets violets.

## Organisation

```
methode/                 la méthode (phase1.py : entraînement ; phase2.py : étiquetage géométrique de la cible)
  ALGORITHME.md          description complète de l'algorithme
  test_synthetique.py    test de bout en bout sur données synthétiques (sans GPU)
m2can/                   code de M2CAN (auteurs), et m2can_carac/ : son adaptateur aux caractéristiques gelées
preparation/             téléchargement et extraction des caractéristiques
configs/                 configurations des runs
lancer_banc.py           lance les deux méthodes sur un jeu préparé
analyser.py              tableaux et comparaison appariée
resultats/               protocoles et résultats déjà obtenus
```

## Résultats déjà obtenus

Dernier point d'évaluation, moyenne sur 3 graines (123456, 234567, 345678), méthode entraînée **3 époques** (avant le
passage à 10 par défaut), M2CAN 10 époques. Comparaisons appariées par (cible, graine), 9 paires par groupe,
protocoles écrits avant les mesures (`resultats/`).

**HAC**, mêmes caractéristiques et mêmes vues fortes pour les deux méthodes :

| | human | animal | cartoon | moyenne |
|---|---|---|---|---|
| méthode | 86.96 | 85.98 | 62.39 | 78.44 |
| méthode sans pseudo-étiquettes | 88.04 | 71.78 | 56.42 | 72.08 |
| M2CAN | 85.75 | 84.47 | 71.26 | 80.49 |

Méthode − M2CAN : −2.05, IC95 [−6.60, +2.50] (indistinguable). Apport des pseudo-étiquettes : +6.36 [+0.91, +11.81].

**Sentiment et esthétique** :

| | TumEmo | Twitter | Yelp | AVA | PCCD | RPCD |
|---|---|---|---|---|---|---|
| méthode | 43.60 | 54.13 | 43.53 | 73.12 | 60.41 | 54.11 |
| méthode sans pseudo-étiquettes | 42.76 | 54.00 | 44.78 | 74.03 | 60.65 | 55.57 |
| M2CAN (configuration publiée) | 38.04 | 36.62 | 36.93 | 52.42 | 48.29 | 54.58 |
| hasard / classe majoritaire du test | 33.33 | 33.33 | 33.33 | 54.07 | 50.30 | 58.56 |
