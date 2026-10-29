# Données pour l'article : fusion géométrique contre M2CAN

Toutes les mesures portent sur des **caractéristiques gelées** (« cadre A »), 3 cibles × 3 graines
(123456, 234567, 345678) par famille, soit 9 runs par configuration.

## Fichiers

| dossier | contenu | section d'`ABLATIONS.md` |
|---|---|---|
| `ABLATIONS.md` | journal complet des expériences | toutes |
| `clip_s27/` | méthode, méthode sans pseudo-étiquettes, M2CAN publié ; sur HAC, aussi M2CAN extracteur 5e-5 | 27 (et 26 à 28) |
| `m2can_regle/` | M2CAN avec extracteur 5e-5 (`extracteur_5e-5`) et `taux_eleves`, esthétique et sentiment | 27 |
| `planchers_s29/` | régression logistique sources seules, auto-apprentissage naïf, oracle cible | 29 |
| `branche_wbhr_s30/` | variante fidèle à WBHR : `support_auto`, `fin_wbhr`, `methode_wbhr` | 30 |
| `h1h2_s30/` | H1 (barycentre contre concaténation) et H2 (hiérarchique contre plat), HAC, support `auto` | 30 |
| `banc_final_s31/` | **dernier banc** : `methode`, `methode_im03`, `methode_im1`, `methode_zn`, `methode_knn`, `methode_wbhr_zn`, avec le script d'analyse et sa sortie | 31 |
| `pilote_e2e_s31/` | méthode de bout en bout, AVA, 1 graine (arrêtée) | 31 |
| `protocoles/` | protocoles écrits **avant** les résultats (configurations, comparaisons, critère) | — |
| `configs_m2can/` | configurations publiées de M2CAN (`config_<cible>.yaml`) | — |
| `references.bib` | références à citer | — |

## Champs d'un `resultats.json`

Un objet par run : `methode` (`fgm` = notre méthode, `m2can`), `config`, `cible`, `graine`, `rc` (0 = run complet).
- `dernier` : exactitude au dernier point d'évaluation. **C'est la métrique rapportée.**
- `meilleur` : meilleur point. Choisi avec les étiquettes du test, donc optimiste ; jamais rapporté comme résultat.
- `oracle_dernier` : exactitude avec la meilleure bijection groupes → classes (erreur d'appariement = `oracle_dernier − dernier`).
- `plancher_dernier` : régression logistique sources → cible sur z (section 31 seulement).
- `tailles_dernier` : taille des groupes de la partition de la cible.
- `pseudo_premier` et `pseudo_dernier` : justesse des pseudo-étiquettes retenues.
- `critere` : point choisi par un critère d'arrêt sans étiquette (abandonné : −2 points, section 29).

Fichiers des planchers (`planchers_s29`) : un objet par (jeu, cible, graine) ; le champ `jeu` donne la famille et
les dimensions. 512 + 512 = CLIP ; 768 + 2048 = BERT + ResNet-50 ; HAC = 768 + 512.

## Statistiques

Écarts appariés par (cible, graine), 9 paires, IC95 de Student à 8 ddl. Critère :
- **MIEUX** : IC95 > 0 et au moins 7/9 paires positives ;
- **MOINS_BIEN** : IC95 < 0 ;
- **INDISTINGUABLE** (« non significatif ») : sinon.

Le script `banc_final_s31/analyse_dcff47f.py` recalcule tout depuis les json.

## Caractéristiques (banc final et sections 27 à 31)

- **Esthétique (AVA, PCCD, RPCD) et sentiment (TumEmo, Twitter, Yelp).** CLIP ViT-B/32 gelé
  (`openai/clip-vit-base-patch32`), projections texte 512-d et image 512-d.
- **HAC (human, animal, cartoon).**
  - Vidéo : CLIP ViT-B/32, 512-d, moyenne des 8 images TSN.
  - Audio : AST AudioSet (`MIT/ast-finetuned-audioset-10-10-0.4593`), 768-d.
- **Vues.** Toutes les familles ont 1 vue faible + 4 fortes, avec les mêmes augmentations que M2CAN.
- **README du dépôt.** Les combinaisons AST + SlowFast (HAC) et BERT + ResNet-50 (esthétique, sentiment) sont les
  **anciens bancs** (sections 23 à 25). CLIP les a remplacées : environ +10 points de vidéo sur HAC.
- **M2CAN reçoit exactement les mêmes vecteurs.** Ils sont complétés par des zéros jusqu'aux dimensions attendues :
  768 pour le texte, 2048 pour l'image ou la vidéo.

## M2CAN réglé

- **Esthétique et sentiment.** Configuration publiée de chaque cible (`configs_m2can/config_<cible>.yaml`).
  - Un seul changement : `extractor_lr` passe de 2e-5 à **5e-5**, le même taux que l'extracteur de notre méthode
    (« taux symétrique »).
  - Lot 8, 10 époques.
  - `taux_eleves` = 5e-5, plus `mlb_lr` et `classifier_lr` à 5e-4 : identique au publié sur l'esthétique.
- **HAC.** Configuration `config_Twitter.yaml` avec `extractor_lr` à 5e-5, lot 4, 10 époques.
- **Référence retenue pour comparer : `extracteur_5e-5`.**
  - HAC : dans `clip_s27/hac.json` ;
  - esthétique et sentiment : dans `m2can_regle/`.

## Notre méthode (banc final)

- Encodeurs gelés, extracteurs 256-d, fusion MLB z 256-d, pas de classifieur sur z.
- Pertes : têtes unimodales, CMCFA, CDCFA, SupCon sur z avec pseudo-étiquettes.
- Phase 2 : barycentres de Wasserstein, partition spectrale de la cible, transport hiérarchique, appariement hongrois.
- Entraînement : 3 époques, extracteur 5e-5.

## À ne pas rapporter

- **La partition spectrale « équilibrée ».** Elle impose des groupes de taille égale, ce que le professeur juge
  être de la triche. Elle est retirée de ces fichiers ; les sections 24, 25 et 27 d'`ABLATIONS.md` en parlent
  encore, à titre historique seulement.
- **Les `meilleur`.**
- **`methode_knn` contre M2CAN sur l'esthétique (+1.71).** Cette configuration a été choisie comme la meilleure des
  six sur le même test : c'est un biais de sélection.
- **Les chiffres 77.9 / 69.1 / 73.8 de M2CAN sur l'esthétique.** Ils viennent des commentaires des configurations,
  pas de l'article.
  - Valeurs de l'article, ResNet-50 + BERT : AVA 79.9, PCCD 69.8, RPCD 74.5 ; TumEmo 63.8, Twitter 84.7, Yelp 63.7.
  - Valeurs de l'article, Long-CLIP : 77.7 / 69.5 / 72.3 et 61.8 / 80.7 / 60.1.

## En cours

M2CAN de bout en bout (BERT et ResNet-50 entraînés, configuration publiée) sur Colab, esthétique. Le premier essai
a planté après l'époque 0 : ses chiffres sont sans valeur. Il a été relancé.

## Résumé chiffré (dernier point, moyenne des 9 runs)

| | HAC | esthétique | sentiment |
|---|---|---|---|
| méthode | 88.64 | 65.07 | 47.35 |
| meilleure variante (§31) | 89.08 (im03) | 65.70 (knn) | 50.41 (im1) |
| M2CAN extracteur 5e-5 | 88.36 | 63.99 | 55.80 |
| M2CAN publié | 83.16 | 60.74 | 45.93 |
| plancher sources seules | 77.15 | 60.03 | 51.38 |
| méthode − M2CAN 5e-5 | +0.27 [−3.68, +4.22], NS | +1.08 [−0.13, +2.28], 7/9, NS | −8.45 [−12.32, −4.58] |
