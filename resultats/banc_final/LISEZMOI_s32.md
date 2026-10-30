# Résultats ajoutés après le banc final (31/10 et 1/11)

Fichiers fournis par l'utilisateur, non modifiés. Les chiffres cités ci-dessous sont lus dans ces fichiers ;
les mots « lecture » et « hypothèse » signalent une interprétation, pas une mesure.

## `m2can_bout_en_bout_AVA/` : M2CAN de bout en bout, AVA, graine 123456 (Colab, 263,9 min)

Configuration : `config_AVA.yaml` publiée, donc les valeurs par défaut de `config.py` (dorsales et extracteurs 2e-5, MLB
et classifieurs 5e-4, lot 8, 10 époques, SAM), qui correspondent au texte de l'article. Exactitude par époque :
58,68 ; 73,55 ; 77,10 ; 76,86 ; 76,27 ; 77,10 ; 76,62 ; 77,10 ; 77,57 ; **77,92**. Le dernier point est aussi le
meilleur. Article : 79,9. Le run PCCD a été interrompu (crédits épuisés), seul son début figure dans la discussion.

## `historiques_derive/` : la dérive, graine 123456

`TumEmo_3ep_*` (CLIP gelé, 3 époques, avec et sans pseudo-étiquettes), `AVA_3ep_*`, et les deux historiques AVA à
10 époques (50 évaluations chacun). Trois lectures :
- TumEmo, 3 époques : 59,7 au pas 1 500, puis 35,5 au pas 2 250 alors que l'oracle de la partition est à 58,4
  (erreur d'appariement), puis 45-47 jusqu'à la fin (partition dégradée, un groupe de ~900 points sur 1 500).
- AVA, 10 époques, sans pseudo-étiquettes : pic 74,1 (pas 9 000), dernier point 65,4 ; avec : pic 72,5, dernier 68,0.
  Le plancher logistique sur z baisse aussi (70,4 → 62,6 sans pseudo-étiquettes).
- M2CAN, lui, ne dérive pas (courbe ci-dessus).

## `dorsales_gelees_s32/` : test R1, bout en bout avec BERT et ResNet-50 gelés, AVA, graine 123456

Protocole écrit avant les résultats (`protocole_e2e_dorsales_gelees.json`). Mêmes réglages que la section 26
(3 × 847 = 2 541 pas). Dernier point : mode évaluation **51,59** (oracle 51,59) ; mode entraînement **45,93**
(oracle 54,07). Repères du protocole : bout en bout entraîné 58,91 ; caractéristiques gelées BERT/ResNet 72,73.
Lecture : à pas égal, le cadre à caractéristiques gelées est à 55,5 (pas 2 250) et ne décolle qu'entre 3 000 et
6 000 pas, donc le bout en bout s'arrête avant de décoller, ce qui désigne le budget de pas et non le pipeline. Le mode
entraînement des dorsales coûte en plus environ 6 points. Hypothèse, non testée : un budget de pas comparable et des
taux de fusion et de têtes dix fois plus élevés (R2, voir `kaggle_e2e/`).

## Expériences prévues, non lancées

Sentiment, 3 époques, 63 runs (≈ 5 h), contre la dérive :

```bash
python lancer_banc.py --feats donnees/sentiment --sortie sorties/derive --methodes methode \
  --configs methode methode_accord methode_accord3 methode_poids_m2can methode_tau02 methode_cmcfa_sources methode_quota50 \
  --parallele 3 --reprendre
```

À lire sur le dernier point **et** le meilleur point ; le succès est que l'écart entre les deux tombe vers 0 et que le
dernier point dépasse le plancher sources seules (51,38). Voir aussi `methode_amorcage` et `methode_calendrier_m2can`
(10 époques).
