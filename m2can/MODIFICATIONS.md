# Code de M2CAN : origine et modifications

Ce dossier contient le code de **M2CAN** (Zhao et al., *Information Fusion* 117, 2025, 102862), repris du dépôt des
auteurs : https://github.com/jingjiang02/M2CAN (licence Apache 2.0, fichier `LICENSE`). Les découpes `split/` des
auteurs ne sont pas copiées : `preparation/jeux_m2can_extraire.py` les télécharge et vérifie leurs empreintes.

## Fichiers identiques à l'original

`common_modules/` (sauf `utils.py`), `data_loader/builder.py`, `data_loader/common_loader.py`, `data_loader/utils.py`,
`utils/loss_helper.py`, `utils/lr_helper.py`, `utils/__init__.py`, et les six `msmm/scripts/config_*.yaml`.

## Fichiers modifiés

| fichier | modification | raison |
|---|---|---|
| `common_modules/utils.py`, `msmm/models/model_msmm.py`, `msmm/train.py`, `utils/utils.py`, `test.py` | `.cuda()` remplacé par `.to(DEVICE)`, avec `DEVICE = cuda si disponible, sinon cpu` | détecter le GPU automatiquement, et pouvoir tourner sans GPU |
| `data_loader/base.py` | `encode_plus(..., pad_to_max_length=True)` remplacé par l'appel direct du tokenizer avec `padding='max_length'` | compatibilité avec les versions récentes de `transformers` (même résultat) |
| `config.py` | chemins par défaut (`data_list`, `data_root`, `bert_model`), `batch_size` 8 → 4, `workers` 8 → 0 | réglages de nos kernels Kaggle (chemins, nom Hugging Face de BERT) |

Les chemins et la taille de lot de `config.py` ne servent pas dans le cadre A : `m2can_carac/run.py` écrit lui-même
les listes de données, remet la taille de lot publiée (8) pour le sentiment et l'esthétique, et charge la
configuration publiée de chaque cible (`msmm/scripts/config_<cible>.yaml`).

## Ajout : `m2can_carac/`

Adaptateur qui fait tourner `msmm/train.py` **sans en modifier une ligne** sur des caractéristiques gelées au format
commun : il remplace à l'exécution l'entrée du modèle (BERT et ResNet-50 deviennent des caractéristiques déjà
extraites) et les chargeurs de données. Détails en tête de `m2can_carac/adapter.py`.
