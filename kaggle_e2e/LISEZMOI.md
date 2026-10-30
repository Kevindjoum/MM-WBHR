# Générateur du bout en bout (Kaggle), version patchée pour R2

`e2e_build.py` est le fichier `kaggle/e2e/build.py` de l'autre dépôt (celui des expériences Kaggle), **patché**.
Il ne tourne pas seul : il importe `kaggle/brasc_drive/build.py` et `template.py` de ce dépôt-là. Il est conservé ici
pour ne pas perdre le patch ; à recopier sur `kaggle/e2e/build.py`. `e2e_build.patch` est le diff (29 lignes, fins de
ligne LF ; l'original est en CRLF, d'où `diff --strip-trailing-cr`).

Options ajoutées, toutes à valeur par défaut = comportement de la section 26 :

| option | défaut | rôle |
|---|---|---|
| `--lr_dorsale` | 2e-6 | taux de BERT et ResNet-50 (texte de M2CAN : 2e-5) |
| `--lr_fusion` | 5e-5 | taux de la fusion MLB (M2CAN `mlb_lr` : 5e-4) |
| `--lr_tetes` | 5e-5 | taux des têtes (M2CAN `classifier_lr` : 5e-4) |
| `--alpha_cmcfa` | 1.0 | poids de CMCFA (`config_AVA.yaml` : 0.5) |
| `--bn_eval` | non | BatchNorm de ResNet-50 figée pendant l'affinage |
| `--calendrier_m2can` | non | époque 1 sur les sources seules, un tour de pseudo-étiquettes par époque, deux tours amorcés par les têtes |

Commande R2 (AVA, graine 123456, à lancer après `git pull` de la branche, dont `methode/` est embarqué par `--paquet`) :

```bash
python kaggle/e2e/build.py --groupe aesthetic --cibles AVA --graines 123456 --account compte3 \
  --full_code <code_ref/full_code> --diff_code <code_ref/diff_code> --paquet <MM-WBHR/methode> \
  --epoques 10 --evaluer 169 \
  --lr_dorsale 2e-5 --lr_extracteur 2e-5 --lr_fusion 5e-4 --lr_tetes 5e-4 --alpha_cmcfa 0.5 \
  --bn_eval --calendrier_m2can --push
```

Attendu (hypothèse, non mesuré) : décollage avant l'époque 4, dernier point ≥ 70. Référence : M2CAN de bout en bout,
77,92 (`resultats/banc_final/m2can_bout_en_bout_AVA/`).
