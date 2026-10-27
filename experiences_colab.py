"""Génère experiences_colab.ipynb (banc d'expériences pour Colab). Lancer : python experiences_colab.py"""
import json

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# Expériences sur la partition de la cible (Colab)

Ce carnet lance, sur GPU, les configurations qui agissent sur la **phase 1** (ce que z apprend) ou sur la **lecture de la
géométrie** (partition), puis compare chacune à la méthode (`methode/methode`), appariée par (cible, graine).

| configuration | ce qu'elle change | question |
|---|---|---|
| `methode` | rien (référence) | |
| `methode_wbhr` | support auto, argmax, classifieur logistique | fin fidèle à WBHR (déjà mesurée) |
| `methode_im03`, `methode_im1` | maximisation de l'information (SHOT) sur les têtes du lot cible, poids 0.3 / 1.0 | B3 : anti-effondrement sans prior d'équilibre |
| `methode_zn` | SupCon de la fusion sur z normalisé L2 | B5 : optimiser la géométrie que lit la phase 2 |
| `methode_wbhr_zn` | les deux précédentes (classifieur + z normalisé) | B4+B5 : les pseudo-étiquettes du classifieur tirent-elles z ? |
| `methode_knn` | partition spectrale sur graphe des 15 plus proches voisins | C7 : éviter le groupe géant sans imposer de taille |
| `methode_lr1e-5`, `methode_lr2e-4` | taux de l'extracteur | B6 : grille du sentiment |

Chaque run journalise aussi l'**oracle** (plafond de la partition) et le **plancher logistique** (régression logistique
sources → cible sur z, sans WBHR) : si l'oracle est bas, c'est la partition ; si le plancher est bas, c'est z.

**Avant de lancer** : mettre les caractéristiques préparées (`donnees/<famille>/meta.json`, ...) sur Google Drive, et
régler la cellule *Paramètres*. Les sorties vont sur Drive : une coupure de Colab ne perd rien, relancer la cellule
*Runs* reprend là où ça s'est arrêté (`--reprendre`).
""")

md("## Paramètres")
code("""
# --- à régler ---
DEPOT     = "https://github.com/Kevindjoum/MM-WBHR.git"
BRANCHE   = "claude/repo-access-lzj47d"
DRIVE     = "/content/drive/MyDrive/MM-WBHR"          # dossier Drive : DRIVE/donnees/<famille> en entrée, DRIVE/sorties en sortie
FAMILLES  = ["sentiment"]                              # parmi "hac", "sentiment", "aesthetic" ; le sentiment d'abord (c'est là que ça coince)
CONFIGS   = ["methode", "methode_im03", "methode_zn", "methode_knn", "methode_wbhr_zn"]   # ordre = priorité
GRAINES   = [123456, 234567, 345678]
CIBLES    = None                                       # None = toutes ; ou ["Twitter"] pour un essai court
PARALLELE = 3                                          # runs simultanés sur le GPU (3 tiennent sur un T4 avec des caractéristiques gelées)
GRILLE_LR = False                                      # True : ajoute methode_lr1e-5 et methode_lr2e-4 (B6) sur la première cible, graine 123456
""")

md("## Installation")
code("""
import os, subprocess, sys
from google.colab import drive
drive.mount("/content/drive")
assert os.path.isdir(f"{DRIVE}/donnees"), f"{DRIVE}/donnees introuvable : y mettre donnees/<famille>/meta.json ..."

if not os.path.isdir("/content/MM-WBHR"):
    subprocess.run(["git", "clone", "--branch", BRANCHE, DEPOT, "/content/MM-WBHR"], check=True)
else:
    subprocess.run(["git", "-C", "/content/MM-WBHR", "pull", "--ff-only"], check=True)
os.chdir("/content/MM-WBHR")
# torch est déjà installé sur Colab ; le reste est léger (pas les dépendances d'extraction HAC)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "scipy", "PyYAML", "tqdm"], check=True)
import torch
print("GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "AUCUN (Exécution > Modifier le type d'exécution > GPU)")
""")

md("""
## Données
Les caractéristiques sont copiées de Drive vers le disque local de Colab (lecture bien plus rapide pendant l'entraînement).
""")
code("""
import shutil, json
for fam in FAMILLES:
    src, dst = f"{DRIVE}/donnees/{fam}", f"/content/donnees/{fam}"
    if not os.path.isdir(dst):
        print(f"copie {fam} ...", end=" ", flush=True)
        shutil.copytree(src, dst)
        print("ok")
    meta = json.load(open(f"{dst}/meta.json"))
    print(fam, ":", meta["domaines"], "| modalités", meta["modalites"], "| classes", meta["classes"])
""")

md("""
## Runs
Un run méthode dure 15 à 20 min sur T4 (10 époques). Avec `PARALLELE = 3`, une famille × 5 configurations × 3 cibles ×
3 graines ≈ 45 runs ≈ 4 à 5 h. Les sorties (journaux, `historique.json`, `resultats.json`) vont dans `DRIVE/sorties/<famille>`.
Relancer cette cellule après une coupure : les runs déjà réussis sont repris, pas refaits.
""")
code("""
def lancer(fam, configs, cibles=None, graines=GRAINES):
    cmd = [sys.executable, "lancer_banc.py", "--feats", f"/content/donnees/{fam}", "--sortie", f"{DRIVE}/sorties/{fam}",
           "--methodes", "methode", "--configs", *configs, "--graines", *map(str, graines),
           "--parallele", str(PARALLELE), "--reprendre"]
    if cibles:
        cmd += ["--cibles", *cibles]
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode

for fam in FAMILLES:
    rc = lancer(fam, CONFIGS, CIBLES)
    print(f"=== {fam} : {'ok' if rc == 0 else 'des runs ont échoué, voir les journal.txt'}\\n", flush=True)

if GRILLE_LR:
    fam = FAMILLES[0]
    premiere = json.load(open(f"/content/donnees/{fam}/meta.json"))["domaines"][0]
    lancer(fam, ["methode_lr1e-5", "methode_lr2e-4"], [premiere], [GRAINES[0]])
""")

md("""
## Résultats
Pour chaque famille : exactitude au dernier point, oracle de la partition, plancher logistique, puis écart apparié de
chaque configuration à `methode/methode` avec son IC95.
""")
code("""
for fam in FAMILLES:
    res = f"{DRIVE}/sorties/{fam}/resultats.json"
    for mesure in ["dernier", "oracle", "plancher"]:
        print(f"\\n######## {fam} — {mesure}\\n")
        subprocess.run([sys.executable, "analyser.py", res, "--mesure", mesure, "--reference", "methode/methode"])
""")

md("""
## Synthèse
Un tableau par famille : configuration × cible (moyenne sur les graines), avec les colonnes « oracle » et « plancher »
de la méthode comme repères. Lecture :

- **exactitude ≈ oracle** : l'appariement est parfait, tout gain doit venir de la partition ou d'une correction après coup ;
- **oracle bas et plancher haut** : z est bon mais la partition le lit mal → `methode_knn` devrait aider ;
- **plancher bas** : z lui-même ne sépare pas les classes → c'est la phase 1 (`methode_im03`, `methode_zn`, taux).
""")
code("""
import pandas as pd
pd.set_option("display.width", 200)
for fam in FAMILLES:
    runs = [r for r in json.load(open(f"{DRIVE}/sorties/{fam}/resultats.json")) if r.get("rc") == 0]
    df = pd.DataFrame(runs)
    df["config"] = df["methode"] + "/" + df["config"]
    for col in ["dernier", "oracle_dernier", "plancher_dernier"]:
        df[col] = df[col].astype(float) * 100
    print(f"\\n######## {fam} — exactitude (dernier point), moyenne sur les graines")
    display(df.pivot_table(index="config", columns="cible", values="dernier", aggfunc="mean").round(2)
              .assign(moyenne=lambda t: t.mean(axis=1).round(2)))
    print(f"\\n{fam} — oracle de la partition et plancher logistique (moyenne cibles × graines)")
    display(df.groupby("config")[["dernier", "oracle_dernier", "plancher_dernier"]].mean().round(2))
""")

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
with open("experiences_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("experiences_colab.ipynb écrit")
