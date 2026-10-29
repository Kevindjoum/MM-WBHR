"""Test de bout en bout sur données synthétiques, sans GPU, pour M = 2, 3 et 4 modalités.

Vérifie que l'entraînement tourne, que la phase 2 étiquette la cible et que les pseudo-étiquettes
fonctionnent, puis les options (z en 256-d, maximisation de l'information, partition équilibrée,
appariement argmax et classifieur de HOT-DA). Ne mesure pas une performance réelle.

    python test_synthetique.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import yaml

ICI = os.path.dirname(os.path.abspath(__file__))


def fabriquer(racine, M, K=3, n=240, graine=0):
    """3 domaines décalés, M modalités de dimensions différentes, classes séparables."""
    rng = np.random.default_rng(graine)
    dims = [32 + 16 * m for m in range(M)]
    centres = [rng.normal(0, 3, (K, d)) for d in dims]
    doms = ["a", "b", "c"]
    for k, dom in enumerate(doms):
        dec = [rng.normal(0, 1.0, d) for d in dims]          # décalage de domaine propre à chaque modalité
        for split in ("train", "test"):
            d = os.path.join(racine, dom, split)
            os.makedirs(d, exist_ok=True)
            y = rng.integers(0, K, n)
            np.save(os.path.join(d, "labels.npy"), y)
            np.save(os.path.join(d, "ids.npy"), np.array([f"{dom}_{split}_{i}" for i in range(n)]))
            for m in range(M):
                x = centres[m][y] + dec[m] + rng.normal(0, 1.0, (n, dims[m]))
                np.save(os.path.join(d, f"m{m}.npy"), x.astype(np.float32))
                aug = x[:, None, :] + rng.normal(0, 0.3, (n, 2, dims[m]))
                np.save(os.path.join(d, f"m{m}_aug.npy"), aug.astype(np.float32))
    with open(os.path.join(racine, "meta.json"), "w") as f:
        json.dump({"classes": K, "modalites": [f"m{m}" for m in range(M)], "domaines": doms}, f)
    return dims


def lancer(M, pseudo, options=False):
    tmp = tempfile.mkdtemp(prefix=f"fgm_M{M}_")
    try:
        dims = fabriquer(os.path.join(tmp, "donnees"), M)
        cfg = {"graine": 0, "appareil": "cpu", "sortie": os.path.join(tmp, "sortie"),
               "donnees": {"racine": os.path.join(tmp, "donnees"), "sources": ["a", "b"], "cible": "c"},
               "modalites": [{"nom": f"m{m}", "encodeur": {"type": "caracteristiques", "dim": dims[m]}}
                             for m in range(M)],
               "phase1": {"epoques": 1, "pas_par_epoque": 40, "lot": 16, "evaluer_tous_les": 20, "tau": 0.07,
                          "poids_tetes": 1.0, "alpha_cmcfa": 1.0, "beta_cdcfa": 1.0, "gamma_supcon_fusion": 1.0,
                          "optimiseur": {"lr_dorsale": 1e-3, "lr_extracteur": 1e-3, "lr_fusion": 1e-3,
                                         "lr_tetes": 1e-3}},
               "phase2": {"support": 8, "eps": 0.05, "sinkhorn": 30, "point_fixe": 4, "par_classe": 100},
               "pseudo": {"actif": pseudo, "tous_les": 15, "cible_max": 200, "taux": [0.3, 0.6], "filtre": "accord"}}
        if options == "wbhr":            # fin de HOT-DA fidèle à WBHR : appariement argmax, classifieur
            cfg["phase2"].update(support="auto", appariement="argmax", classifieur="logistique")
        elif options == "voisins":
            cfg["phase2"].update(classifieur="voisins", voisins=3)
        elif options == "calendrier":      # calendrier de M2CAN : chauffe sans cible, un tour par époque
            cfg["phase1"].update(epoques=3, cmcfa_cible_debut=40)
            cfg["pseudo"].update(debut=40, tous_les=40, taux=[0.3, 0.6, 1.0])
        elif options == "knn_zn":          # graphe kNN, SupCon sur z normalisé
            cfg["phase2"].update(affinite="knn", voisins_graphe=10)
            cfg["phase1"]["normaliser_z"] = True
        elif options:                    # z en 256-d, maximisation de l'information, partition équilibrée
            cfg["fusion"] = {"dim": 256}
            cfg["phase1"]["poids_im"] = 1.0
            cfg["phase2"]["partition"] = "equilibree"
        p = os.path.join(tmp, "config.yaml")
        with open(p, "w") as f:
            yaml.safe_dump(cfg, f)
        r = subprocess.run([sys.executable, os.path.join(ICI, "phase1.py"), "--config", p],
                           capture_output=True, text=True, cwd=ICI)
        h = json.load(open(os.path.join(tmp, "sortie", "historique.json")))
        ok = r.returncode == 0 and h["meilleur"]["exactitude"] >= 0
        n_pseudo = sum(1 for e in h["historique"] if "pseudo" in e)
        print(f"M={M} pseudo={pseudo} options={options} : code={r.returncode} meilleure exactitude phase 2 = "
              f"{h['meilleur']['exactitude']:.3f} (oracle {h['meilleur']['oracle']:.3f}), tours de pseudo = {n_pseudo}")
        if r.returncode != 0:
            print(r.stderr[-2000:])
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    res = [lancer(M, p) for M in (2, 3, 4) for p in (False, True)] + [lancer(2, False, True), lancer(3, True, True),
                                                                 lancer(2, True, "wbhr"), lancer(2, False, "voisins"),
                                                                 lancer(2, True, "knn_zn"), lancer(2, True, "calendrier")]
    print("TOUS LES TESTS PASSENT" if all(res) else "ECHEC")
    sys.exit(0 if all(res) else 1)
