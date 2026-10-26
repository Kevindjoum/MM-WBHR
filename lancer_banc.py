"""Lance la méthode et M2CAN sur un jeu de caractéristiques préparé, pour chaque cible et chaque graine.

    python lancer_banc.py --feats donnees/sentiment --sortie sorties/sentiment
    python lancer_banc.py --feats donnees/hac --sortie sorties/hac --methodes methode --cibles human --graines 123456

Chaque run est un sous-processus indépendant (sa sortie dans <sortie>/<méthode>_<config>/<cible>_s<graine>/journal.txt).
Le résumé de tous les runs est écrit dans <sortie>/resultats.json ; analyser.py en tire les tableaux.
GPU utilisé s'il existe (--appareil auto), sinon processeur.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

ICI = os.path.dirname(os.path.abspath(__file__))
CONFIGS_METHODE = {"methode": "configs/methode.yaml", "methode_sans_pseudo": "configs/methode_sans_pseudo.yaml",
                   "methode_wbhr": "configs/methode_wbhr.yaml"}


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def appareil(choix):
    if choix != "auto":
        return choix
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def dimension(meta, feats, m):
    """Dimension de la modalite m : dans meta.json si presente, sinon lue dans le premier fichier .npy."""
    if "dims" in meta and m in meta["dims"]:
        return int(meta["dims"][m])
    import numpy as np
    return int(np.load(os.path.join(feats, meta["domaines"][0], "train", f"{m}.npy"), mmap_mode="r").shape[1])


def resume_methode(path):
    h = json.load(open(path, encoding="utf-8"))
    ev = [e for e in h["historique"] if "exactitude" in e]
    k = min(range(len(ev)), key=lambda i: ev[i].get("ratio_intra_inter") or 9e9)
    ps = [e["pseudo"] for e in h["historique"] if "pseudo" in e]
    return {"dernier": ev[-1]["exactitude"], "meilleur": max(e["exactitude"] for e in ev), "critere": ev[k]["exactitude"],
            "oracle_dernier": ev[-1]["oracle"], "n_evals": len(ev),
            "pseudo_dernier_tour": ps[-1] if ps else None}


def run(meth, conf, cible, graine, args, meta, dev):
    sortie = os.path.abspath(os.path.join(args.sortie, f"{meth}_{conf}", f"{cible}_s{graine}"))
    os.makedirs(sortie, exist_ok=True)
    t0 = time.time()
    if meth == "methode":
        cfg = yaml.safe_load(open(os.path.join(ICI, "methode", "config_exemple.yaml"), encoding="utf-8"))
        deep_update(cfg, yaml.safe_load(open(os.path.join(ICI, CONFIGS_METHODE[conf]), encoding="utf-8")) or {})
        cfg["modalites"] = [{"nom": m, "encodeur": {"type": "caracteristiques", "dim": dimension(meta, args.feats, m)}}
                            for m in meta["modalites"][:2]]
        cfg["donnees"] = {"racine": os.path.abspath(args.feats), "cible": cible,
                          "sources": [d for d in meta["domaines"] if d != cible]}
        cfg.update(graine=graine, appareil=dev, sortie=sortie)
        cp = os.path.join(sortie, "config.yaml")
        yaml.safe_dump(cfg, open(cp, "w", encoding="utf-8"), allow_unicode=True)
        cmd, cwd = [sys.executable, "phase1.py", "--config", cp], os.path.join(ICI, "methode")
    else:
        opts = yaml.safe_load(open(os.path.join(ICI, "configs", "m2can.yaml"), encoding="utf-8")) or {}
        cmd = [sys.executable, "m2can_carac/run.py", "--feats", os.path.abspath(args.feats), "--target", cible,
               "--out", sortie, "--seed", str(graine)]
        for k, v in opts.items():
            cmd += [f"--{k}", str(v)]
        cwd = os.path.join(ICI, "m2can")
    env = dict(os.environ)
    if dev == "cpu":                    # M2CAN choisit son appareil seul : on lui cache le GPU
        env["CUDA_VISIBLE_DEVICES"] = ""
    with open(os.path.join(sortie, "journal.txt"), "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    res = {"methode": meth, "config": conf, "cible": cible, "graine": graine, "rc": rc,
           "minutes": round((time.time() - t0) / 60, 1)}
    try:
        if meth == "methode":
            res.update(resume_methode(os.path.join(sortie, "historique.json")))
        else:
            r = json.load(open(os.path.join(sortie, "results.json"), encoding="utf-8"))
            accs = [e["acc"] for e in r["evals"]]
            res.update(dernier=accs[-1], meilleur=max(accs), n_evals=len(accs))
    except Exception as e:
        res["erreur"] = f"{e!r} ; voir {os.path.join(sortie, 'journal.txt')}"
    print("[run] " + json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description="Méthode et M2CAN sur des caractéristiques préparées")
    ap.add_argument("--feats", required=True, help="dossier préparé (meta.json + <domaine>/<split>/...)")
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--methodes", nargs="+", default=["methode", "m2can"], choices=["methode", "m2can"])
    ap.add_argument("--configs", nargs="+", default=["methode", "methode_sans_pseudo"], choices=list(CONFIGS_METHODE),
                    help="configurations de la méthode")
    ap.add_argument("--cibles", nargs="+", default=None, help="défaut : tous les domaines de meta.json")
    ap.add_argument("--graines", nargs="+", type=int, default=[123456, 234567, 345678])
    ap.add_argument("--parallele", type=int, default=1, help="runs simultanés (plusieurs tiennent sur un GPU)")
    ap.add_argument("--appareil", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.feats, "meta.json"), encoding="utf-8"))
    cibles = args.cibles or list(meta["domaines"])
    dev = appareil(args.appareil)
    taches = []
    for c in cibles:
        for g in args.graines:
            if "methode" in args.methodes:
                taches += [("methode", conf, c, g) for conf in args.configs]
            if "m2can" in args.methodes:
                taches.append(("m2can", "publie", c, g))
    os.makedirs(args.sortie, exist_ok=True)
    print(f"{len(taches)} runs sur {dev}, {args.parallele} à la fois ; caractéristiques {args.feats}", flush=True)
    with ThreadPoolExecutor(args.parallele) as ex:
        res = list(ex.map(lambda t: run(*t, args, meta, dev), taches))
    chemin = os.path.join(args.sortie, "resultats.json")
    anciens = json.load(open(chemin, encoding="utf-8")) if os.path.exists(chemin) else []
    cle = lambda r: (r["methode"], r["config"], r["cible"], r["graine"])
    fusion = {cle(r): r for r in anciens}
    fusion.update({cle(r): r for r in res})
    json.dump(list(fusion.values()), open(chemin, "w", encoding="utf-8"), indent=1)
    print(f"résumé : {chemin} ; tableaux : python analyser.py {chemin}")
    sys.exit(0 if all(r["rc"] == 0 for r in res) else 1)


if __name__ == "__main__":
    main()
