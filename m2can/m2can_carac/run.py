"""Lance M2CAN (msmm/train.py, code d'origine non modifie) sur des caracteristiques gelees au format commun.

    python m2can_carac/run.py --feats <racine> --target <domaine> --out <dossier> [--seed 123456] [--epochs 10]

Le dossier parent de m2can_carac/ doit etre la racine du code M2CAN (m2can1-full-code).

Protocole : leave-one-domain-out ; sources et hyperparametres = configuration publiee de la cible (voir make_config). Sources = train
etiquete ; cible = train NON etiquete pour l'adaptation, evaluation sur son test, pris exactement une fois.
10 epoques (valeur par defaut de M2CAN). Seule l'entree change (adapter.py).
"""
import argparse
import copy
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)


def make_config(meta, listes, target, seed, out_dir, n_sample, epochs, batch_size):
    """Configuration M2CAN de la cible : la configuration PUBLIEE msmm/scripts/config_<cible>.yaml quand elle existe
    (sentiment, esthetique : ordre des sources, taux d'apprentissage, poids des pertes, nombre de classes), dont seules
    les listes de donnees sont remplacees ; sinon (HAC) celle de Twitter, comme pour les runs HAC de reference.
    Taille de lot : 8 (valeur par defaut publiee de M2CAN) avec une configuration publiee, 4 sinon (runs HAC).
    Taille d'epoque : n_sample par defaut de M2CAN pour la cible (15000 en sentiment, 3388 en esthetique), 15000 sinon."""
    import config as m2can_config
    doms = list(meta["domaines"])
    publie = os.path.join(CODE_ROOT, "msmm", "scripts", f"config_{target}.yaml")
    if os.path.exists(publie):
        with open(publie) as f:
            cfg = yaml.safe_load(f)
        types = list(cfg["dataset"]["types"])
        assert types[-1] == target and set(types) <= set(doms), (types, doms)
        lot = 8 if batch_size is None else batch_size
    else:
        with open(os.path.join(CODE_ROOT, "msmm", "scripts", "config_Twitter.yaml")) as f:
            pub = yaml.safe_load(f)
        types = [d for d in doms if d != target] + [target]
        cfg = {"net": {"num_classes": int(meta["classes"]), "start_rate": 0.3, "update_rate": 0.3},
               "trainer": copy.deepcopy(pub["trainer"]), "criterion": copy.deepcopy(pub["criterion"])}
        lot = 4 if batch_size is None else batch_size
    assert int(cfg["net"]["num_classes"]) == int(meta["classes"]), (cfg["net"], meta["classes"])
    defauts = m2can_config.DEFAULTS["dataset"]["train"]
    train = {d: {"data_list": listes[(d, "train")], "data_root": "",
                 "n_sample": n_sample or defauts.get(d, {}).get("n_sample", 15000)} for d in types}
    val = {}
    for d in types:
        with open(listes[(d, "test")]) as f:
            val[d] = {"data_list": listes[(d, "test")], "data_root": "", "n_sample": len(json.load(f))}
    cfg.setdefault("dataset", {})
    cfg["dataset"].update({"batch_size": lot, "types": types, "train": train, "val": val})
    cfg["seed"], cfg["snapshot_dir"] = seed, out_dir
    if epochs is not None:
        cfg.setdefault("trainer", {})["epochs"] = epochs
    path = os.path.join(out_dir, f"config_{target}.yaml")
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    print(f"[m2can] configuration : {'publiee ' + os.path.basename(publie) if os.path.exists(publie) else 'Twitter (HAC)'}"
          f", sources {types[:-1]}, cible {target}, lot {lot}, n_sample {train[target]['n_sample']}", flush=True)
    return path


def summarize(out, target, seed):
    with open(os.path.join(out, "results.json")) as f:
        r = json.load(f)
    accs = [e["acc"] for e in r["evals"]]
    res = {"method": "m2can", "target": target, "seed": seed, "best": max(accs), "last": accs[-1], "evals": r["evals"]}
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("[m2can-result] " + json.dumps({k: res[k] for k in ("method", "target", "seed", "best", "last")}), flush=True)


def main():
    ap = argparse.ArgumentParser(description="M2CAN sur caracteristiques gelees (format commun)")
    ap.add_argument("--feats", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=123456, help="123456 = graine publiee de M2CAN")
    ap.add_argument("--n_sample", type=int, default=None, help="taille d'une epoque ; defaut : celle de M2CAN pour la cible")
    ap.add_argument("--batch_size", type=int, default=None, help="defaut : 8 (config publiee) ou 4 (HAC)")
    ap.add_argument("--epochs", type=int, default=None, help="defaut : 10 (M2CAN)")
    args = ap.parse_args()

    out, feats = os.path.abspath(args.out), os.path.abspath(args.feats)
    os.makedirs(out, exist_ok=True)
    sys.path.insert(0, CODE_ROOT)
    os.chdir(CODE_ROOT)
    from m2can_carac import adapter

    entry = adapter.install(feats, "m2can", os.path.join(out, "results.json"), args.target)
    listes = adapter.STORE.ecrire_listes(os.path.join(out, "listes"))
    cfg_path = make_config(adapter.STORE.meta, listes, args.target, args.seed, out, args.n_sample, args.epochs,
                           args.batch_size)
    print("[m2can] train.py --config " + cfg_path, flush=True)
    sys.argv = ["train.py", "--config", cfg_path]
    entry()
    summarize(out, args.target, args.seed)


if __name__ == "__main__":
    main()
