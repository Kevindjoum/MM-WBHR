"""Phase 1 : entraînement de la fusion géométrique pour M modalités, avec évaluation par la phase 2.

Option : perte de maximisation de l'information sur la cible (`phase1.poids_im`, 0 par défaut).

Option : pseudo-étiquettes WBHR (section `pseudo` de la configuration). Tous les `tous_les` pas, la phase 2
étiquette le TRAIN cible ; on garde les points où la moyenne des têtes unimodales prédit la même classe que
WBHR (filtre `accord`), puis, par classe, les plus centraux de leur groupe, dans la limite d'un quota qui
croît d'un tour à l'autre. Ces points rejoignent les sources dans le SupCon sur la fusion.
`pseudo.debut` (0 par défaut) : aucun tour avant ce pas ; le quota repart de son premier taux au premier tour.
`pseudo.amorcage_tours` (0 par défaut) : les premiers tours prennent leurs pseudo-étiquettes aux têtes unimodales
(sélection par classe à parts égales, par confiance), la géométrie ensuite ; voir `pseudo_etiquettes`.
`phase1.cmcfa_cible_debut` (0 par défaut) : avant ce pas, la cible n'entre pas dans CMCFA (sources seules, comme
l'époque 0 de M2CAN). Calendrier de M2CAN (une époque de chauffe, un tour de pseudo-étiquettes par époque, quotas
0.3 / 0.6 / 0.9 / 1.0 par époque) : configs/methode_calendrier_m2can.yaml.

    python phase1.py --config config.yaml
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from donnees import EchantillonneurDomaines, JeuCaracteristiques, lire_meta
from modele import ModeleFusionGeometrique
from pertes import pertes_phase1
import phase2


def groupes_parametres(modele, opt):
    g = []
    dors = [p for e in modele.encodeurs.values() for p in e.dorsale.parameters()]
    if dors:
        g.append({"params": dors, "lr": opt["lr_dorsale"]})
    g.append({"params": [p for e in modele.encodeurs.values() for p in e.extracteur.parameters()],
              "lr": opt["lr_extracteur"]})
    g.append({"params": list(modele.fusion.parameters()), "lr": opt["lr_fusion"]})
    g.append({"params": list(modele.tetes.parameters()), "lr": opt["lr_tetes"]})
    return g


def pseudo_etiquettes(modele, src, cib_train, K, conf2, confp, tour):
    """Étiquette le train cible par WBHR et renvoie ({id: classe}, journal du tour).

    Option `pseudo.amorcage_tours` (0 par défaut) : pendant les premiers tours, les pseudo-étiquettes viennent de la
    moyenne des têtes unimodales (classe prédite, score = confiance), sélectionnées par classe à parts égales, comme
    la sélection de M2CAN ; la géométrie ne sert qu'à partir du tour `amorcage_tours`. Utile quand z n'a pas encore de
    structure (bout en bout) : une partition effondrée donne des étiquettes effondrées, les têtes, ancrées par les
    vraies étiquettes sources, non.
    """
    amorcage = tour < int(confp.get("amorcage_tours", 0))
    was = modele.training
    modele.eval()
    try:
        zt, pt, ids, yt = phase2.representations(modele, cib_train, confp.get("cible_max", 3000))
        if not amorcage:
            s = [phase2.representations(modele, j, conf2.get("sources_max")) for j in src]
            res = phase2.etiqueter([x[0] for x in s], [x[3] for x in s], zt, K, conf2, graine=tour, pt=pt)
    finally:
        if was:
            modele.train()
    if amorcage:
        yw, marge = pt.argmax(1), pt.max(1)
        ok = np.ones(len(yw), bool)
    else:
        yw, marge = res["etiquettes"], res["marge"]
        ok = (pt.argmax(1) == yw) if confp.get("filtre", "accord") == "accord" else np.ones(len(yw), bool)
    taux = confp.get("taux", [0.3, 0.6, 0.9, 1.0])
    quota = int(taux[min(tour, len(taux) - 1)] * len(yw) / K)
    garde = []
    for c in range(K):
        cand = np.where((yw == c) & ok)[0]
        garde.extend(cand[np.argsort(-marge[cand])][:quota].tolist())
    garde = np.array(garde, dtype=int)
    journal = {"tour": tour, "source": "tetes" if amorcage else "wbhr",
               "exactitude_wbhr_complet": float((yw == yt).mean()), "n_retenus": int(len(garde)),
               "precision_retenus": float((yw[garde] == yt[garde]).mean()) if len(garde) else None}
    return {ids[j]: int(yw[j]) for j in garde}, journal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pas_max", type=int, default=None, help="arrêt anticipé (test rapide)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    app = cfg.get("appareil", "auto")
    if app == "auto":
        app = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[appareil] {app}", flush=True)
    graine = int(cfg.get("graine", 0))
    torch.manual_seed(graine); np.random.seed(graine)
    meta = lire_meta(cfg["donnees"]["racine"])
    K = meta["classes"]
    noms = [m["nom"] for m in cfg["modalites"]]
    r, sorties = cfg["donnees"]["racine"], cfg["sortie"]
    os.makedirs(sorties, exist_ok=True)

    src = [JeuCaracteristiques(r, d, "train", noms, app) for d in cfg["donnees"]["sources"]]
    cib_train = JeuCaracteristiques(r, cfg["donnees"]["cible"], "train", noms, app)
    cib_test = JeuCaracteristiques(r, cfg["donnees"]["cible"], "test", noms, app)
    ech = EchantillonneurDomaines(src + [cib_train], cfg["phase1"]["lot"], graine)

    modele = ModeleFusionGeometrique(cfg["modalites"], K, (cfg.get("fusion") or {}).get("dim")).to(app)
    p1, opt = cfg["phase1"], cfg["phase1"]["optimiseur"]
    optim = torch.optim.Adam(groupes_parametres(modele, opt), lr=opt["lr_fusion"], betas=(0.9, 0.999), eps=1e-8)
    total = p1["epoques"] * p1["pas_par_epoque"]
    poly = lambda t: max((1 - t / total) ** opt.get("puissance", 0.9), opt.get("lr_min", 1e-8) / opt["lr_fusion"])
    sched = torch.optim.lr_scheduler.LambdaLR(optim, [poly] * len(optim.param_groups))

    confp = cfg.get("pseudo", {}) or {}
    conf2 = cfg.get("phase2", {}) or {}
    pseudo, tour, histo, meilleur = {}, 0, [], {"exactitude": -1}
    t0, pas = time.time(), 0
    for ep in range(p1["epoques"]):
        for s in range(p1["pas_par_epoque"]):
            pas += 1
            if confp.get("actif") and pas % confp.get("tous_les", 750) == 0 and pas >= confp.get("debut", 0):
                pseudo, j = pseudo_etiquettes(modele, src, cib_train, K, conf2, confp, tour)
                tour += 1
                print("[pseudo] " + json.dumps(j), flush=True)
                histo.append({"pas": pas, "pseudo": j})
            lots = ech.suivant()
            lt, lc, ld, ls, li = pertes_phase1(modele, lots, p1["tau"], pseudo, p1.get("normaliser_z", False),
                                              pas > p1.get("cmcfa_cible_debut", 0))
            perte = p1["poids_tetes"] * lt + p1["alpha_cmcfa"] * lc + p1["beta_cdcfa"] * ld + p1["gamma_supcon_fusion"] * ls
            if p1.get("poids_im", 0):
                perte = perte + p1["poids_im"] * li
            optim.zero_grad()
            perte.backward()
            optim.step()
            sched.step()
            if pas % 50 == 0:
                print(f"ep{ep} pas{pas}/{total} perte={perte.item():.4f} tetes={lt.item():.4f} cmcfa={lc.item():.4f} "
                      f"cdcfa={ld.item():.4f} sc_fusion={ls.item():.4f} im={li.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
            fin = (s + 1 == p1["pas_par_epoque"])
            if pas % p1["evaluer_tous_les"] == 0 or fin or pas == args.pas_max:
                d = phase2.evaluer(modele, src, cib_test, K, conf2, graine,
                                   cib_train if conf2.get("evaluation_jointe") else None, confp.get("cible_max", 3000),
                                   os.path.join(sorties, "z_dernier.npz") if conf2.get("sauver_z") else None)
                d.update({"epoque": ep, "pas": pas})
                histo.append(d)
                print("[phase2] " + json.dumps(d), flush=True)
                if d["exactitude"] > meilleur["exactitude"]:
                    meilleur = d
                    torch.save(modele.state_dict(), os.path.join(sorties, "modele_meilleur.pt"))
                with open(os.path.join(sorties, "historique.json"), "w") as f:
                    json.dump({"config": cfg, "meilleur": meilleur, "historique": histo}, f, indent=1)
            if args.pas_max and pas >= args.pas_max:
                break
        if args.pas_max and pas >= args.pas_max:
            break
    torch.save(modele.state_dict(), os.path.join(sorties, "modele_final.pt"))
    print(f"[fin] meilleur point (choisi sur la cible, optimiste) : {json.dumps(meilleur)}")


if __name__ == "__main__":
    main()
