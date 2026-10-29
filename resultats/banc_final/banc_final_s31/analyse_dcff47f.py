"""Analyse des configurations de la branche (commit dcff47f), protocoles protocole_branche_dcff47f_<groupe>.json.

Ecarts apparies par (cible, graine), dernier point ; IC95 t a 8 ddl ; criteres des protocoles.
Secondaire : meilleure configuration contre M2CAN extracteur 5e-5 (section 27).
"""
import json
import os

import numpy as np
from scipy import stats

D = os.path.dirname(os.path.abspath(__file__))


def charger(dossier):
    R = {}
    for r in json.load(open(os.path.join(D, dossier, "resultats.json"), encoding="utf-8")):
        R[(r["methode"] + "/" + r["config"], r["cible"], r["graine"])] = r
    return R


def v(R, k, c, g, m="dernier"):
    x = R[(k, c, g)].get(m)
    return float("nan") if x is None else x * 100


def verdict(d):
    m = d.mean()
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    pos = int((d > 0).sum())
    lab = "MIEUX" if (m - h > 0 and pos >= 7) else ("MOINS_BIEN" if m + h < 0 else "INDISTINGUABLE")
    return f"{m:+6.2f}  IC95 [{m - h:+6.2f}, {m + h:+6.2f}]  {pos}/{len(d)} positifs  -> {lab}", \
        {"moyenne": round(m, 3), "ic95": [round(m - h, 3), round(m + h, 3)], "positifs": pos, "verdict": lab}


M2CAN = {"hac": ("sortie_clip_hac", "m2can/extracteur_5e-5"),
         "aesthetic": ("sortie_m2can_taux_aesthetic", "m2can/extracteur_5e-5"),
         "sentiment": ("sortie_m2can_taux_sentiment", "m2can/extracteur_5e-5")}
out = {}
for g in ("hac", "aesthetic", "sentiment"):
    proto = json.load(open(os.path.join(D, f"protocole_branche_dcff47f_{g}.json"), encoding="utf-8"))
    R = charger(f"sortie_dcff47f_{g}")
    cib, gr = proto["cibles"], proto["graines"]
    confs = list(proto["configurations"]["fgm"])
    print(f"\n===== {g} : {len(R)} runs, rc != 0 : {[k for k, r in R.items() if r['rc'] != 0]}")
    print("dernier point, moyenne 3 graines (plancher logistique)")
    moy = {}
    for k in confs:
        kk = "fgm/" + k
        moy[k] = np.mean([v(R, kk, c, s) for c in cib for s in gr])
        print(f"  {k:18s}" + "".join(f"  {c} {np.mean([v(R, kk, c, s) for s in gr]):6.2f} "
                                     f"({np.mean([v(R, kk, c, s, 'plancher_dernier') for s in gr]):6.2f})" for c in cib)
              + f"   moy {moy[k]:6.2f}")
    out[g] = {"moyennes": {k: round(x, 3) for k, x in moy.items()}, "contre_methode": {}}
    for k in confs[1:]:
        d = np.array([v(R, "fgm/" + k, c, s) - v(R, "fgm/methode", c, s) for c in cib for s in gr])
        txt, res = verdict(d)
        out[g]["contre_methode"][k] = res
        print(f"  {k:18s} - methode : {txt}")
    print("  tailles des groupes au dernier point (graine 123456) :")
    for k in confs:
        print(f"    {k:18s}", {c: R[("fgm/" + k, c, 123456)].get("tailles_dernier") for c in cib})
    meilleure = max(confs, key=lambda k: moy[k])
    dm, km = M2CAN[g]
    RM = charger(dm)
    print(f"  M2CAN extracteur 5e-5 : " + "  ".join(f"{c} {np.mean([v(RM, km, c, s) for s in gr]):6.2f}" for c in cib)
          + f"   moy {np.mean([v(RM, km, c, s) for c in cib for s in gr]):6.2f}")
    for k in sorted({meilleure, "methode"}):
        d = np.array([v(R, "fgm/" + k, c, s) - v(RM, km, c, s) for c in cib for s in gr])
        txt, res = verdict(d)
        out[g][f"{k} - M2CAN 5e-5"] = res
        print(f"  {k:18s} - M2CAN 5e-5 : {txt}" + ("   (meilleure configuration)" if k == meilleure else ""))
json.dump(out, open(os.path.join(D, "analyse_dcff47f.json"), "w"), indent=1)
