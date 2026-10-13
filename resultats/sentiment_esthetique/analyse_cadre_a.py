"""Analyse des bancs cadre A (protocole_<groupe>.json) : la methode contre M2CAN, appariee par (cible, graine).

    python analyse_cadre_a.py sentiment|aesthetic
Entree : log_bench_<groupe>.txt (lignes [run]). Sortie : analyse_<groupe>.json, et le texte sur la sortie standard.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

D = os.path.dirname(os.path.abspath(__file__))
G = sys.argv[1]
proto = json.load(open(os.path.join(D, f"protocole_{G}.json"), encoding="utf-8"))
R = {}
for l in open(os.path.join(D, f"log_bench_{G}.txt"), encoding="utf-8"):
    if l.startswith("[run] "):
        r = json.loads(l[6:])
        R[(r["config"], r["cible"], r["graine"])] = r
cles = [(c, g) for c in proto["cibles"] for g in proto["graines"]]
noms = {"methode": "methode", "methode_sans_pseudo": "methode_sans_pseudo", "M2CAN": "publie"}
print(f"{G} : {len(R)} runs, rc != 0 : {sum(r['rc'] != 0 for r in R.values())}\n")


def acc(n, c, g, m="dernier"):
    return R[(noms[n], c, g)][m] * 100


print("Dernier point par graine, moyenne (meilleur point moyen, plafond optimiste) (%)")
maj = proto["reperes"]["classe_majoritaire_test"]
for c in proto["cibles"]:
    print(f"  {c}   (classe majoritaire du test : {maj[c]})")
    for n in noms:
        x = [acc(n, c, g) for g in proto["graines"]]
        b = [acc(n, c, g, "meilleur") for g in proto["graines"]]
        print(f"    {n:20s} " + "  ".join(f"{v:6.2f}" for v in x) + f"   moy {np.mean(x):6.2f}  (meilleur {np.mean(b):6.2f})")
print("  moyenne sur les cibles")
for n in noms:
    print(f"    {n:20s} {np.mean([acc(n, c, g) for c, g in cles]):6.2f}")


def bilan(nom, d):
    d = np.array(d)
    m, n = d.mean(), len(d)
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    pos = int((d > 0).sum())
    v = "MIEUX" if (m - h > 0 and pos >= 7) else ("MOINS_BIEN" if m + h < 0 else "INDISTINGUABLE")
    print(f"  {nom:40s} {m:+6.2f}  IC95 [{m - h:+6.2f}, {m + h:+6.2f}]  {pos}/{n} positifs  -> {v}")
    return {"moyenne": round(m, 3), "ic95": [round(m - h, 3), round(m + h, 3)], "positifs": pos, "n": n, "verdict": v,
            "ecarts": d.round(3).tolist()}


print("\nComparaisons appariees (9 paires cible x graine, dernier point)")
out = {
    "principale_methode_moins_M2CAN": bilan("methode - M2CAN", [acc("methode", c, g) - acc("M2CAN", c, g) for c, g in cles]),
    "sans_pseudo_moins_M2CAN": bilan("methode_sans_pseudo - M2CAN", [acc("methode_sans_pseudo", c, g) - acc("M2CAN", c, g) for c, g in cles]),
    "apport_pseudo": bilan("methode - methode_sans_pseudo", [acc("methode", c, g) - acc("methode_sans_pseudo", c, g) for c, g in cles]),
}
print("\nPar cible, ecart moyen methode - M2CAN")
for c in proto["cibles"]:
    print(f"  {c:8s} {np.mean([acc('methode', c, g) - acc('M2CAN', c, g) for g in proto['graines']]):+6.2f}")
print("\nTailles des groupes au dernier point (methode) et justesse des pseudo-etiquettes retenues (dernier tour)")
for c, g in cles:
    r = R[("methode", c, g)]
    pd = r.get("pseudo_dernier") or {}
    print(f"  {c:8s} {g}  groupes {r.get('tailles_dernier')}  pseudo : {pd.get('n_retenus')} retenus, "
          f"justesse {pd.get('precision_retenus')}")
json.dump(out, open(os.path.join(D, f"analyse_{G}.json"), "w"), indent=1)
