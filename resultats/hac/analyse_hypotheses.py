"""Analyse de protocole_hypotheses.json : H1 (barycentre contre concatenation) et H2 (hierarchique contre plat) sur HAC.

Entrees : log_hyp_<cible>.txt (lignes [run]) et sortie_hyp_<cible>/sorties/<cible>/<config>_s<graine>/historique.json.
Analyse partielle possible : seules les cibles presentes sont traitees, et le verdict n'est rendu qu'avec les 9 paires.
"""
import json
import os

import numpy as np
from scipy import stats

D = os.path.dirname(os.path.abspath(__file__))
proto = json.load(open(os.path.join(D, "protocole_hypotheses.json"), encoding="utf-8"))
CONFIGS = list(proto["configurations"])
R, H = {}, {}
for c in proto["cibles"]:
    p = os.path.join(D, f"log_hyp_{c}.txt")
    if not os.path.exists(p):
        continue
    for l in open(p, encoding="utf-8"):
        if l.startswith("[run] "):
            r = json.loads(l[6:])
            R[(r["config"], c, r["graine"])] = r
            hp = os.path.join(D, f"sortie_hyp_{c}", "sorties", c, f"{r['config']}_s{r['graine']}", "historique.json")
            if os.path.exists(hp):
                H[(r["config"], c, r["graine"])] = json.load(open(hp, encoding="utf-8"))["historique"]

cibles = [c for c in proto["cibles"] if all((k, c, g) in R for k in CONFIGS for g in proto["graines"])]
cles = [(c, g) for c in cibles for g in proto["graines"]]
print(f"cibles completes : {cibles} ; {len(R)} runs ; rc != 0 : {sum(r['rc'] != 0 for r in R.values())}\n")


def acc(k, c, g, m="dernier"):
    return R[(k, c, g)][m] * 100


def cout(k, c, g):
    ev = [e for e in H.get((k, c, g), []) if "cout_representant_classe_cible" in e]
    return ev[-1]["cout_representant_classe_cible"] if ev else float("nan")


print("Dernier point (%), puis oracle du dernier point entre parentheses")
for c in cibles:
    print(f"  {c}")
    for k in CONFIGS:
        x = [acc(k, c, g) for g in proto["graines"]]
        o = [acc(k, c, g, "oracle_dernier") for g in proto["graines"]]
        print(f"    {k:20s} " + "  ".join(f"{a:6.2f} ({b:6.2f})" for a, b in zip(x, o)) + f"   moy {np.mean(x):6.2f}")


def bilan(nom, d):
    d = np.array(d)
    n = len(d)
    m = d.mean()
    if n > 1:
        h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    else:
        h = float("nan")
    pos = int((d > 0).sum())
    if n == 9:
        if m > 0 and m - h > 0 and pos >= 7:
            v = "SOUTENUE"
        elif m < 0 and m + h < 0:
            v = "REFUTEE"
        else:
            v = "NON_CONCLUANTE"
    else:
        v = f"(partiel, {n}/9 paires : pas de verdict)"
    print(f"  {nom:48s} {m:+6.2f}  IC95 [{m - h:+6.2f}, {m + h:+6.2f}]  {pos}/{n} positifs  -> {v}")
    return {"moyenne": m, "ic95": [m - h, m + h], "positifs": pos, "n": n, "verdict": v, "ecarts": d.round(4).tolist()}


print("\nComparaisons (ecart en points, paires cible x graine)")
out = {}
out["H1_exactitude"] = bilan("H1 : methode - h1_concatenation", [acc("methode", c, g) - acc("h1_concatenation", c, g) for c, g in cles])
out["H2_exactitude"] = bilan("H2 : methode - h2_plat", [acc("methode", c, g) - acc("h2_plat", c, g) for c, g in cles])
out["complementaire"] = bilan("methode - plat_concatenation", [acc("methode", c, g) - acc("plat_concatenation", c, g) for c, g in cles])

print("\nDiagnostic H1 : cout de transport representant -> sa classe cible (test, vraies etiquettes), dernier point")
dc = []
for c, g in cles:
    a, b = cout("methode", c, g), cout("h1_concatenation", c, g)
    dc.append(b - a)
    print(f"  {c:8s} {g}  barycentre {a:.4f}  concatenation {b:.4f}  ecart {b - a:+.4f}")
dc = np.array(dc)
print(f"  barycentre plus proche dans {int((dc > 0).sum())}/{len(dc)} paires ; ecart moyen {dc.mean():+.4f}")
out["H1_diagnostic_cout"] = {"concatenation_moins_barycentre": dc.round(5).tolist(), "barycentre_plus_proche": int((dc > 0).sum()), "n": len(dc)}

print("\nAppariement : dernier point == oracle ?")
for k in CONFIGS:
    eg = sum(abs(acc(k, c, g) - acc(k, c, g, "oracle_dernier")) < 1e-9 for c, g in cles)
    print(f"  {k:20s} {eg}/{len(cles)}")
json.dump(out, open(os.path.join(D, "analyse_hypotheses.json"), "w"), indent=1)
