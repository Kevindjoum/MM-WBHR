"""Tableaux à partir de resultats.json (écrit par lancer_banc.py) : moyenne ± écart-type par méthode et par cible,
puis écart de la méthode à M2CAN, apparié par (cible, graine).

    python analyser.py sorties/sentiment/resultats.json [--mesure dernier|meilleur]

« dernier » (défaut) = exactitude au dernier point d'évaluation ; « meilleur » = meilleur point, choisi avec les
étiquettes du test cible, donc optimiste.
"""
import argparse
import json
import math
import statistics as st


def ic95(d):
    if len(d) < 2:
        return float("nan"), float("nan")
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(len(d), 1.96)
    h = t * st.stdev(d) / math.sqrt(len(d))
    return st.mean(d) - h, st.mean(d) + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("resultats")
    ap.add_argument("--mesure", default="dernier", choices=["dernier", "meilleur"])
    args = ap.parse_args()
    runs = [r for r in json.load(open(args.resultats, encoding="utf-8")) if r.get("rc") == 0 and args.mesure in r]
    echecs = [r for r in json.load(open(args.resultats, encoding="utf-8")) if r.get("rc") != 0]
    groupes = sorted({(r["methode"], r["config"]) for r in runs})
    cibles = sorted({r["cible"] for r in runs})
    val = {(r["methode"], r["config"], r["cible"], r["graine"]): r[args.mesure] * 100 for r in runs}

    print(f"Exactitude ({args.mesure} point), moyenne ± écart-type sur les graines, en %\n")
    print(f"{'méthode / configuration':34s}" + "".join(f"{c:>18s}" for c in cibles) + f"{'moyenne':>12s}")
    for m, c in groupes:
        cells, moy = [], []
        for cb in cibles:
            v = [x for (m2, c2, cb2, _), x in val.items() if (m2, c2, cb2) == (m, c, cb)]
            if v:
                moy.append(st.mean(v))
                cells.append(f"{st.mean(v):6.2f} ± {st.stdev(v) if len(v) > 1 else 0:4.2f} (n={len(v)})")
            else:
                cells.append("-")
        print(f"{m + ' / ' + c:34s}" + "".join(f"{x:>18s}" for x in cells) + f"{st.mean(moy) if moy else float('nan'):12.2f}")

    print("\nÉcart à M2CAN, apparié par (cible, graine)")
    for m, c in groupes:
        if m == "m2can":
            continue
        d = [x - val[("m2can", "publie", cb, g)] for (m2, c2, cb, g), x in val.items()
             if (m2, c2) == (m, c) and ("m2can", "publie", cb, g) in val]
        if not d:
            continue
        lo, hi = ic95(d)
        print(f"  {m} / {c} : {st.mean(d):+6.2f} points, IC95 [{lo:+6.2f}, {hi:+6.2f}], "
              f"{sum(x > 0 for x in d)}/{len(d)} paires en faveur de la méthode")
    if echecs:
        print(f"\n{len(echecs)} run(s) en échec : voir journal.txt de chacun")
        for r in echecs:
            print(f"  {r['methode']} / {r['config']} / {r['cible']} / {r['graine']}")


if __name__ == "__main__":
    main()
