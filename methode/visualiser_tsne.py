"""Visualisation t-SNE de l'espace fusionné z, à partir du fichier écrit par l'option phase2.sauver_z.

Une seule projection t-SNE (distance cosinus, celle de la phase 2) des points des sources, de la cible et des
barycentres ensemble :
  - chaque source : points d'une couleur différente ;
  - la cible : astérisques ;
  - les barycentres de Wasserstein des classes : tirets violets.

    python visualiser_tsne.py --npz sorties/xxx/z_dernier.npz --sortie sorties/xxx/tsne.pdf [--cible human]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

COULEURS_SOURCES = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
COULEUR_CIBLE = "#555555"
VIOLET = "#7b2cbf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--sortie", required=True, help="fichier .pdf (un .png est écrit à côté)")
    ap.add_argument("--cible", default="cible", help="nom du domaine cible, pour la légende")
    ap.add_argument("--perplexite", type=float, default=30.0)
    ap.add_argument("--graine", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    zs, src, zt, zb = d["z_sources"], d["source"], d["z_cible"], d["z_barycentres"]
    noms = [str(n) for n in d["noms_sources"]]
    Z = np.concatenate([zs, zt, zb]).astype(np.float64)
    E = TSNE(n_components=2, metric="cosine", init="pca", perplexity=args.perplexite,
             random_state=args.graine).fit_transform(Z)
    es, et, eb = E[:len(zs)], E[len(zs):len(zs) + len(zt)], E[len(zs) + len(zt):]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for k, nom in enumerate(noms):
        m = src == k
        ax.scatter(es[m, 0], es[m, 1], s=6, c=COULEURS_SOURCES[k % len(COULEURS_SOURCES)], alpha=0.55,
                   linewidths=0, label=f"source : {nom}")
    ax.scatter(et[:, 0], et[:, 1], s=22, marker="*", c=COULEUR_CIBLE, alpha=0.6, linewidths=0,
               label=f"cible : {args.cible}")
    if len(eb):
        ax.scatter(eb[:, 0], eb[:, 1], s=90, marker="_", c=VIOLET, linewidths=2.2, label="barycentres")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cccccc")
    ax.legend(loc="best", fontsize=8, frameon=False, markerscale=1.6)
    fig.tight_layout()
    fig.savefig(args.sortie)
    fig.savefig(args.sortie.rsplit(".", 1)[0] + ".png", dpi=200)
    print(f"[tsne] {args.sortie} : {len(zs)} points sources, {len(zt)} cible, {len(zb)} points de barycentres")


if __name__ == "__main__":
    main()
