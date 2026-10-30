"""Phase 2 : étiquetage transductif de la cible par WBHR et transport optimal hiérarchique (HOT-DA).

Reprise de evaluate_hotda (hotda_inloop.py) ; ne dépend que de l'espace fusionné z, donc d'aucune modalité.

  1. barycentre de Wasserstein par classe sur les sources ;
  2. partition spectrale de la cible (affinité exp(-coût cosinus / médiane)) ;
  3. coût de Sinkhorn classe x groupe ;
  4. transport entre structures (Sinkhorn, marges uniformes) puis appariement hongrois sur le plan ;
  5. chaque point cible prend la classe appariée à son groupe.

Option `partition: equilibree` (au lieu de `spectrale`, défaut) : les étapes 1 à 4 sont remplacées par un
transport équilibré des points cibles vers les centres de classe des sources (coût cosinus divisé par sa
médiane, Sinkhorn, marges uniformes, eps 0.05), chaque point prenant la classe où va le plus de sa masse.
Les groupes sont alors les classes prédites (ABLATIONS section 15 : 69.07 contre 66.20 sur Twitter).

Options de fidélité à WBHR / HOT-DA (défauts = la méthode) :
  support: 16 | auto                          16 (défaut, algorithme 2 de WBHR) ; auto : cardinalité du cadre
                théorique, |Supp(rho_h^aux)| = (1/N) sum_i |C_h^i|, la taille moyenne de la classe h dans les
                sources (après le tirage de `par_classe` points au plus par source).
  appariement: hongrois | argmax | hongrois_accord
                hongrois (défaut) : bijection groupes <-> classes sur le plan G ;
                argmax : comme WBHR, chaque classe h va au groupe argmax_l G[h, l] (pas forcément bijectif) et
                chaque groupe prend la classe argmax_h G[h, l] ;
                hongrois_accord : bijection sur G/max(G) + poids_accord (1) x part des votes des têtes unimodales
                pour la classe h dans le groupe l. Quand le plan G est presque plat (cible loin des sources), le
                hongrois seul peut rendre une permutation fausse alors que la partition est bonne (TumEmo : 35 %
                d'exactitude pour un oracle de 58 %) ; les têtes, ancrées par les étiquettes sources, départagent.
  classifieur: aucun | logistique | voisins    aucun (défaut) : chaque point cible prend la classe appariée à son
                groupe. Sinon, fin de HOT-DA : les points du représentant B_h sont transportés par projection
                barycentrique vers leur groupe apparié (B~_h = |B_h| gamma_{h,l} Z_l), un classifieur est entraîné
                sur les points transportés étiquetés h (régression logistique, ou k plus proches voisins avec
                `voisins`, 1 par défaut), sur les z normalisés L2, puis appliqué à tous les points cibles.

Options de test des hypothèses (défauts = la méthode) :
  representant: barycentre | concatenation   (H1) chaque classe source est résumée par le barycentre de Wasserstein
                des nuages de cette classe dans chaque source (défaut), ou par ces nuages mis bout à bout.
  transport: hierarchique | plat              (H2) étapes 2 à 4 (partition, transport entre structures, appariement,
                défaut), ou transport direct des points cibles vers les points des représentants de classe (Sinkhorn,
                marges uniformes, coût cosinus, même eps) ; chaque point prend la classe qui reçoit le plus de sa masse.

Options de partition (défauts = la méthode) :
  affinite: dense | knn             graphe complet exp(-C/sigma) (défaut) ou graphe des `voisins_graphe` (15) plus
                proches voisins, symétrisé ; voir `affinite`.

Options d'évaluation (défauts = la méthode) :
  plancher: true                    (défaut) ajoute `plancher_logistique` au journal : régression logistique sur les z
                des sources appliquée au test cible, ce que vaut z sans WBHR.
  evaluation_jointe: true           le test cible est partitionné avec les points du train cible (sans étiquette).
  sauver_z: true                    écrit z_dernier.npz (sources, cible, barycentres) pour visualiser_tsne.py.

Utilisation autonome, sur un modèle déjà entraîné :
    python phase2.py --config config.yaml --modele sorties/xxx/modele_meilleur.pt
"""
import argparse

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from ot_outils import cosine_cost, cout_transport, free_support_barycenter, plan_transport, projection_barycentrique, sinkhorn_log


@torch.no_grad()
def representations(modele, jeu, limite=None):
    """z (fusionné), prédictions moyennes des têtes, ids et étiquettes d'un jeu (modèle en évaluation)."""
    zs, ps, ids, ys, n = [], [], [], [], 0
    for lot in jeu.tout():
        f = modele.encoder(lot["x"])
        zs.append(modele.fusionner(f).float())
        ps.append(torch.stack([torch.softmax(l, 1) for l in modele.logits_tetes(f)]).mean(0).cpu())
        ids += lot["ids"]; ys.append(lot["y"]); n += len(lot["ids"])
        if limite and n >= limite:
            break
    z, p = torch.cat(zs), torch.cat(ps).numpy()
    y = np.concatenate(ys)
    if limite:
        z, p, ids, y = z[:limite], p[:limite], ids[:limite], y[:limite]
    return z, p, ids, y


@torch.no_grad()
def etiqueter(z_src, y_src, zt, K, conf, graine=0, pt=None):
    """Cœur de la phase 2. z_src : liste de N tenseurs (sources) ; y_src : liste de N tableaux ; zt : cible.
    pt (n, K) : prédiction moyenne des têtes unimodales sur la cible, pour `appariement: hongrois_accord`.

    Renvoie un dictionnaire : étiquettes, groupes, appariement, marge géométrique, coûts W.
    """
    if conf.get("partition", "spectrale") == "equilibree":
        return etiqueter_equilibre(z_src, y_src, zt, K)
    rng = np.random.default_rng(graine)
    barys = {}
    for h in range(K):
        clouds = []
        for z, y in zip(z_src, y_src):
            idx = np.where(y == h)[0]
            if idx.size:
                sel = rng.choice(idx, min(conf.get("par_classe", 400), idx.size), replace=False)
                clouds.append(z[torch.as_tensor(sel, device=z.device)])
        if not clouds:
            continue
        if conf.get("representant", "barycentre") == "concatenation":
            barys[h] = torch.cat(clouds)
        else:
            S = conf.get("support", 16)
            if S == "auto":            # cadre théorique de WBHR : |Supp(rho_h^aux)| = (1/N) sum_i |C_h^i|
                S = max(1, int(round(sum(c.shape[0] for c in clouds) / len(clouds))))
            barys[h] = free_support_barycenter(clouds, support=S, eps=conf.get("eps", 0.02),
                                               sinkhorn_iters=conf.get("sinkhorn", 60),
                                               outer_iters=conf.get("point_fixe", 8))[0]
    C = cosine_cost(zt, zt).cpu().numpy()
    if conf.get("transport", "hierarchique") == "plat":
        return etiqueter_plat(barys, zt, K, conf, C)
    A = affinite(C, conf)
    groupes = SpectralClustering(n_clusters=K, affinity="precomputed", assign_labels="kmeans",
                                 random_state=graine).fit_predict(A)
    eps, it = conf.get("eps", 0.02), conf.get("sinkhorn", 60)
    W = np.full((K, K), 4.0)
    for h in barys:
        for l in range(K):
            sel = np.where(groupes == l)[0]
            if sel.size:
                W[h, l] = cout_transport(barys[h], zt[torch.as_tensor(sel, device=zt.device)], eps, it)
    u = torch.full((K,), 1.0 / K)
    G = sinkhorn_log(torch.tensor(W, dtype=torch.float32), u, u, eps, 300).numpy()
    app = conf.get("appariement", "hongrois")
    if app == "argmax":
        c2g = {h: int(G[h].argmax()) for h in barys}                  # classe -> groupe, comme WBHR
        g2c = {l: int(G[:, l].argmax()) for l in range(K)}            # groupe -> classe, pour les étiquettes
    else:
        S = G / max(float(G.max()), 1e-12)
        if app == "hongrois_accord" and pt is not None:
            # part des votes des têtes pour la classe h dans le groupe l : départage un plan G presque plat
            V = np.stack([np.asarray(pt)[groupes == l].mean(0) if (groupes == l).any() else np.full(K, 1.0 / K)
                          for l in range(K)], 1)                       # (K classes, K groupes)
            S = S + float(conf.get("poids_accord", 1.0)) * V
        rows, cols = linear_sum_assignment(-S)
        g2c = {int(c): int(r) for r, c in zip(rows, cols)}
        c2g = {h: l for l, h in g2c.items()}
    etiq = np.array([g2c.get(int(g), 0) for g in groupes])

    res = {"etiquettes": etiq, "groupes": groupes, "appariement": g2c, "marge": marge_geometrique(zt, groupes, K),
           "W": W, "C": C, "representants": barys}
    if conf.get("classifieur", "aucun") != "aucun":
        res["etiquettes"] = classifier_transporte(barys, c2g, zt, groupes, conf, eps, it, graine)
    return res


def affinite(C, conf):
    """Matrice de ressemblance de la partition spectrale à partir du coût cosinus C.

    dense (défaut) : A = exp(-C / sigma), sigma = médiane des coûts non nuls, graphe complet.
    knn : même poids, mais seuls les `voisins_graphe` (15) plus proches voisins de chaque point sont reliés
          (graphe symétrisé par le maximum) ; n'impose aucune taille de groupe, évite le groupe géant d'une cible diffuse.
    """
    sigma = float(np.median(C[C > 0])) if np.any(C > 0) else 1.0
    A = np.exp(-C / max(sigma, 1e-8))
    np.fill_diagonal(A, 1.0)
    if conf.get("affinite", "dense") == "knn":
        k = min(int(conf.get("voisins_graphe", 15)), C.shape[0] - 1)
        voisins = np.argpartition(C, k, axis=1)[:, :k + 1]            # k plus proches, soi-même compris
        M = np.zeros_like(A, dtype=bool)
        np.put_along_axis(M, voisins, True, axis=1)
        M |= M.T
        A = np.where(M, A, 0.0)
    return A


def plancher_logistique(z_src, y_src, zt, yt, graine=0):
    """Exactitude sur la cible d'une régression logistique entraînée sur les z des sources (normalisés L2) :
    ce que vaut l'espace fusionné sans l'étiquetage géométrique. Diagnostic seulement."""
    from sklearn.linear_model import LogisticRegression
    n = lambda z: torch.nn.functional.normalize(z, dim=-1).cpu().numpy()
    X, y = np.concatenate([n(z) for z in z_src]), np.concatenate([np.asarray(v) for v in y_src])
    clf = LogisticRegression(max_iter=1000, random_state=graine).fit(X, y)
    return float((clf.predict(n(zt)) == np.asarray(yt)).mean())


@torch.no_grad()
def classifier_transporte(barys, c2g, zt, groupes, conf, eps, it, graine=0):
    """Fin de HOT-DA (option classifieur) : projection barycentrique des représentants vers leur groupe apparié,
    puis classifieur entraîné sur les points transportés et appliqué à toute la cible."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    X, y = [], []
    for h, l in c2g.items():
        if h not in barys:
            continue
        sel = np.where(groupes == l)[0]
        if not sel.size:
            continue
        Zl = zt[torch.as_tensor(sel, device=zt.device)]
        P, _ = plan_transport(barys[h], Zl, eps, it)
        X.append(projection_barycentrique(barys[h], Zl, P))
        y.append(np.full(barys[h].shape[0], h))
    X = torch.nn.functional.normalize(torch.cat(X), dim=-1).cpu().numpy()
    y = np.concatenate(y)
    if conf.get("classifieur") == "voisins":
        clf = KNeighborsClassifier(n_neighbors=min(conf.get("voisins", 1), len(y)), metric="cosine")
    else:
        clf = LogisticRegression(max_iter=1000, random_state=graine)
    if len(np.unique(y)) < 2:          # une seule classe transportée : rien à apprendre
        return np.full(len(groupes), int(y[0]))
    clf.fit(X, y)
    return clf.predict(torch.nn.functional.normalize(zt, dim=-1).cpu().numpy()).astype(int)


@torch.no_grad()
def etiqueter_plat(barys, zt, K, conf, C):
    """Transport plat (option transport: plat) : points cibles -> points des représentants de classe, en une fois."""
    classes = sorted(barys)
    S = torch.cat([barys[h] for h in classes])
    ys = np.concatenate([np.full(barys[h].shape[0], h) for h in classes])
    Cts = cosine_cost(zt, S).double()
    a = torch.full((zt.shape[0],), 1.0 / zt.shape[0], dtype=torch.float64, device=Cts.device)
    b = torch.full((S.shape[0],), 1.0 / S.shape[0], dtype=torch.float64, device=Cts.device)
    P = sinkhorn_log(Cts, a, b, conf.get("eps", 0.02), 300).cpu().numpy()
    masse = np.stack([P[:, ys == h].sum(1) if h in barys else np.zeros(P.shape[0]) for h in range(K)], 1)
    etiq = masse.argmax(1)
    return {"etiquettes": etiq, "groupes": etiq, "appariement": {h: h for h in range(K)},
            "marge": marge_geometrique(zt, etiq, K), "W": None, "C": C, "representants": barys}


def marge_geometrique(zt, groupes, K):
    centres = torch.stack([zt[torch.as_tensor(np.where(groupes == l)[0], device=zt.device)].mean(0)
                           if (groupes == l).any() else zt.mean(0) for l in range(K)])
    D = cosine_cost(zt, centres).cpu().numpy()
    propre = D[np.arange(len(groupes)), groupes]
    D[np.arange(len(groupes)), groupes] = np.inf
    return D.min(1) / (propre + 1e-8)


@torch.no_grad()
def etiqueter_equilibre(z_src, y_src, zt, K):
    """Transport équilibré de la cible vers les centres de classe des sources (option partition: equilibree)."""
    zs, ys = torch.cat(z_src), np.concatenate(y_src)
    centres = torch.stack([zs[torch.as_tensor(np.where(ys == h)[0], device=zs.device)].mean(0) for h in range(K)])
    Cg = cosine_cost(zt, centres).double()
    Cg = Cg / Cg.median().clamp_min(1e-12)
    a = torch.full((zt.shape[0],), 1.0 / zt.shape[0], dtype=torch.float64, device=Cg.device)
    b = torch.full((K,), 1.0 / K, dtype=torch.float64, device=Cg.device)
    etiq = sinkhorn_log(Cg, a, b, 0.05, 500).argmax(1).cpu().numpy()
    C = cosine_cost(zt, zt).cpu().numpy()
    return {"etiquettes": etiq, "groupes": etiq, "appariement": {h: h for h in range(K)},
            "marge": marge_geometrique(zt, etiq, K), "W": None, "C": C}


def diagnostics(res, y_vrai, K):
    """Exactitude, plafond oracle de la partition, silhouette, ratio intra/inter (évaluation seulement)."""
    g, C = res["groupes"], res["C"]
    M = np.zeros((K, K), dtype=int)
    for l in range(K):
        for h in range(K):
            M[h, l] = int(((g == l) & (y_vrai == h)).sum())
    r, c = linear_sum_assignment(-M)
    try:
        sil = float(silhouette_score(C, g, metric="precomputed"))
    except Exception:
        sil = float("nan")
    intra = float(np.mean([C[np.ix_(g == l, g == l)].mean() for l in range(K) if (g == l).sum() > 1]))
    inter = float(np.mean([C[np.ix_(g == l, g != l)].mean() for l in range(K) if (g == l).any()]))
    return {"exactitude": float((res["etiquettes"] == y_vrai).mean()), "oracle": float(M[r, c].sum() / len(g)),
            "silhouette": sil, "ratio_intra_inter": intra / inter if inter else float("nan"),
            "tailles_groupes": np.bincount(g, minlength=K).tolist()}


def sauver_representations(chemin, src, noms_sources, zt, yt, res, par_source=1000, graine=0):
    """Option sauver_z : z des sources (au plus par_source points chacune), z de la cible de test avec ses vraies
    étiquettes, groupes et étiquettes prédites, et les points des représentants de classe (barycentres)."""
    rng = np.random.default_rng(graine)
    zs, ys, ds = [], [], []
    for k, s in enumerate(src):
        idx = rng.choice(len(s[3]), min(par_source, len(s[3])), replace=False)
        zs.append(s[0][torch.as_tensor(idx, device=s[0].device)].cpu().numpy())
        ys.append(np.asarray(s[3])[idx])
        ds.append(np.full(len(idx), k))
    reps = res.get("representants") or {}
    hs = sorted(reps)
    np.savez_compressed(
        chemin, z_sources=np.concatenate(zs), y_sources=np.concatenate(ys), source=np.concatenate(ds),
        noms_sources=np.array(noms_sources), z_cible=zt.cpu().numpy(), y_cible=np.asarray(yt),
        groupes=np.asarray(res["groupes"]), etiquettes=np.asarray(res["etiquettes"]),
        z_barycentres=np.concatenate([reps[h].cpu().numpy() for h in hs]) if hs else np.zeros((0, zt.shape[1])),
        classe_barycentres=np.concatenate([np.full(reps[h].shape[0], h) for h in hs]) if hs else np.zeros(0))


def evaluer(modele, jeux_sources, jeu_cible, K, conf, graine=0, jeu_cible_train=None, n_train=3000, chemin_z=None):
    """jeu_cible_train (option evaluation_jointe) : le test cible est étiqueté avec au plus n_train points du train
    cible (sans étiquette), en une seule partition ; seuls les points du test sont mesurés.
    chemin_z (option sauver_z) : fichier .npz des représentations, pour la visualisation (visualiser_tsne.py)."""
    was = modele.training
    modele.eval()
    try:
        src = [representations(modele, j, conf.get("sources_max")) for j in jeux_sources]
        zt, pt, _, yt = representations(modele, jeu_cible)
        nt = len(yt)
        if jeu_cible_train is None:
            zall, pall = zt, pt
        else:
            ztr, ptr, _, _ = representations(modele, jeu_cible_train, n_train)
            zall, pall = torch.cat([zt, ztr]), np.concatenate([pt, ptr])
        res = etiqueter([s[0] for s in src], [s[3] for s in src], zall, K, conf, graine, pt=pall)
        if jeu_cible_train is not None:
            res = dict(res, etiquettes=res["etiquettes"][:nt], groupes=res["groupes"][:nt], C=res["C"][:nt, :nt])
        d = diagnostics(res, yt, K)
        if conf.get("plancher", True):
            try:
                d["plancher_logistique"] = plancher_logistique([s[0] for s in src], [s[3] for s in src], zt, yt, graine)
            except Exception as e:   # diagnostic : ne doit jamais interrompre l'entraînement
                d["plancher_logistique"] = float("nan"); print(f"[phase2] plancher logistique : {e!r}", flush=True)
        reps = res.get("representants")
        if reps:   # diagnostic de H1 : distance de transport entre chaque représentant et SA classe cible (étiquettes vraies)
            eps, it = conf.get("eps", 0.02), conf.get("sinkhorn", 60)
            couts = [cout_transport(reps[h], zt[torch.as_tensor(np.where(yt == h)[0], device=zt.device)], eps, it)
                     for h in reps if (yt == h).any()]
            d["cout_representant_classe_cible"] = float(np.mean(couts)) if couts else float("nan")
        if chemin_z:
            sauver_representations(chemin_z, src, [getattr(j, "domaine", f"source {k + 1}") for k, j in enumerate(jeux_sources)],
                                   zt, yt, res, graine=graine)
        return d
    finally:
        if was:
            modele.train()


def main():
    import yaml
    from donnees import JeuCaracteristiques, lire_meta
    from modele import ModeleFusionGeometrique

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--modele", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    app = cfg.get("appareil", "auto")
    if app == "auto":
        app = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[appareil] {app}", flush=True)
    meta = lire_meta(cfg["donnees"]["racine"])
    noms = [m["nom"] for m in cfg["modalites"]]
    modele = ModeleFusionGeometrique(cfg["modalites"], meta["classes"]).to(app)
    modele.load_state_dict(torch.load(args.modele, map_location=app))
    r = cfg["donnees"]["racine"]
    src = [JeuCaracteristiques(r, d, "train", noms, app) for d in cfg["donnees"]["sources"]]
    cib = JeuCaracteristiques(r, cfg["donnees"]["cible"], "test", noms, app)
    print(evaluer(modele, src, cib, meta["classes"], cfg.get("phase2", {})))


if __name__ == "__main__":
    main()
