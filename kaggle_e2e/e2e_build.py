"""Genere (et pousse) le kernel GPU de la methode (fusion_geometrique_m) avec encodeurs ENTRAINES de bout en bout.

Meme algorithme que le cadre A (phase1.py, pertes.py, phase2.py de fusion_geometrique_m, inchanges) ; seules
changent l'entree et les dorsales :
  - donnees brutes : chargeurs de M2CAN (common_loader via phase1_diff.build_full_loader), images Drive et decoupes
    GitHub des auteurs, comme les kernels d'extraction (vue faible + une vue forte par exemple, a chaque pas) ;
  - dorsales : BERT (sortie pooler, prepare_text) et ResNet-50 (moyenne sur les images, prepare_image) tels que
    MsmmModel les initialise, ENTRAINES au taux lr_dorsale (2e-6, celui de M2CAN).
Les classes JeuCaracteristiques, EchantillonneurDomaines et lire_meta de phase1.py sont remplacees a l'execution.

    python kaggle/e2e/build.py --groupe aesthetic --cibles AVA --graines 123456 --account compte3 \
        --full_code <code_ref/full_code> --diff_code <code_ref/diff_code> --paquet <fusion_geometrique_m> \
        [--epoques 3] [--pas_max 60 --suffixe=-smoke] [--push]
"""
import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "brasc_drive_build", os.path.join(REPO, "kaggle", "brasc_drive", "build.py"))
_bd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bd)
USERS, bundle, split_sha, token = _bd.USERS, _bd.bundle, _bd.split_sha, _bd.token

FGM = ["phase1.py", "phase2.py", "pertes.py", "modele.py", "fusion.py", "encodeurs.py", "ot_outils.py",
       "donnees.py", "config_exemple.yaml"]

E2E = r'''

FGM_B64 = "__FGM__"
CONF = __CONF__


def lancer():
    import numpy as np
    import torch
    import torch.nn as nn
    import yaml
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
    from transformers import BertTokenizer
    os.chdir(WORK)
    sys.path.insert(0, WORK)
    import config
    if LAUNCHER:                                   # esthetique sans /workspaces : chemins reecrits en memoire
        ws = f"{DATA}/ws"
        for sp in ("train", "val"):
            for d in DOMAINS["aesthetic"]:
                e = config.DEFAULTS["dataset"][sp][d]
                e["data_list"] = e["data_list"].replace("/workspaces/M2CAN1", ws)
                e["data_root"] = e["data_root"].replace("/workspaces/M2CAN1", ws)
    import phase1_diff as P1
    from common_modules.utils import prepare_text, prepare_image
    from msmm.models.model_msmm import MsmmModel

    fgm_dir = "/kaggle/working/fgm"
    raw = base64.b64decode(FGM_B64)
    log(f"fusion_geometrique_m embarque : sha256 {hashlib.sha256(raw).hexdigest()}")
    os.makedirs(fgm_dir, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        tf.extractall(fgm_dir, filter="data")
    sys.path.insert(0, fgm_dir)
    import encodeurs
    import phase1 as F1

    MODS = {}
    GEL = CONF.get("dorsales", "entrainees")       # entrainees (section 26) | gelees_eval | gelees_train

    def _geler(net):
        if GEL != "entrainees":
            for p in net.parameters():
                p.requires_grad_(False)
        return net

    class DorsaleTexte(nn.Module):
        """BERT de M2CAN, sortie pooler (prepare_text) ; entraine, ou gele selon CONF["dorsales"] :
        gelees_eval = mode evaluation (sans dropout), comme l'extraction des caracteristiques gelees ;
        gelees_train = poids figes mais mode entrainement (dropout actif), comme un taux nul."""
        def __init__(self, **kw):
            super().__init__()
            self.net, self.dim = _geler(MODS["bert"]), 768

        def train(self, mode=True):
            super().train(mode)
            if GEL == "gelees_eval":
                self.net.eval()
            return self

        def forward(self, x):
            if GEL != "entrainees":
                with torch.no_grad():
                    return prepare_text(self.net, x)
            return prepare_text(self.net, x)

    class DorsaleImage(nn.Module):
        """ResNet-50 de M2CAN, moyenne sur les images du message (prepare_image) ; entraine, ou gele :
        gelees_eval = BatchNorm sur les statistiques ImageNet (mode evaluation), comme l'extraction gelee ;
        gelees_train = poids figes mais BatchNorm en mode entrainement (statistiques du lot, moyennes glissantes
        mises a jour), comme un taux nul."""
        def __init__(self, **kw):
            super().__init__()
            self.net, self.dim = _geler(MODS["resnet"]), 2048

        def train(self, mode=True):
            super().train(mode)
            if GEL == "gelees_eval":
                self.net.eval()
            elif mode and CONF.get("bn_eval"):      # affinage avec BatchNorm figee (statistiques ImageNet) :
                for m in self.net.modules():        # le lot d'une image ne permet pas d'estimer les statistiques
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.eval()
            return self

        def forward(self, x):
            if GEL != "entrainees":
                with torch.no_grad():
                    return prepare_image(self.net, x[0], x[1])
            return prepare_image(self.net, x[0], x[1])

    encodeurs.DORSALES["m2can_texte"] = DorsaleTexte
    encodeurs.DORSALES["m2can_image"] = DorsaleImage
    TXT = ("input_ids", "token_type_ids", "attention_mask")

    def formater(b, dev, aug):
        out = {"x": {"text": {k: b[k].to(dev) for k in TXT}, "image": (b["image"].to(dev), b["image_mask"].to(dev))},
               "y": b["label"].to(dev), "ids": [str(i) for i in b["id"]]}
        if aug:
            out["x_aug"] = {"text": {k: b[k + "_aug"].to(dev) for k in TXT},
                            "image": (b["image_aug"].to(dev), b["image_mask"].to(dev))}
        return out

    class JeuBrut:
        """Parcours complet dans un ordre fixe tire au hasard : les limites (sources_max, cible_max) de phase2
        prennent alors un echantillon de toutes les classes, quel que soit l'ordre des listes de M2CAN."""
        def __init__(self, dset, appareil):
            self.dset, self.appareil, self.n = dset, appareil, len(dset)
            self.ordre = Subset(dset, np.random.RandomState(0).permutation(self.n).tolist())

        def tout(self, taille=32):
            ld = DataLoader(self.ordre, batch_size=taille, num_workers=4, sampler=SequentialSampler(self.ordre))
            for b in ld:
                o = formater(b, self.appareil, False)
                yield {"x": o["x"], "y": b["label"].numpy(), "ids": o["ids"]}

    class EchantillonneurBrut:
        """Un lot de meme taille par domaine (sources puis cible), en cyclant ; chargeurs a 2 processus."""
        def __init__(self, jeux, taille_lot, graine):
            self.jeux, self.lds = jeux, []
            for k, j in enumerate(jeux):
                g = torch.Generator()
                g.manual_seed(graine + k)
                self.lds.append(DataLoader(j.dset, batch_size=taille_lot, num_workers=2, drop_last=True,
                                           sampler=RandomSampler(j.dset, generator=g), persistent_workers=True))
            self.its = [iter(l) for l in self.lds]

        def suivant(self):
            lots = []
            for k, j in enumerate(self.jeux):
                try:
                    b = next(self.its[k])
                except StopIteration:
                    self.its[k] = iter(self.lds[k])
                    b = next(self.its[k])
                lots.append(formater(b, j.appareil, True))
            return lots

    groupe = P["group"]
    doms = DOMAINS[groupe]
    resultats = []
    for cible in CONF["cibles"]:
        cfg = P1.load_config(f"msmm/scripts/config_{cible}.yaml")
        cfg["save_path"] = "/kaggle/working/e2e_tmp"
        cfg["snapshot_dir"] = "e2e_tmp"
        os.makedirs(cfg["save_path"], exist_ok=True)
        tok = BertTokenizer.from_pretrained(cfg["dataset"]["bert_model"])
        K = int(cfg["net"]["num_classes"])
        dsets = {}
        for d in doms:
            for sp, so in (("train", "train"), ("val", "test")):
                dsets[(d, so)] = P1.build_full_loader(cfg, d, sp, tok, 32, 0).dataset
                log(f"  {d}/{so} : {len(dsets[(d, so)])} exemples")
        F1.JeuCaracteristiques = lambda r, d, split, noms, app: JeuBrut(dsets[(d, split)], app)
        F1.lire_meta = lambda r: {"classes": K, "modalites": ["text", "image"], "domaines": list(doms)}
        F1.EchantillonneurDomaines = EchantillonneurBrut
        n_tr = len(dsets[(cible, "train")])
        ppe = CONF.get("pas_par_epoque") or max(1, n_tr // 4)
        per = CONF.get("periode") or max(1, ppe // 5)
        ev_per = CONF.get("evaluer") or ppe
        for graine in CONF["graines"]:
            torch.manual_seed(graine)
            np.random.seed(graine)
            base = MsmmModel(cfg)                           # BERT et ResNet-50 pre-entraines, comme M2CAN
            MODS["bert"], MODS["resnet"] = base.bert_model, base.model_image
            del base
            c = yaml.safe_load(open(os.path.join(fgm_dir, "config_exemple.yaml"), encoding="utf-8"))
            for k, v in CONF["surcharges"].items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        if isinstance(v2, dict):
                            c.setdefault(k, {}).setdefault(k2, {}).update(v2)
                        else:
                            c.setdefault(k, {})[k2] = v2
                else:
                    c[k] = v
            c["modalites"] = [{"nom": "text", "encodeur": {"type": "m2can_texte"}},
                              {"nom": "image", "encodeur": {"type": "m2can_image"}}]
            c["donnees"] = {"racine": "brut", "sources": [d for d in doms if d != cible], "cible": cible}
            sortie = f"/kaggle/working/sorties/{cible}_s{graine}"
            c.update(graine=graine, appareil="cuda", sortie=sortie)
            c["phase1"].update(epoques=CONF["epoques"], pas_par_epoque=ppe, evaluer_tous_les=ev_per)
            c.setdefault("pseudo", {})["tous_les"] = per
            if CONF.get("calendrier_m2can"):        # epoque 1 sources seules, un tour par epoque, amorcage par les tetes
                c["phase1"]["cmcfa_cible_debut"] = ppe
                c["pseudo"].update(debut=ppe, tous_les=ppe, taux=[0.3, 0.6, 0.9, 1.0], amorcage_tours=2)
                per = ppe
            os.makedirs(sortie, exist_ok=True)
            cp = f"{sortie}/config.yaml"
            yaml.safe_dump(c, open(cp, "w"), allow_unicode=True)
            log(f"=== {cible} graine {graine} : {CONF['epoques']} epoques x {ppe} pas, pseudo-etiquettes tous les {per}, evaluation tous les {ev_per}, "
                f"lr dorsale {c['phase1']['optimiseur']['lr_dorsale']}, extracteur {c['phase1']['optimiseur']['lr_extracteur']}")
            t0 = time.time()
            sys.argv = ["phase1.py", "--config", cp] + (["--pas_max", str(CONF["pas_max"])] if CONF.get("pas_max") else [])
            rc = 0
            try:
                os.chdir(fgm_dir)
                F1.main()
            except Exception as e:
                import traceback
                traceback.print_exc()
                rc = repr(e)[:300]
            finally:
                os.chdir(WORK)
            res = {"cible": cible, "graine": graine, "rc": rc, "minutes": round((time.time() - t0) / 60, 1)}
            try:
                h = json.load(open(f"{sortie}/historique.json"))
                ev = [e for e in h["historique"] if "exactitude" in e]
                ps = [e["pseudo"] for e in h["historique"] if "pseudo" in e]
                res.update(dernier=ev[-1]["exactitude"], meilleur=max(e["exactitude"] for e in ev),
                           oracle_dernier=ev[-1]["oracle"], tailles_dernier=ev[-1]["tailles_groupes"], n_evals=len(ev),
                           pseudo_premier=ps[0] if ps else None, pseudo_dernier=ps[-1] if ps else None)
            except Exception as e:
                res["erreur_resume"] = repr(e)[:200]
            for fn in os.listdir(sortie):
                if fn.endswith(".pt"):
                    os.remove(os.path.join(sortie, fn))
            log("[run] " + json.dumps(res))
            resultats.append(res)
            MODS.clear()
            torch.cuda.empty_cache()
            json.dump(resultats, open("/kaggle/working/resultats.json", "w"), indent=1)


def main():
    global LAUNCHER
    log(f"=== methode de bout en bout : {json.dumps(P)} ; {json.dumps(CONF)}")
    setup_torch()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ekphrasis", "emoji"], check=True)
    unpack_code()
    log("=== donnees ===")
    LAUNCHER = prepare_data()
    check_paths(LAUNCHER)
    lancer()
    shutil.rmtree(DATA, ignore_errors=True)
    shutil.rmtree(WORK, ignore_errors=True)
    log("=== FIN ===")


LAUNCHER = None
if __name__ == "__main__":
    main()
'''


def archive(chemins):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel, src in sorted(chemins.items()):
            with open(src, "rb") as fh:
                data = fh.read()
            ti = tarfile.TarInfo(rel)
            ti.size, ti.mtime, ti.mode = len(data), 0, 0o644
            tf.addfile(ti, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groupe", choices=["sentiment", "aesthetic"], required=True)
    ap.add_argument("--cibles", nargs="+", required=True)
    ap.add_argument("--graines", nargs="+", type=int, default=[123456, 234567, 345678])
    ap.add_argument("--epoques", type=int, default=3)
    ap.add_argument("--pas_par_epoque", type=int, default=None, help="defaut : taille du train cible / 4")
    ap.add_argument("--periode", type=int, default=None,
                    help="evaluation et pseudo-etiquettes ; defaut : pas_par_epoque / 5 (750 pour 3750, comme le cadre A)")
    ap.add_argument("--evaluer", type=int, default=None,
                    help="periode d'evaluation sur le test cible ; defaut : une fois par epoque (mesure = dernier point)")
    ap.add_argument("--dim_z", type=int, default=256, help="dimension de l'espace fusionne (ALGORITHME.md : 256)")
    ap.add_argument("--lr_extracteur", type=float, default=5e-05, help="ALGORITHME.md : 5e-5 ; section 19 / M2CAN : 2e-5")
    ap.add_argument("--lr_dorsale", type=float, default=2e-06, help="section 26 : 2e-6 ; section 19 / M2CAN : 2e-5")
    ap.add_argument("--lr_fusion", type=float, default=5e-05, help="section 26 : 5e-5 ; section 19 / M2CAN (mlb_lr) : 5e-4")
    ap.add_argument("--lr_tetes", type=float, default=5e-05, help="section 26 : 5e-5 ; section 19 / M2CAN (classifier_lr) : 5e-4")
    ap.add_argument("--alpha_cmcfa", type=float, default=1.0, help="poids de CMCFA ; section 19 / config_AVA : 0.5")
    ap.add_argument("--bn_eval", action="store_true",
                    help="BatchNorm de ResNet-50 figee (mode evaluation) pendant l'affinage ; les poids s'entrainent")
    ap.add_argument("--calendrier_m2can", action="store_true",
                    help="epoque 1 sources seules (cmcfa_cible_debut), un tour de pseudo-etiquettes par epoque "
                         "(debut = tous_les = pas_par_epoque), quotas 0.3/0.6/0.9/1 par epoque, 2 tours amorces par les tetes")
    ap.add_argument("--poids_im", type=float, default=0.0, help="maximisation de l'information (SHOT) sur la cible ; 0 = methode")
    ap.add_argument("--pas_max", type=int, default=None)
    ap.add_argument("--dorsales", choices=["entrainees", "gelees_eval", "gelees_train"], default="entrainees",
                    help="entrainees (section 26) ; gelees_eval : poids figes, mode evaluation (= caracteristiques "
                         "gelees) ; gelees_train : poids figes, mode entrainement (BatchNorm et dropout actifs)")
    ap.add_argument("--full_code", required=True)
    ap.add_argument("--diff_code", required=True)
    ap.add_argument("--paquet", required=True)
    ap.add_argument("--account", choices=sorted(USERS), default="compte3")
    ap.add_argument("--suffixe", default="")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    with open(os.path.join(REPO, "kaggle", "brasc_drive", "template.py"), encoding="utf-8") as f:
        tpl = f.read()
    tete = tpl[:tpl.index("def run_seed(")]
    b64, manifest = bundle(args.full_code, args.diff_code)
    fgm = {f: os.path.join(args.paquet, f) for f in FGM}
    h = hashlib.sha256()
    for rel, p in sorted(fgm.items()):
        with open(p, "rb") as fh:
            h.update(rel.encode() + b"\0" + fh.read())
    conf = {"cibles": args.cibles, "graines": args.graines, "epoques": args.epoques,
            "pas_par_epoque": args.pas_par_epoque, "periode": args.periode, "evaluer": args.evaluer, "pas_max": args.pas_max,
            "surcharges": {"fusion": {"dim": args.dim_z}, "pseudo": {"actif": True},
                           "phase1": {"poids_im": args.poids_im, "alpha_cmcfa": args.alpha_cmcfa,
                                      "optimiseur": {"lr_extracteur": args.lr_extracteur,
                                                     "lr_fusion": args.lr_fusion, "lr_tetes": args.lr_tetes,
                                                     "lr_dorsale": args.lr_dorsale if args.dorsales == "entrainees" else 0.0}},
                           "phase2": {"sources_max": 1600}},
            "dorsales": args.dorsales, "bn_eval": args.bn_eval, "calendrier_m2can": args.calendrier_m2can,
            "sha256_methode": h.hexdigest()}
    params = {"group": args.groupe, "target": "e2e", "source": "drive"}
    code = (tete.replace("__PARAMS__", repr(params)).replace("__MANIFEST__", json.dumps(manifest, indent=1))
            .replace("__SPLIT_SHA__", json.dumps(split_sha(), indent=1)).replace("__BUNDLE__", b64)
            + E2E.replace("__FGM__", archive(fgm)).replace("__CONF__", repr(conf)))
    compile(code, "e2e.py", "exec")
    slug = f"e2e-{args.groupe}-{'-'.join(c.lower() for c in args.cibles)}{args.suffixe}"
    build = tempfile.mkdtemp(prefix="e2e_")
    kdir = os.path.join(build, slug)
    os.makedirs(kdir)
    with open(os.path.join(kdir, "e2e.py"), "w", encoding="utf-8") as f:
        f.write(code)
    meta = {"id": f"{USERS[args.account]}/{slug}", "title": slug, "code_file": "e2e.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    with open(os.path.join(kdir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[genere] {meta['id']} ({len(code) / 1024:.0f} Ko, methode sha {h.hexdigest()[:12]})")
    if args.push:
        env = os.environ.copy()
        tok = token(args.account)
        if tok:
            env["KAGGLE_API_TOKEN"] = tok
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", slug], cwd=build,
                           capture_output=True, text=True, env=env)
        print("   ", ((r.stdout or r.stderr).strip().splitlines() or ["?"])[-1])


if __name__ == "__main__":
    main()
