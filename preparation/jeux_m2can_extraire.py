"""Prépare un groupe de jeux de M2CAN (sentiment ou esthétique) pour le cadre A : télécharge, extrait, écrit.

    python preparation/jeux_m2can_extraire.py --groupe sentiment --brut donnees_brutes --sortie donnees/sentiment
    python preparation/jeux_m2can_extraire.py --groupe aesthetic --brut donnees_brutes --sortie donnees/aesthetic

1. Images : les .zip publiés par les auteurs de M2CAN sur Google Drive (lien de leur README), taille vérifiée.
   Découpes train/val : le dépôt GitHub des auteurs (jingjiang02/M2CAN), empreintes sha256 vérifiées.
2. Caractéristiques gelées, avec les encodeurs de M2CAN tels qu'il les initialise (sans entraînement) :
   texte = BERT, sortie pooler (prepare_text), 768-d ; image = ResNet-50 ImageNet, moyenne sur les images du post
   (prepare_image), 2048-d. Vue faible = transformations faibles de M2CAN ; --vues_fortes vues fortes = autant de
   passages avec ses transformations fortes (texte : 10 % des mots supprimés ; image : recadrage, symétrie, couleurs),
   pour le train seulement.
3. Sortie au format commun (methode/donnees.py), val de M2CAN -> test :
   <sortie>/meta.json ; <sortie>/<domaine>/{train,test}/{text,image,labels,ids}.npy ; train/{text,image}_aug.npy.

GPU détecté automatiquement (--appareil auto) ; sur processeur, l'extraction prend plusieurs heures.
Le code de M2CAN (dossier m2can/ de ce dépôt) est importé tel quel ; seuls ses chemins par défaut (config.DEFAULTS)
sont remplacés en mémoire par ceux des données téléchargées.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

ICI = os.path.dirname(os.path.abspath(__file__))
M2CAN = os.path.join(os.path.dirname(ICI), "m2can")
GH = "https://raw.githubusercontent.com/jingjiang02/M2CAN/main/split"
DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
ZIPS = {   # identifiant Google Drive, taille exacte (octets), nombre minimal de fichiers
    "TumEmo": ("1tstsycMxGURIR6h3qeedOLxETYJYt-6y", 1463439561, 16500),
    "Twitter": ("1WbXZYmWhfx9fKEmz1fZjMKSTzrGylGhK", 2838739023, 20290),
    "Yelp": ("1qypfwNJPRaKiFzJ-Aap1Nw0Eue1vrlpi", 5683595603, 60446),
    "AVA": ("1DVJD5MwmnX0FecDYRq1FQuE39J0qC9Iz", 553246909, 4235),
    "PCCD": ("1BdXjEyvSiONG7h6jIli22lT9i6vy31I2", 886492747, 4235),
    "RPCD": ("1FWgVKJQuC6wcs_NqTBkPDujJZKlGSe79", 4161232968, 4235),
}
DOMAINES = {"sentiment": ("TumEmo", "Twitter", "Yelp"), "aesthetic": ("AVA", "PCCD", "RPCD")}
# dossier de décompression de chaque .zip, et racine des images attendue par common_loader.get_image
DECOMPRESSION = {"TumEmo": "tumemo", "Twitter": "twitter", "Yelp": "yelp/data",
                 "AVA": "datasets", "PCCD": "datasets", "RPCD": "datasets"}
RACINE = {"TumEmo": "tumemo/TumEmo", "Twitter": "twitter/Twitter", "Yelp": "yelp/data/Yelp",
          "AVA": "datasets/AVA/images4AVA/images", "PCCD": "datasets/PCCD/PCCD/images/full",
          "RPCD": "datasets/RPCD/RPCD"}
SPLIT_SHA = {
    "sentiment/TumEmo/train.json": "81d25820c877b0381831f7fb8ab0180634771a4890d19ec4d283178ec05fb933",
    "sentiment/TumEmo/val.json": "da9df622dedd7c2b04ab1b5fef147db7385d8bc016501d87588a1602a607e4be",
    "sentiment/Twitter/train.json": "fa4df243a0206c9ea78fd16b06b36f394218a18c6c26471ff5c4e4c6d8fd4efd",
    "sentiment/Twitter/val.json": "538a47cd2ccc5bc3cc8eb2e39667c40c90545c901b5e5c07fcb0d53f36984bc9",
    "sentiment/Yelp/train.json": "d7acb80869efc579722b0bf93f59bd8fb65974e6e3fbe651e36ea394b437e1d4",
    "sentiment/Yelp/val.json": "96ea399fa064ae3af6041174fba1773ec6e1007bef2f865e115457acd691c861",
    "aesthetic/AVA/train.json": "8add3a6ef56e39914a28f846127c447a21f762fe5e42a2262b2c48cdb30f2248",
    "aesthetic/AVA/val.json": "9aa9bdb0eb77d2509ab711a7ddc529aad1f53da77ee94b130bb7285bd9d47949",
    "aesthetic/PCCD/train.json": "77563b6c79dc45c59bc63bd6681e8bb75ae37b1aa9c472b2c8d3b3582dd50a9e",
    "aesthetic/PCCD/val.json": "ea25a79a49e4de8967bf314b1c6e3fde9fce008c2c0885c48bd6b49a9dbab7fb",
    "aesthetic/RPCD/train.json": "eb8a5f665a9891a9d80bbd557af83bc6844948861a738e6294a6faabfa1c46fa",
    "aesthetic/RPCD/val.json": "dc4c9351f7147029c1ae70142e8fc91da612e062e3aa3169325865e93a5a1882",
}


def telecharger_zip(nom, brut):
    fid, taille, n_min = ZIPS[nom]
    dest = os.path.join(brut, DECOMPRESSION[nom])
    temoin = os.path.join(brut, f".{nom}_ok")
    if os.path.exists(temoin):
        print(f"  {nom} : déjà prêt", flush=True)
        return
    os.makedirs(brut, exist_ok=True)
    z = os.path.join(brut, f"{nom}.zip")
    for essai in range(6):
        subprocess.run(["curl", "-L", "-sS", "--retry", "5", "--retry-delay", "10", "-C", "-", "-o", z,
                        DRIVE.format(fid)])
        recu = os.path.getsize(z) if os.path.exists(z) else 0
        if recu == taille:
            break
        if recu > taille or 0 < recu < 1_000_000:          # page HTML (quota Google Drive) : on repart de zéro
            os.remove(z)
        time.sleep(30 * (essai + 1))
    else:
        raise RuntimeError(f"{nom}.zip incomplet ({recu}/{taille} octets) : quota Google Drive ? Réessayer plus tard.")
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dest)
    os.remove(z)
    n = sum(1 for _ in glob.iglob(os.path.join(dest, "**", "*.*"), recursive=True))
    assert n >= n_min, f"{nom} : {n} fichiers décompressés, {n_min} attendus"
    open(temoin, "w").close()
    print(f"  {nom} : {taille / 1e9:.2f} Go, {n} fichiers", flush=True)


def decoupes(groupe, brut):
    out = {}
    for d in DOMAINES[groupe]:
        for s in ("train", "val"):
            rel = f"{groupe}/{d}/{s}.json"
            p = os.path.join(brut, "split", rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if not os.path.exists(p):
                subprocess.run(["curl", "-L", "-sS", "--retry", "5", "-o", p, f"{GH}/{rel}"], check=True)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            assert h == SPLIT_SHA[rel], f"découpe {rel} différente de la référence ({h[:12]})"
            out[(d, s)] = p
    print(f"  découpes {groupe} : 6 fichiers, empreintes conformes", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="Préparation cadre A d'un groupe de jeux de M2CAN")
    ap.add_argument("--groupe", choices=sorted(DOMAINES), required=True)
    ap.add_argument("--brut", default="donnees_brutes", help="téléchargements et images décompressées")
    ap.add_argument("--sortie", default=None, help="défaut : donnees/<groupe>")
    ap.add_argument("--vues_fortes", type=int, default=4)
    ap.add_argument("--lot", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--appareil", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()
    brut = os.path.abspath(args.brut)
    sortie = os.path.abspath(args.sortie or os.path.join("donnees", args.groupe))
    doms = DOMAINES[args.groupe]

    print("=== téléchargement ===", flush=True)
    for d in doms:
        telecharger_zip(d, brut)
    listes = decoupes(args.groupe, brut)

    print("=== extraction ===", flush=True)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, SequentialSampler
    from transformers import BertTokenizer
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.appareil == "auto" else args.appareil
    print(f"[appareil] {dev}", flush=True)
    sys.path.insert(0, M2CAN)
    os.chdir(M2CAN)
    import config as m2can_config
    for sp in ("train", "val"):
        for d in doms:
            e = m2can_config.DEFAULTS["dataset"][sp][d]
            e["data_list"], e["data_root"] = listes[(d, sp)], os.path.join(brut, RACINE[d])
    from config import load_config
    from common_modules.utils import prepare_text, prepare_image
    from data_loader.common_loader import common_loader
    from data_loader.utils import demojizer_selector, twitter_preprocessor
    from msmm.models.model_msmm import MsmmModel

    cfg = load_config(os.path.join(M2CAN, "msmm", "scripts", f"config_{doms[-1]}.yaml"))
    cfg["save_path"], cfg["snapshot_dir"] = os.path.join(sortie, "_tmp"), "_tmp"
    os.makedirs(cfg["save_path"], exist_ok=True)
    torch.manual_seed(0)
    model = MsmmModel(cfg).to(dev).eval()
    bert, resnet = model.bert_model, model.model_image
    tok = BertTokenizer.from_pretrained(cfg["dataset"]["bert_model"])
    K = int(cfg["net"]["num_classes"])

    def jeu(d, sp):
        c = dict(cfg["dataset"])
        c.update(c.get(sp, {}))
        c.update(c.get(d, {}))
        n = len(json.load(open(c["data_list"])))
        return common_loader(
            data_list=c["data_list"], root_dir=c["data_root"], splits=sp, text_processor=tok,
            encode_kwargs=c.get("text_encode"), weak_image_kwargs=c.get("transforms_weak"),
            strong_image_kwargs=c.get("transforms_strong"),
            twitter_preprocessor=twitter_preprocessor() if d == "Twitter" else None,
            demojizer=demojizer_selector(c["bert_model"]), seed=0, n_sup=n)

    @torch.no_grad()
    def passe(dset, faible, forte):
        ld = DataLoader(dset, batch_size=args.lot, num_workers=args.workers, sampler=SequentialSampler(dset))
        T, I, Ta, Ia, Y, ids = [], [], [], [], [], []
        for b in ld:
            for k in ("input_ids", "token_type_ids", "attention_mask", "input_ids_aug", "token_type_ids_aug",
                      "attention_mask_aug", "image", "image_aug", "image_mask"):
                b[k] = b[k].to(dev)
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
                if faible:
                    T.append(prepare_text(bert, b).float().cpu())
                    I.append(prepare_image(resnet, b["image"], b["image_mask"]).float().cpu())
                if forte:
                    Ta.append(prepare_text(bert, b, aug=True)[1].float().cpu())
                    Ia.append(prepare_image(resnet, b["image_aug"], b["image_mask"]).float().cpu())
            Y.append(b["label"].cpu())
            ids += [str(i) for i in b["id"]]
        cat = lambda L: torch.cat(L).numpy() if L else None
        return cat(T), cat(I), cat(Ta), cat(Ia), torch.cat(Y).numpy(), ids

    resume = {}
    for d in doms:
        for sp, sp_out in (("train", "train"), ("val", "test")):
            t0 = time.time()
            o = os.path.join(sortie, d, sp_out)
            os.makedirs(o, exist_ok=True)
            dset = jeu(d, sp)
            fortes = sp == "train" and args.vues_fortes > 0
            T, I, Ta, Ia, Y, ids = passe(dset, True, fortes)
            if fortes:
                TA, IA = [Ta], [Ia]
                for _ in range(1, args.vues_fortes):
                    _, _, ta, ia, y2, ids2 = passe(dset, False, True)
                    assert ids2 == ids and (y2 == Y).all(), "ordre des exemples instable entre passages"
                    TA.append(ta)
                    IA.append(ia)
                np.save(os.path.join(o, "text_aug.npy"), np.stack(TA, 1).astype(np.float16))
                np.save(os.path.join(o, "image_aug.npy"), np.stack(IA, 1).astype(np.float16))
            np.save(os.path.join(o, "text.npy"), T.astype(np.float32))
            np.save(os.path.join(o, "image.npy"), I.astype(np.float32))
            np.save(os.path.join(o, "labels.npy"), Y.astype(np.int64))
            np.save(os.path.join(o, "ids.npy"), np.array(ids))
            resume[f"{d}/{sp_out}"] = {"n": int(len(Y)), "classes": np.bincount(Y, minlength=K).tolist(),
                                      "minutes": round((time.time() - t0) / 60, 1)}
            print(f"[extrait] {d}/{sp_out} : {json.dumps(resume[f'{d}/{sp_out}'])}", flush=True)
    meta = {"classes": K, "modalites": ["text", "image"], "domaines": list(doms), "groupe": args.groupe,
            "dims": {"text": 768, "image": 2048}, "vues_fortes": args.vues_fortes,
            "encodeurs": f"BERT (config_{doms[-1]}.yaml, sortie pooler) et ResNet-50 ImageNet de M2CAN, poids "
                         "pré-entraînés non ajustés, mode évaluation",
            "decoupes": "train et val de M2CAN (val -> test)", "resume": resume}
    with open(os.path.join(sortie, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"=== terminé : {sortie} ===", flush=True)


if __name__ == "__main__":
    main()
