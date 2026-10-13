"""HAC : extraction des caracteristiques gelees (cadre A), pour la methode et pour M2CAN.

    python preparation/hac_extraire.py --sortie donnees_brutes/hac_feats --brut donnees_brutes/hac
    python preparation/hac_convertir.py --feats donnees_brutes/hac_feats --sortie donnees/hac --video slowfast --vues_fortes

HAC est telecharge depuis Hugging Face (hdong51/Human-Animal-Cartoon). GPU detecte automatiquement ; sur
processeur, l'extraction prend plusieurs heures. Dependances supplementaires : av, soundfile, fvcore, iopath
(SlowFast via pytorchvideo, charge par torch.hub).

Pour chacun des 3381 clips listes dans les CSV officiels, une vue faible + 4 vues
fortes par encodeur :

  video, encodeur 1 — ResNet-50 DE M2CAN (common_modules/base/resnet.py, poids
          ImageNet resnet50-0676ba61), fc = identite, sur 8 frames reparties sur
          le clip (TSN), moyenne -> 2048-d. C'est l'operation de prepare_image
          sur un post a plusieurs images : variante « fidele a M2CAN ».
            faible : centre des 8 segments + transforms_weak de M2CAN
                     (resize 256, center crop 224)
            fortes : tirage aleatoire par segment + transforms_strong de M2CAN
                     (random resized crop, flip 0.5, color jitter p=0.8, gris p=0.2)
  video, encodeur 2 — SlowFast R101 8x8 Kinetics-400 (pytorchvideo), l'encodeur
          RGB de SimMMDG, l'article de reference de HAC -> 2304-d
          (2048 voie lente + 256 voie rapide). 32 frames au pas de 2, voie lente
          = 1 frame sur 4, normalisation Kinetics de pytorchvideo, 256 px.
            faible : fenetre centrale + center crop 256
            fortes : fenetre aleatoire + memes augmentations M2CAN (en 256)
  audio — AST AudioSet (MIT/ast-finetuned-audioset-10-10-0.4593), pooler 768-d,
          le pendant audio du pooler BERT-base.
            faible : clip entier ; fortes : decalage circulaire + gain + SpecAugment

Robustesse (le premier essai a ete tue sans journal, vraisemblablement faute de
memoire) : decodage dans des THREADS (pas de memoire partagee entre processus),
frames ramenees a 256 px de petit cote DES le decodage, seules les frames utiles
sont converties en RGB, transformations sur GPU, progression ecrite au fil de l'eau.

Sorties (dossier --sortie) :
  <domaine>/{clips,labels}.npy
  <domaine>/{audio,video,sf}_{weak,strong}.npy   (video = ResNet-50, sf = SlowFast)
  splits/<domaine>_{train,test}.json             (format des data_list de M2CAN)
  meta.json                                      (reglages, echecs, sondes lineaires)
"""
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import argparse

_ap = argparse.ArgumentParser(description="Extraction des caracteristiques HAC (cadre A)")
_ap.add_argument("--sortie", default="donnees_brutes/hac_feats", help="dossier des caracteristiques extraites")
_ap.add_argument("--brut", default="donnees_brutes/hac", help="telechargements HAC (Hugging Face)")
_ap.add_argument("--limite", type=int, default=None, help="test rapide : clips par (domaine, split)")
_ap.add_argument("--appareil", default="auto", choices=["auto", "cuda", "cpu"])
_ARGS = _ap.parse_args()
LIMIT = _ARGS.limite
OUT = os.path.abspath(_ARGS.sortie)
BRUT = os.path.abspath(_ARGS.brut)
os.makedirs(OUT, exist_ok=True)
PROGRESS = os.path.join(OUT, "progress.txt")


def log(msg):
    print(msg, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(msg + "\n")



import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF

HF_REPO = "hdong51/Human-Animal-Cartoon"
DOMAINS = ("human", "animal", "cartoon")
N_STRONG = 4
TSN_FRAMES = 8                                  # ResNet-50 : 8 frames sur tout le clip
SF_FRAMES, SF_INTERVAL, SF_ALPHA = 32, 2, 4     # SlowFast 8x8 : 32 frames, pas de 2, lente = 1/4
SHORT_SIDE = 256
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
IMNET_MEAN, IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # transforms de M2CAN
KIN_MEAN, KIN_STD = [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]         # pytorchvideo / Kinetics
SEED = 123456
HAC = None


def prepare_hac():
    """HAC depuis Hugging Face, sans aucun dataset Kaggle monte : depuis la mise a jour
    d'image d'octobre 2026, tout kernel qui monte une entree echoue avant meme de demarrer
    (verifie avec des sondes triviales). Seuls videos/ et audio/ sont decompresses : le
    flot optique ne sert pas a la paire video+audio."""
    import shutil
    from huggingface_hub import hf_hub_download
    root, dl = os.path.join(BRUT, "HAC"), os.path.join(BRUT, "telechargements")
    os.makedirs(root, exist_ok=True)
    os.makedirs(dl, exist_ok=True)
    for d in DOMAINS:
        for split in ("train", "test"):
            f = f"HAC_{split}_only_{d}.csv"
            shutil.copy(hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=f, local_dir=dl),
                        os.path.join(root, f))
    for d in DOMAINS:
        t = time.time()
        z = hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=f"{d}.zip", local_dir=dl)
        subprocess.run(["unzip", "-q", "-o", z, f"{d}/videos/*", f"{d}/audio/*", "-d", root])
        if not os.path.isdir(os.path.join(root, d, "videos")):     # disposition inattendue : tout extraire
            ex = os.path.join(BRUT, f"_ex_{d}")
            subprocess.run(["unzip", "-q", "-o", z, "-d", ex], check=True)
            for r_, dirs, _ in os.walk(ex):
                if "videos" in dirs and "audio" in dirs:
                    shutil.move(r_, os.path.join(root, d))
                    break
            shutil.rmtree(ex, ignore_errors=True)
        os.remove(z)
        log(f"  {d} : {len(os.listdir(os.path.join(root, d, 'videos')))} videos, "
            f"{len(os.listdir(os.path.join(root, d, 'audio')))} audios  ({time.time() - t:.0f} s)")
    return root


def read_csv(path):
    with open(path, newline="") as fh:
        return [(r[0].strip(), int(r[1])) for r in csv.reader(fh) if r and r[0].strip()]


# ------------------------------------------------------------------ plans temporels
def tsn_indices(n, rng):
    if rng is None:
        return [min(n - 1, int((i + 0.5) * n / TSN_FRAMES)) for i in range(TSN_FRAMES)]
    return [min(n - 1, int((i + rng.random()) * n / TSN_FRAMES)) for i in range(TSN_FRAMES)]


def window_indices(n, rng):
    span = SF_FRAMES * SF_INTERVAL
    start = 0 if n <= span else ((n - span) // 2 if rng is None else rng.randint(0, n - span))
    return [min(n - 1, start + i * SF_INTERVAL) for i in range(SF_FRAMES)]


# ------------------------------------------------------------------ decodage (threads)
def _open(path):
    import av
    c = av.open(path)
    st = c.streams.video[0]
    st.thread_type = "AUTO"
    return c, st


def decode_clip(path, rng):
    """{'tsn': [5 x (8,h,w,3)], 'sf': [5 x (32,h,w,3)]} uint8, petit cote = 256 px."""
    c, st = _open(path)
    try:
        n = st.frames or 0
        if n <= 0:                                 # nombre inconnu : on compte d'abord
            n = sum(1 for _ in c.decode(st))
            c.close()
            c, st = _open(path)
        if n <= 0:
            raise RuntimeError("0 frame")
        H = st.codec_context.height or st.height
        W = st.codec_context.width or st.width
        s = SHORT_SIDE / min(H, W)
        h2, w2 = max(2, int(round(H * s / 2)) * 2), max(2, int(round(W * s / 2)) * 2)
        plans = {"tsn": [tsn_indices(n, None)] + [tsn_indices(n, rng) for _ in range(N_STRONG)],
                 "sf": [window_indices(n, None)] + [window_indices(n, rng) for _ in range(N_STRONG)]}
        need = {i for views in plans.values() for idx in views for i in idx}
        stop, frames = max(need), {}
        for k, f in enumerate(c.decode(st)):
            if k in need:
                frames[k] = f.to_ndarray(format="rgb24", width=w2, height=h2)
            if k >= stop:
                break
    finally:
        c.close()
    if not frames:
        raise RuntimeError("aucune frame decodee")
    avail = np.array(sorted(frames))

    def pick(i):                                   # metadonnee surestimee : frame la plus proche
        return frames[i] if i in frames else frames[int(avail[np.abs(avail - i).argmin()])]

    return {key: [np.stack([pick(i) for i in idx]) for idx in views] for key, views in plans.items()}


_FE = None


def feature_extractor():
    global _FE
    if _FE is None:
        from transformers import ASTFeatureExtractor
        _FE = ASTFeatureExtractor.from_pretrained(AST_ID)
    return _FE


def load_wav(path):
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), 16000)
        wav = resample_poly(wav, 16000 // g, int(sr) // g).astype(np.float32)
    if wav.size < 2:
        raise RuntimeError("audio vide")
    return wav


def spec_augment(x, rng):
    x = x.clone()
    for _ in range(2):
        f = rng.randint(0, 24)
        f0 = rng.randint(0, x.shape[1] - f)
        x[:, f0:f0 + f] = 0
    for _ in range(2):
        t = rng.randint(0, 96)
        t0 = rng.randint(0, x.shape[0] - t)
        x[t0:t0 + t, :] = 0
    return x


def audio_views(path, rng):
    wav = load_wav(path)
    views = [wav] + [(np.roll(wav, rng.randint(0, len(wav) - 1)) * rng.uniform(0.8, 1.2)).astype(np.float32)
                     for _ in range(N_STRONG)]
    feats = feature_extractor()(views, sampling_rate=16000, return_tensors="pt")["input_values"]
    for v in range(1, 1 + N_STRONG):
        feats[v] = spec_augment(feats[v], rng)
    return feats                                   # float32 (5, 1024, 128)


def load_clip(item):
    d, clip, y = item
    rng = random.Random(f"{SEED}-{d}-{clip}")      # reproductible quel que soit l'ordre des threads
    out = {"domain": d, "clip": clip, "label": y, "err": "", "video": None, "audio": None}
    try:
        out["video"] = decode_clip(os.path.join(HAC, d, "videos", clip), rng)
    except Exception as e:
        out["err"] += f"video:{e!r} "
    try:
        out["audio"] = audio_views(os.path.join(HAC, d, "audio", clip[:-4] + ".wav"), rng)
    except Exception as e:
        out["err"] += f"audio:{e!r} "
    return out


def prefetch(items, workers=4, window=8):
    """Charge les clips dans des threads, rend les resultats DANS L'ORDRE."""
    with ThreadPoolExecutor(workers) as ex:
        it, futs = iter(items), []
        for item in it:
            futs.append(ex.submit(load_clip, item))
            if len(futs) >= window:
                break
        while futs:
            f = futs.pop(0)
            nxt = next(it, None)
            if nxt is not None:
                futs.append(ex.submit(load_clip, nxt))
            yield f.result()


# ------------------------------------------------------------------ transformations (GPU)
_CJ = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)


def weak_view(x, size):                            # petit cote deja a 256 : resize 256 = identite
    return TF.center_crop(x, size)


def strong_view(x, size):                          # transforms_strong de M2CAN, memes parametres par clip
    i, j, h, w = T.RandomResizedCrop.get_params(x[0], scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3))
    x = TF.resized_crop(x, i, j, h, w, [size, size], antialias=True)
    if torch.rand(1).item() < 0.5:
        x = TF.hflip(x)
    if torch.rand(1).item() < 0.8:
        fn_idx, b, c, s, hue = T.ColorJitter.get_params(_CJ.brightness, _CJ.contrast,
                                                        _CJ.saturation, _CJ.hue)
        for fid in fn_idx.tolist():
            if fid == 0:
                x = TF.adjust_brightness(x, b)
            elif fid == 1:
                x = TF.adjust_contrast(x, c)
            elif fid == 2:
                x = TF.adjust_saturation(x, s)
            else:
                x = TF.adjust_hue(x, hue)
    if torch.rand(1).item() < 0.2:
        x = TF.rgb_to_grayscale(x, num_output_channels=3)
    return x


def views_on_gpu(arrs, size, dev):
    """arrs : 5 x (F,h,w,3) uint8 -> (5, F, 3, size, size) float [0,1] sur GPU."""
    out = []
    for v, a in enumerate(arrs):
        x = torch.from_numpy(a).to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        out.append(weak_view(x, size) if v == 0 else strong_view(x, size))
    return torch.stack(out)


def load_slowfast(dev):
    m = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r101", pretrained=True, trust_repo=True)
    head = m.blocks[-1]
    head.proj = nn.Identity()                      # tete Kinetics retiree : features poolees 2304-d
    if getattr(head, "activation", None) is not None:
        head.activation = None
    m.eval().to(dev)
    with torch.no_grad():
        o = m([torch.zeros(1, 3, SF_FRAMES // SF_ALPHA, 256, 256, device=dev),
               torch.zeros(1, 3, SF_FRAMES, 256, 256, device=dev)])
    log(f"SlowFast R101 charge : sortie {tuple(o.shape)}")
    assert o.shape[-1] == 2304, o.shape
    return m


def linear_probe(store, key):
    """Plancher indicatif : regression logistique entrainee sur les 2 sources (train,
    vues faibles video+audio), testee sur le test cible, sans aucune adaptation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    res = {}
    for tgt in DOMAINS:
        Xs, ys = [], []
        for d in DOMAINS:
            if d == tgt:
                continue
            m = store[d]["split"] == "train"
            Xs.append(np.hstack([store[d][key + "_w"][m], store[d]["audio_w"][m]]))
            ys.append(store[d]["labels"][m])
        m = store[tgt]["split"] == "test"
        Xt = np.hstack([store[tgt][key + "_w"][m], store[tgt]["audio_w"][m]])
        sc = StandardScaler().fit(np.vstack(Xs))
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(np.vstack(Xs)), np.concatenate(ys))
        res[tgt] = float((clf.predict(sc.transform(Xt)) == store[tgt]["labels"][m]).mean())
    return res


def main():
    global HAC
    torch.manual_seed(SEED)
    import av
    import transformers
    log(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
        f"transformers {transformers.__version__} | av {av.__version__} | cpu {os.cpu_count()}")
    if _ARGS.appareil == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(_ARGS.appareil)
    log(f"[appareil] {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else " : extraction lente sur processeur"))
    log("=== telechargement HAC (Hugging Face) ===")
    HAC = prepare_hac()
    log(f"HAC : {HAC}")

    # ResNet-50 de torchvision, IMAGENET1K_V1 : exactement le fichier de poids et l'architecture
    # de M2CAN (common_modules/base/resnet.py copie torchvision, model_urls = resnet50-0676ba61.pth)
    from torchvision.models import resnet50, ResNet50_Weights
    w = ResNet50_Weights.IMAGENET1K_V1
    assert w.url.endswith("resnet50-0676ba61.pth"), w.url
    net = resnet50(weights=w)
    net.fc = nn.Identity()
    net.eval().to(dev)
    sf_net = load_slowfast(dev)
    from transformers import ASTModel
    ast = ASTModel.from_pretrained(AST_ID).eval().to(dev)
    im_mean = torch.tensor(IMNET_MEAN, device=dev).view(1, 1, 3, 1, 1)
    im_std = torch.tensor(IMNET_STD, device=dev).view(1, 1, 3, 1, 1)
    k_mean = torch.tensor(KIN_MEAN, device=dev).view(1, 1, 3, 1, 1)
    k_std = torch.tensor(KIN_STD, device=dev).view(1, 1, 3, 1, 1)
    slow_idx = torch.linspace(0, SF_FRAMES - 1, SF_FRAMES // SF_ALPHA).long().to(dev)

    os.makedirs(os.path.join(OUT, "splits"), exist_ok=True)
    items, split_of, rows_of = [], {}, {}
    for d in DOMAINS:
        for split in ("train", "test"):
            rows = read_csv(os.path.join(HAC, f"HAC_{split}_only_{d}.csv"))
            if LIMIT is not None:
                rows = rows[:LIMIT]
            rows_of[(d, split)] = rows
            for c, y in rows:
                if (d, c) not in split_of:
                    split_of[(d, c)] = split
                    items.append((d, c, y))
    log(f"{len(items)} clips a extraire (LIMIT={LIMIT})")
    feature_extractor()                            # initialise avant les threads (pas de course)

    store = {d: {k: [] for k in ("clips", "labels", "split", "audio_w", "audio_s",
                                  "video_w", "video_s", "sf_w", "sf_s")} for d in DOMAINS}
    fails, t0 = [], time.time()
    with torch.no_grad():
        for k, s in enumerate(prefetch(items)):
            d = s["domain"]
            if s["video"] is None or s["audio"] is None:
                fails.append({"domain": d, "clip": s["clip"], "err": s["err"]})
                log(f"  [echec] {d}/{s['clip']}: {s['err']}")
                continue
            # ResNet-50 (M2CAN) : 5 vues x 8 frames, 224 px, normalisation ImageNet
            v = views_on_gpu(s["video"]["tsn"], 224, dev)                   # (5, 8, 3, 224, 224)
            v = ((v - im_mean) / im_std).flatten(0, 1)
            fv = net(v).view(1 + N_STRONG, TSN_FRAMES, -1).mean(1)          # (5, 2048)
            # SlowFast R101 : 5 vues x 32 frames, 256 px, normalisation Kinetics
            x = views_on_gpu(s["video"]["sf"], 256, dev)                    # (5, 32, 3, 256, 256)
            fast = ((x - k_mean) / k_std).permute(0, 2, 1, 3, 4).contiguous()   # (5, 3, 32, 256, 256)
            fsf = sf_net([fast.index_select(2, slow_idx), fast])           # (5, 2304)
            # AST : 5 vues
            fa = ast(input_values=s["audio"].to(dev)).pooler_output       # (5, 768)
            st = store[d]
            st["clips"].append(s["clip"])
            st["labels"].append(s["label"])
            st["split"].append(split_of[(d, s["clip"])])
            for key, val in (("video", fv), ("sf", fsf), ("audio", fa)):
                st[key + "_w"].append(val[0].float().cpu().numpy())
                st[key + "_s"].append(val[1:].float().cpu().numpy())
            if (k + 1) % 100 == 0 or k + 1 == len(items):
                el = time.time() - t0
                log(f"  {k + 1}/{len(items)}  {el / 60:.1f} min  "
                    f"ETA {el / (k + 1) * (len(items) - k - 1) / 60:.1f} min  "
                    + (f"(gpu max {torch.cuda.max_memory_allocated() / 2**30:.1f} Go)" if dev.type == "cuda" else ""))

    for d in DOMAINS:
        st = store[d]
        for key in list(st):
            if key != "clips":
                st[key] = np.asarray(st[key])
        dd = os.path.join(OUT, d)
        os.makedirs(dd, exist_ok=True)
        np.save(os.path.join(dd, "clips.npy"), np.asarray(st["clips"]))
        np.save(os.path.join(dd, "labels.npy"), st["labels"].astype(np.int64))
        for key in ("audio", "video", "sf"):
            for suf, name in (("_w", "weak"), ("_s", "strong")):
                arr = st[key + suf]
                assert np.isfinite(arr).all(), f"valeurs non finies dans {d}/{key}{suf}"
                np.save(os.path.join(dd, f"{key}_{name}.npy"), arr.astype(np.float16))
        log(f"{d}: {len(st['clips'])} clips  audio {st['audio_w'].shape}  "
            f"resnet {st['video_w'].shape}  slowfast {st['sf_w'].shape}  fortes {st['sf_s'].shape}")

    bad = {(f["domain"], f["clip"]) for f in fails}
    for (d, split), rows in rows_of.items():
        rows = [(c, y) for c, y in rows if (d, c) not in bad]
        with open(os.path.join(OUT, "splits", f"{d}_{split}.json"), "w") as fh:
            json.dump([{"image": [f"{d}/{c}"], "label": y, "text": ""} for c, y in rows], fh)
        log(f"splits/{d}_{split}.json : {len(rows)}")

    probes = {}
    for key, name in (("video", "resnet50"), ("sf", "slowfast")):
        try:
            probes[name] = linear_probe(store, key)
            log(f"sonde lineaire source-only [{name} + AST] : "
                f"{ {t: round(v * 100, 2) for t, v in probes[name].items()} }")
        except Exception as e:
            log(f"sonde lineaire [{name}] impossible : {e!r}")

    meta = {
        "limit": LIMIT, "n_strong": N_STRONG, "seed": SEED,
        "encoders": {
            "video": "ResNet-50 IMAGENET1K_V1 [resnet50-0676ba61, memes poids et architecture que "
                     "common_modules/base/resnet.py de M2CAN], fc=Identity, moyenne sur 8 frames TSN, "
                     "224 px -> 2048",
            "sf": "pytorchvideo slowfast_r101 (8x8, Kinetics-400), tete retiree, "
                  "32 frames pas 2, lente = 1/4, 256 px -> 2304",
            "audio": f"{AST_ID} pooler_output -> 768",
        },
        "augment_video": "faible = centre ; fortes = tirage temporel aleatoire + random resized crop "
                         "+ flip 0.5 + color jitter(0.4,0.4,0.4,0.1) p=0.8 + gris p=0.2 (M2CAN), par clip",
        "augment_audio": "roll circulaire + gain U(0.8,1.2) + 2 masques freq<=24 + 2 masques temps<=96",
        "counts": {d: int(len(store[d]["clips"])) for d in DOMAINS},
        "failures": fails,
        "linear_probe_source_only": probes,
        "minutes": round((time.time() - t0) / 60, 2),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    with open(os.path.join(OUT, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    log(f"echecs : {len(fails)}")
    log(f"TERMINE en {meta['minutes']} min")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        sys.exit(1)
