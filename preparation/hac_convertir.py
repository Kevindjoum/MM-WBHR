"""Convertit les caractéristiques HAC déjà extraites (kernel hac-extract-features) vers le format de donnees.py.

Entrée : dossier hac_feats/ avec <domaine>/{clips,labels,audio_weak,video_weak,sf_weak}.npy et
splits/<domaine>_{train,test}.json. Sortie : <sortie>/<domaine>/{train,test}/{audio,video,labels,ids}.npy.
Avec --vues_fortes, les vues augmentées fortes extraites par hac-extract-features (<mod>_strong.npy, 4 vues par
clip, les augmentations de M2CAN) sont écrites en {audio,video}_aug.npy ; sans elles, la vue augmentée est la vue
faible elle-même.

    python preparer_hac.py --feats chemin/hac_feats --sortie donnees/hac --video slowfast
"""
import argparse
import json
import os

import numpy as np

DOMAINES = ("human", "animal", "cartoon")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--video", choices=["slowfast", "resnet"], default="slowfast")
    ap.add_argument("--vues_fortes", action="store_true", help="écrire aussi les vues augmentées fortes")
    args = ap.parse_args()
    fichier_video = {"slowfast": "sf_weak.npy", "resnet": "video_weak.npy"}[args.video]
    for dom in DOMAINES:
        d = os.path.join(args.feats, dom)
        clips = [str(c) for c in np.load(os.path.join(d, "clips.npy"))]
        ligne = {c: i for i, c in enumerate(clips)}
        V = np.load(os.path.join(d, fichier_video)).astype(np.float32)
        A = np.load(os.path.join(d, "audio_weak.npy")).astype(np.float32)
        if args.vues_fortes:
            Vf = np.load(os.path.join(d, fichier_video.replace("_weak", "_strong"))).astype(np.float32)
            Af = np.load(os.path.join(d, "audio_strong.npy")).astype(np.float32)
            if Vf.shape[0] != V.shape[0] or Af.shape[0] != A.shape[0]:
                raise ValueError(f"{dom} : vues fortes {Vf.shape}, {Af.shape} incompatibles avec {V.shape}, {A.shape}")
        for split in ("train", "test"):
            with open(os.path.join(args.feats, "splits", f"{dom}_{split}.json")) as f:
                lignes = json.load(f)
            idx = [ligne[l["image"][0].split("/", 1)[1]] for l in lignes]
            o = os.path.join(args.sortie, dom, split)
            os.makedirs(o, exist_ok=True)
            np.save(os.path.join(o, "video.npy"), V[idx])
            np.save(os.path.join(o, "audio.npy"), A[idx])
            if args.vues_fortes:
                np.save(os.path.join(o, "video_aug.npy"), Vf[idx])
                np.save(os.path.join(o, "audio_aug.npy"), Af[idx])
            np.save(os.path.join(o, "labels.npy"), np.array([int(l["label"]) for l in lignes]))
            np.save(os.path.join(o, "ids.npy"), np.array([l["image"][0] for l in lignes]))
            print(f"{dom}/{split} : {len(idx)} clips, video {V.shape[1]}-d, audio {A.shape[1]}-d"
                  + (f", vues fortes {Vf.shape[1:]} / {Af.shape[1:]}" if args.vues_fortes else ""))
    with open(os.path.join(args.sortie, "meta.json"), "w") as f:
        json.dump({"classes": 7, "modalites": ["audio", "video"], "domaines": list(DOMAINES)}, f)


if __name__ == "__main__":
    main()
