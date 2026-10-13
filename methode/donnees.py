"""Jeux de données de caractéristiques pré-extraites, pour un nombre quelconque de modalités.

Format attendu sur disque :

    <racine>/meta.json                       {"classes": K, "modalites": [...], "domaines": [...]}
    <racine>/<domaine>/<split>/labels.npy    (N,) entiers dans [0, K)
    <racine>/<domaine>/<split>/ids.npy       (N,) identifiants (chaînes)
    <racine>/<domaine>/<split>/<modalite>.npy        (N, d_m) vue « faible »
    <racine>/<domaine>/<split>/<modalite>_aug.npy    optionnel : (N, d_m) ou (N, A, d_m) vues augmentées

<split> vaut "train" ou "test". Sans fichier _aug, la vue augmentée est la vue faible elle-même.
Le script preparer_hac.py convertit les caractéristiques HAC déjà extraites vers ce format.
"""
import json
import os

import numpy as np
import torch


class JeuCaracteristiques:
    def __init__(self, racine, domaine, split, modalites, appareil="cpu"):
        d = os.path.join(racine, domaine, split)
        self.domaine, self.modalites, self.appareil = domaine, modalites, appareil
        self.y = np.load(os.path.join(d, "labels.npy")).astype(np.int64)
        self.ids = [str(i) for i in np.load(os.path.join(d, "ids.npy"), allow_pickle=True)]
        self.x, self.x_aug = {}, {}
        for m in modalites:
            self.x[m] = torch.tensor(np.load(os.path.join(d, f"{m}.npy")), dtype=torch.float32)
            p = os.path.join(d, f"{m}_aug.npy")
            self.x_aug[m] = torch.tensor(np.load(p), dtype=torch.float32) if os.path.exists(p) else None
        self.n = len(self.y)

    def lot(self, idx, rng):
        """Lot des indices idx : vues faible et augmentée (une vue augmentée tirée au hasard si plusieurs)."""
        x = {m: self.x[m][idx].to(self.appareil) for m in self.modalites}
        x_aug = {}
        for m in self.modalites:
            a = self.x_aug[m]
            if a is None:
                x_aug[m] = x[m]
            elif a.dim() == 3:
                k = rng.integers(0, a.shape[1], len(idx))
                x_aug[m] = a[idx, k].to(self.appareil)
            else:
                x_aug[m] = a[idx].to(self.appareil)
        return {"x": x, "x_aug": x_aug, "y": torch.tensor(self.y[idx], device=self.appareil),
                "ids": [self.ids[i] for i in idx]}

    def tout(self, taille=256):
        """Itère sur tout le jeu, vues faibles, dans l'ordre (évaluation)."""
        for s in range(0, self.n, taille):
            idx = np.arange(s, min(s + taille, self.n))
            yield {"x": {m: self.x[m][idx].to(self.appareil) for m in self.modalites},
                   "y": self.y[idx], "ids": [self.ids[i] for i in idx]}


class EchantillonneurDomaines:
    """Tire à chaque pas un lot de même taille dans chaque domaine (sources puis cible), en cyclant."""

    def __init__(self, jeux, taille_lot, graine):
        self.jeux, self.B = jeux, taille_lot
        self.rng = np.random.default_rng(graine)
        self.ordres = [self.rng.permutation(j.n) for j in jeux]
        self.pos = [0] * len(jeux)

    def suivant(self):
        lots = []
        for k, j in enumerate(self.jeux):
            if self.pos[k] + self.B > j.n:
                self.ordres[k], self.pos[k] = self.rng.permutation(j.n), 0
            idx = self.ordres[k][self.pos[k]:self.pos[k] + self.B]
            self.pos[k] += self.B
            lots.append(j.lot(idx, self.rng))
        return lots


def lire_meta(racine):
    with open(os.path.join(racine, "meta.json")) as f:
        return json.load(f)
