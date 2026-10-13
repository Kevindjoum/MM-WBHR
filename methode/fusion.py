"""Fusion M-aire des représentations unimodales (256-d chacune) vers l'espace fusionné z (64-d par défaut).

M = 2 : MLB de M2CAN, à l'identique :
    z = P(sigmoid(U e_0 ∘ V e_1)) + h_1(e_0) + h_2(e_1)
M >= 3 : généralisation directe (produit terme à terme des M projections) :
    z = P(sigmoid(U_0 e_0 ∘ ... ∘ U_{M-1} e_{M-1})) + sum_m h_m(e_m)

z n'est PAS normalisé (comme dans le code d'origine). Sa dimension vaut 64 (M2CAN) sauf si `sortie` est donné.
"""
import torch
import torch.nn as nn

from encodeurs import DIM


class FusionMLB(nn.Module):
    def __init__(self, M, dim=DIM, sortie=None):
        super().__init__()
        self.M = M
        sortie = sortie or dim // 4
        self.U = nn.ModuleList([nn.Linear(dim, dim // 2) for _ in range(M)])
        self.P = nn.Linear(dim // 2, sortie)
        self.h = nn.ModuleList([nn.Linear(dim, sortie) for _ in range(M)])

    def forward(self, feats):
        """feats : liste de M tenseurs (B, 256), dans l'ordre des modalités de la configuration."""
        prod = self.U[0](feats[0])
        for m in range(1, self.M):
            prod = prod * self.U[m](feats[m])
        out = self.P(torch.sigmoid(prod))
        for m in range(self.M):
            out = out + self.h[m](feats[m])
        return out
