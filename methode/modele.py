"""Modèle de la fusion géométrique pour M modalités : un encodeur et une tête par modalité, une fusion M-aire.

Aucun classifieur n'est entraîné sur l'espace fusionné z : c'est le principe de la méthode.
"""
import torch.nn as nn

from encodeurs import DIM, EncodeurModalite
from fusion import FusionMLB


def tete(K):
    """Tête unimodale de M2CAN : 256 -> 128 -> 64 -> K."""
    return nn.Sequential(nn.Linear(DIM, 128), nn.ReLU(inplace=True), nn.Linear(128, 64), nn.Linear(64, K))


class ModeleFusionGeometrique(nn.Module):
    def __init__(self, modalites, K, dim_fusion=None):
        """modalites : liste ordonnée de dictionnaires {"nom": ..., "encodeur": {...}} ;
        dim_fusion : dimension de z (64 si None, comme M2CAN)."""
        super().__init__()
        self.noms = [m["nom"] for m in modalites]
        self.M, self.K = len(self.noms), K
        if self.M < 2:
            raise ValueError("il faut au moins 2 modalités")
        self.encodeurs = nn.ModuleDict({m["nom"]: EncodeurModalite(m["encodeur"]) for m in modalites})
        self.tetes = nn.ModuleDict({n: tete(K) for n in self.noms})
        self.fusion = FusionMLB(self.M, sortie=dim_fusion)

    def encoder(self, entrees):
        """entrees : {modalité: tenseur}. Renvoie la liste des M représentations 256-d, dans l'ordre."""
        return [self.encodeurs[n](entrees[n]) for n in self.noms]

    def logits_tetes(self, feats):
        return [self.tetes[n](f) for n, f in zip(self.noms, feats)]

    def fusionner(self, feats):
        return self.fusion(feats)
