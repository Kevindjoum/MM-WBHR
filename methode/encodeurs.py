"""Encodeurs par modalité.

Chaque modalité m passe par deux blocs :
  1. une dorsale (backbone) qui produit un vecteur de dimension d_m ;
  2. un extracteur Linear(d_m, 256) + ReLU, puis normalisation L2 (comme les extracteurs de M2CAN).

Types de dorsale disponibles (clé `type` de la configuration) :
  - `caracteristiques` : l'entrée est déjà un vecteur (caractéristiques extraites une fois : SlowFast, AST,
    ResNet, BERT...). La dorsale est l'identité. C'est le mode testé.
  - `bert` : modèle Hugging Face (texte) ; l'entrée est un dictionnaire de jetons ; sortie = jeton [CLS].
  - `torchvision` : réseau d'images torchvision (resnet50 par défaut) ; l'entrée est un tenseur d'images.
Pour `bert` et `torchvision`, le jeu de données doit fournir les entrées brutes correspondantes.

Ajouter un nouveau type : écrire une classe nn.Module qui renvoie un tenseur (B, d_m), puis l'enregistrer
dans DORSALES.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 256


class Identite(nn.Module):
    def forward(self, x):
        return x


class DorsaleBert(nn.Module):
    def __init__(self, nom="bert-base-uncased"):
        super().__init__()
        from transformers import AutoModel
        self.net = AutoModel.from_pretrained(nom)
        self.dim = self.net.config.hidden_size

    def forward(self, jetons):
        return self.net(**jetons).last_hidden_state[:, 0]


class DorsaleTorchvision(nn.Module):
    def __init__(self, nom="resnet50", pre_entraine=True):
        super().__init__()
        import torchvision
        net = getattr(torchvision.models, nom)(weights="DEFAULT" if pre_entraine else None)
        self.dim = net.fc.in_features
        net.fc = nn.Identity()
        self.net = net

    def forward(self, images):
        return self.net(images)


DORSALES = {"caracteristiques": None, "bert": DorsaleBert, "torchvision": DorsaleTorchvision}


class EncodeurModalite(nn.Module):
    """Dorsale + extracteur 256-d normalisé."""

    def __init__(self, conf):
        super().__init__()
        t = conf.get("type", "caracteristiques")
        if t == "caracteristiques":
            self.dorsale, d = Identite(), int(conf["dim"])
        elif t in DORSALES:
            kwargs = {k: v for k, v in conf.items() if k not in ("type", "dim")}
            self.dorsale = DORSALES[t](**kwargs)
            d = self.dorsale.dim
        else:
            raise ValueError(f"type d'encodeur inconnu : {t} (disponibles : {sorted(DORSALES)})")
        self.extracteur = nn.Sequential(nn.Linear(d, DIM), nn.ReLU(inplace=True))

    def forward(self, x):
        return F.normalize(self.extracteur(self.dorsale(x)), dim=-1)
