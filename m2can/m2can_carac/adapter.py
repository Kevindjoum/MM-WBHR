"""
Adaptateur generique : le cadre M2CAN (msmm/train.py) sur des caracteristiques gelees au format commun.

Format commun (celui de fusion_geometrique_m/donnees.py, produit par les scripts de preparation) :
    <racine>/meta.json                                  {"classes": K, "modalites": [m0, m1], "domaines": [...]}
    <racine>/<domaine>/{train,test}/<m>.npy             (N, d_m) vue faible
    <racine>/<domaine>/{train,test}/<m>_aug.npy         optionnel : (N, A, d_m) vues fortes
    <racine>/<domaine>/{train,test}/{labels,ids}.npy

Aucune ligne du code de reference n'est modifiee ; on ne remplace que l'ENTREE, a l'execution, comme l'adaptateur
HAC (hac/adapter.py) dont ce module reprend tout le reste a l'identique :
    slot "texte" <- modalite 0 (texte BERT pour le sentiment et l'esthetique, audio AST pour HAC), 768-d attendus
    slot "image" <- modalite 1 (image ResNet-50, ou video), 2048-d attendus
Une dimension plus grande que celle du slot est ramenee par une ACP FIXE ajustee sur les seules sources (vues faibles
et fortes de leur train, la cible exclue), comme pour SlowFast dans l'adaptateur HAC ; une dimension plus petite est
completee par des zeros (la couche lineaire suivante ignore alors ces entrees).

Chaque item porte toutes les cles qu'attendent get_batch, validate, validate_pseudo :
    input_ids  modalite 0, vue faible      input_ids_aug  modalite 0, une vue forte tiree au hasard
    image      modalite 1, vue faible      image_aug      modalite 1, une vue forte tiree au hasard
    label, id = "<domaine>/<split>/<ligne>" ; token_type_ids, attention_mask (+ _aug), image_mask : factices
"""
import copy
import json
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

SLOTS = {"texte": 768, "image": 2048}

STORE = None          # FeatureStore, installe par install()
RESULTS = None        # journal des evaluations M2CAN, ecrit dans results_path


class FeatureStore:
    """Caracteristiques gelees au format commun, indexees par "<domaine>/<split>/<ligne>"."""

    def __init__(self, root, target):
        with open(os.path.join(root, "meta.json")) as f:
            self.meta = json.load(f)
        self.root, self.target = root, target
        self.doms = list(self.meta["domaines"])
        assert target in self.doms, f"cible {target} absente de {self.doms}"
        m0, m1 = self.meta["modalites"][:2]
        self.arrays = {}
        for d in self.doms:
            for sp in ("train", "test"):
                p = os.path.join(root, d, sp)
                arr = {"labels": np.load(os.path.join(p, "labels.npy")).astype(np.int64)}
                for slot, m in (("texte", m0), ("image", m1)):
                    w = np.load(os.path.join(p, f"{m}.npy")).astype(np.float32)
                    pa = os.path.join(p, f"{m}_aug.npy")
                    a = np.load(pa).astype(np.float32) if os.path.exists(pa) else w[:, None, :]
                    if a.ndim == 2:
                        a = a[:, None, :]
                    arr[f"{slot}_faible"], arr[f"{slot}_forte"] = w, a
                self.arrays[(d, sp)] = arr
        for slot, dim in SLOTS.items():
            d_in = self.arrays[(self.doms[0], "train")][f"{slot}_faible"].shape[1]
            if d_in > dim:
                self._acp(slot, dim)
            elif d_in < dim:
                for arr in self.arrays.values():
                    for k in (f"{slot}_faible", f"{slot}_forte"):
                        pad = [(0, 0)] * (arr[k].ndim - 1) + [(0, dim - d_in)]
                        arr[k] = np.pad(arr[k], pad)
                print(f"[carac] slot {slot} : {d_in} -> {dim} par completion de zeros", flush=True)
        for arr in self.arrays.values():
            assert arr["texte_faible"].shape[1] == SLOTS["texte"] and arr["image_faible"].shape[1] == SLOTS["image"]

    def _acp(self, slot, dim):
        X = []
        for d in self.doms:
            if d == self.target:
                continue
            a = self.arrays[(d, "train")]
            X.append(a[f"{slot}_faible"].astype(np.float64))
            X.append(a[f"{slot}_forte"].reshape(-1, a[f"{slot}_forte"].shape[-1]).astype(np.float64))
        X = np.concatenate(X)
        mu = X.mean(axis=0)
        w, V = np.linalg.eigh(np.cov(X - mu, rowvar=False))
        w, V = w[::-1], V[:, ::-1][:, :dim]
        kept = float(w[:dim].sum() / w.sum())
        print(f"[carac] slot {slot} : {X.shape[1]} -> {dim} par ACP sur les sources ({X.shape[0]} vecteurs), "
              f"{kept * 100:.3f} % de variance conservee", flush=True)
        P, mu = V.astype(np.float32), mu.astype(np.float32)
        for arr in self.arrays.values():
            arr[f"{slot}_faible"] = (arr[f"{slot}_faible"] - mu) @ P
            arr[f"{slot}_forte"] = (arr[f"{slot}_forte"] - mu) @ P

    def ecrire_listes(self, dossier):
        """Listes au format des data_list de M2CAN ({"image": [id], "label": y}), une par domaine et split."""
        os.makedirs(dossier, exist_ok=True)
        chemins = {}
        for (d, sp), arr in self.arrays.items():
            lignes = [{"image": [f"{d}/{sp}/{i}"], "label": int(y)} for i, y in enumerate(arr["labels"])]
            p = os.path.join(dossier, f"{d}_{sp}.json")
            with open(p, "w") as f:
                json.dump(lignes, f)
            chemins[(d, sp)] = p
        return chemins

    def get(self, cid):
        d, sp, i = cid.split("/")
        a, i = self.arrays[(d, sp)], int(i)
        return a["texte_faible"][i], a["texte_forte"][i], a["image_faible"][i], a["image_forte"][i]


def _f32(x):
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


class HacDataset(Dataset):
    """Pendant de common_loader sur caracteristiques : memes cles, meme semantique de n_sup."""

    def __init__(self, data_list, seed=0, n_sup=None, unique=False):
        with open(data_list, "r") as rf:
            lines = json.load(rf)
        self.data_list = data_list
        self.ids = ['-'.join(line["image"]) for line in lines]
        self.labels = torch.tensor([int(line["label"]) for line in lines], dtype=int)
        if unique:
            self.effective_inds = list(range(len(self.ids)))
        else:
            self.get_effective_inds(seed, len(self.ids) if n_sup is None else n_sup)
        assert len(self.effective_inds) > 0
        logging.getLogger("global").info("# samples: {}".format(len(self.effective_inds)))

    def get_effective_inds(self, seed, n_sup):
        # copie de BaseVisionAndLanguageDataset.get_effective_inds : la liste est
        # repetee si elle compte moins de n_sup elements, puis on en tire n_sup.
        self.effective_inds = list(range(len(self.ids)))
        random.seed(seed)
        if len(self.effective_inds) < n_sup:
            num_repeat = math.ceil(n_sup / len(self.effective_inds))
            self.effective_inds = self.effective_inds * num_repeat
        self.effective_inds = random.sample(self.effective_inds, n_sup)

    def __len__(self):
        return len(self.effective_inds)

    def __getitem__(self, index):
        eff_index = self.effective_inds[index]
        cid = self.ids[eff_index]
        a_w, a_s, v_w, v_s = STORE.get(cid)
        dummy = torch.zeros(1, dtype=torch.long)
        return {
            "input_ids": _f32(a_w),
            "input_ids_aug": _f32(a_s[random.randrange(len(a_s))]),
            "token_type_ids": dummy, "attention_mask": dummy,
            "token_type_ids_aug": dummy, "attention_mask_aug": dummy,
            "image": _f32(v_w),
            "image_aug": _f32(v_s[random.randrange(len(v_s))]),
            "image_mask": torch.ones(1, dtype=torch.long),
            "label": self.labels[eff_index],
            "id": cid,
        }


# ---------------------------------------------------------------- chargeurs
class NoSingletonBatchSampler(BatchSampler):
    """Comme BatchSampler(drop_last=False), mais sans dernier lot de taille 1 :
    les deux derniers lots b+1 sont redecoupes en deux moities (4+1 -> 3+2).
    Meme nombre de lots, chaque exemple vu une fois."""

    def __init__(self, sampler, batch_size):
        super().__init__(sampler, batch_size, drop_last=False)

    def __iter__(self):
        idx, b = list(self.sampler), self.batch_size
        batches = [idx[i:i + b] for i in range(0, len(idx), b)]
        if len(batches) >= 2 and len(batches[-1]) == 1:
            tail = batches[-2] + batches[-1]
            cut = (len(tail) + 1) // 2
            batches[-2:] = [tail[:cut], tail[cut:]]
        return iter(batches)


class EvalLoader:
    """Chargeur d'evaluation (validate, validate_pseudo, evaluate_hotda).

    MsmmModel.forward_test fait `.squeeze()` sur la prediction : un dernier lot
    d'UN seul exemple perd sa dimension de lot et validate plante sur argmax(dim=1).
    Jamais rencontre avec les 1500 exemples de validation de Twitter/Yelp/TumEmo
    (multiple de 4) ; sur HAC un test de taille 4k+1 suffit. Les tetes n'ont pas
    de BatchNorm, le decoupage des lots ne change donc aucune prediction.
    Expose dataset et batch_size, que lit _regroupe de hotda_inloop."""

    def __init__(self, dataset, sampler, batch_size):
        self.dataset, self.batch_size = dataset, batch_size
        self._loader = DataLoader(dataset, batch_sampler=NoSingletonBatchSampler(sampler, batch_size),
                                  num_workers=0, pin_memory=False)

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        return iter(self._loader)


def _split_cfg(all_cfg, split, _type):
    cfg = copy.deepcopy(all_cfg["dataset"])
    cfg.update(cfg.get(split, {}))
    cfg.update(cfg.get(_type, {}))
    return cfg


def build_hac_data_loader(split, _type, all_cfg, tokenizer=None, seed=0):
    """Pendant de build_common_data_loader : RandomSampler, drop_last en train."""
    cfg = _split_cfg(all_cfg, split, _type)
    dset = HacDataset(cfg["data_list"], seed=seed, n_sup=cfg.get("n_sample", 1585))
    if split != "train":
        return EvalLoader(dset, RandomSampler(dset), cfg.get("batch_size", 1))
    return DataLoader(dset, batch_size=cfg.get("batch_size", 1), num_workers=cfg.get("workers", 0),
                      sampler=RandomSampler(dset), pin_memory=False, drop_last=True)


def hac_get_loader(cfg, source_combined=False, seed=0):
    """Pendant de data_loader.builder.get_loader, sans tokenizer BERT."""
    cfg_dataset = cfg["dataset"]
    train_loaders, val_loaders = [], []
    origin_batch_size = cfg["dataset"]["batch_size"]
    for _type in cfg_dataset["types"]:
        print(_type)
        if source_combined:
            if _type != cfg_dataset["types"][-1]:
                cfg["dataset"]["batch_size"] = origin_batch_size // 2
            else:
                cfg["dataset"]["batch_size"] = origin_batch_size
        train_loaders.append(build_hac_data_loader("train", _type, cfg, seed=seed))
        val_loaders.append(build_hac_data_loader("val", _type, cfg, seed=seed))
    return train_loaders, val_loaders


def hac_build_full_loader(all_cfg, _type, split, tokenizer, batch_size, seed):
    """Pendant de phase1_diff.build_full_loader (export des representations)."""
    cfg = _split_cfg(all_cfg, split, _type)
    with open(cfg["data_list"], "r") as f:
        n = len(json.load(f))
    dset = HacDataset(cfg["data_list"], seed=seed, n_sup=n)
    return DataLoader(dset, batch_size=batch_size, num_workers=0,
                      sampler=SequentialSampler(dset), pin_memory=False, drop_last=False)


def unique_view(loader):
    """validate_pseudo de M2CAN exige un passage SANS doublon sur la cible : il
    leve 'already exists' sinon. Sur Twitter c'etait garanti, la cible comptant
    plus d'exemples que n_sample ; sur HAC (|cible| < 15000) la cible est
    repetee pour remplir l'epoque. On lui passe donc la vue dedoublonnee du MEME
    fichier, chaque clip une fois : l'algorithme de pseudo-etiquetage est inchange."""
    uniq = HacDataset(loader.dataset.data_list, unique=True)
    return EvalLoader(uniq, SequentialSampler(uniq), loader.batch_size)


# ---------------------------------------------------------------- entree des modeles
def hac_prepare_text(bert_model, text_data, aug=False):
    """Pendant de prepare_text : le slot texte porte deja les features audio."""
    if aug:
        return text_data["input_ids"], text_data["input_ids_aug"]
    return text_data["input_ids"]


def hac_prepare_image(model_image, image_data, image_mask):
    """Pendant de prepare_image : la moyenne ResNet-50 sur les frames est deja faite."""
    return image_data


class FrozenBackbone(nn.Module):
    """Tient la place de BERT / ResNet-50, geles (features deja extraites).
    Le parametre factice garde les groupes d'optimiseur non vides ; il ne
    recoit jamais de gradient, Adam et SAM l'ignorent."""

    def __init__(self):
        super().__init__()
        self.frozen = nn.Parameter(torch.zeros(1), requires_grad=False)


class _BertStub:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return FrozenBackbone()


def _resnet50_stub(pretrained=False, **kwargs):
    return FrozenBackbone()


class _TokenizerStub:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return None


# ---------------------------------------------------------------- journal M2CAN
def _record_validate(orig, results_path):
    def validate(model, data_loaders, logger, epoch):
        acc = orig(model, data_loaders, logger, epoch)
        RESULTS["evals"].append({"epoch": epoch, "acc": acc})
        RESULTS["best"] = max(RESULTS["best"], acc)
        with open(results_path, "w") as f:
            json.dump(RESULTS, f, indent=2)
        return acc
    return validate


# ---------------------------------------------------------------- installation
def install(feats_root, method, results_path, target):
    """Charge les caracteristiques et branche l'entree. Renvoie le main() a lancer."""
    global STORE, RESULTS
    STORE = FeatureStore(feats_root, target)

    # config._merge laisse les chemins par defaut de config.py (DEFAULTS) ECRASER data_list / data_root des domaines
    # qu'il connait (TumEmo, Twitter, Yelp, AVA, PCCD, RPCD) : on retire ces deux cles de ignore_key, a l'execution,
    # pour que les listes de caracteristiques ecrites par ecrire_listes soient lues.
    import config as m2can_config
    m2can_config.ignore_key.difference_update({"data_list", "data_root"})

    import common_modules.utils as cu
    import msmm.models.model_msmm as mm
    for mod in (cu, mm):
        mod.prepare_text = hac_prepare_text
        mod.prepare_image = hac_prepare_image
    mm.BertModel = _BertStub
    mm.resnet50 = _resnet50_stub

    if method == "brasC":
        import hotda_inloop
        import phase1_diff
        for mod in (hotda_inloop, phase1_diff):
            mod.prepare_text = hac_prepare_text
            mod.prepare_image = hac_prepare_image
        phase1_diff.get_loader = hac_get_loader
        phase1_diff.build_full_loader = hac_build_full_loader
        phase1_diff.BertTokenizer = _TokenizerStub
        return phase1_diff.main

    if method == "m2can":
        import msmm.train as mt
        mt.get_loader = hac_get_loader
        orig_vp = mm.MsmmModel.validate_pseudo

        def validate_pseudo(self, target_dataloader, pseudo_rate):
            return orig_vp(self, unique_view(target_dataloader), pseudo_rate)

        mm.MsmmModel.validate_pseudo = validate_pseudo
        RESULTS = {"method": "m2can", "evals": [], "best": 0.0}
        cu.validate = _record_validate(cu.validate, results_path)
        return mt.main

    raise ValueError(method)
