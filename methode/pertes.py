"""Fonctions de coût de la phase 1 (voir docs/fusion_geometrique.md, section 4).

  L_geo = poids_tetes * L_tetes + alpha * L_CMCFA + beta * L_CDCFA + gamma * L_SC_fusion + poids_im * L_IM

L_IM (option, 0 par défaut) : maximisation de l'information de SHOT (Liang et al., 2020) sur les têtes
unimodales du lot cible, sans étiquette : entropie moyenne des prédictions moins entropie de la
prédiction moyenne, moyennée sur les M têtes.

Cohérence entre modalités dans CMCFA :
  - M = 2 : V(x) = KL(p_1 || p_0), exactement comme M2CAN (KLDivLoss(log p_0, p_1)) ;
  - M >= 3 : V(x) = JSD(p_0, ..., p_{M-1}) = H(moyenne des p_m) - moyenne des H(p_m).
  Poids w(x) = exp(-V(x)).
"""
import itertools

import torch
import torch.nn.functional as F


def supcon(z, labels, tau=0.07):
    """SupCon (Khosla et al.), à l'identique de cdcfa.py. z (B, d) tel quel, labels (B,)."""
    B = z.shape[0]
    if B < 2:
        return torch.zeros((), device=z.device)
    logits = (z @ z.t()) / tau
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    n_pos = pos.sum(dim=1)
    valid = n_pos > 0
    if not bool(valid.any()):
        return torch.zeros((), device=z.device)
    return -((pos.float() * log_prob).sum(dim=1)[valid] / n_pos[valid].float()).mean()


def info_max(logits):
    """Perte de maximisation de l'information d'UNE tête : moyenne_i H(p_i) - H(moyenne_i p_i)."""
    p = F.softmax(logits, 1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1).mean()
    pm = p.mean(0)
    return ent + (pm * torch.log(pm.clamp_min(1e-12))).sum()


def coherence(logits):
    """V(x) pour chaque exemple : KL si 2 modalités, JSD au-delà. logits : liste de M tenseurs (B, K)."""
    if len(logits) == 2:
        # KLDivLoss(log_softmax(l0), softmax(l1)) = KL(p1 || p0), comme CriterionMultiModal de M2CAN
        return F.kl_div(F.log_softmax(logits[0], 1), F.softmax(logits[1], 1), reduction="none").sum(1)
    P = torch.stack([F.softmax(l, 1) for l in logits])                    # (M, B, K)
    ent = lambda p: -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
    return ent(P.mean(0)) - ent(P).mean(0)


def infonce_paire(a, a_aug, b, b_aug):
    """InfoNCE de M2CAN entre deux modalités, quatre combinaisons de vues, sans température."""
    pos = (torch.exp((a * b).sum(-1)) + torch.exp((a * b_aug).sum(-1))
           + torch.exp((a_aug * b).sum(-1)) + torch.exp((a_aug * b_aug).sum(-1)))
    den = (torch.exp(a @ b.t()) + torch.exp(a @ b_aug.t())
           + torch.exp(a_aug @ b.t()) + torch.exp(a_aug @ b_aug.t())).sum(-1)
    return -torch.log(pos / den)


def cmcfa(feats, feats_aug, logits):
    """CMCFA pondéré pour UN domaine. feats, feats_aug : listes de M tenseurs (B, 256) ; logits : M x (B, K).

    M = 2 : identique à M2CAN (la modalité 0 joue le rôle du texte, la 1 celui de l'image).
    M >= 3 : InfoNCE moyenné sur toutes les paires de modalités.
    """
    paires = list(itertools.combinations(range(len(feats)), 2))
    l = sum(infonce_paire(feats[i], feats_aug[i], feats[j], feats_aug[j]) for i, j in paires) / len(paires)
    V = coherence(logits)
    return (l * torch.exp(-V)).mean() + V.mean()


def pertes_phase1(modele, lots, tau, pseudo=None, normaliser_z=False, cmcfa_cible=True):
    """Calcule les quatre termes sur un pas.

    lots : liste de dictionnaires par domaine, sources d'abord, cible en dernier, chacun avec
           "x" ({modalité: tenseur}), "x_aug" ({modalité: tenseur}), "y" (sources), "ids" (cible).
    pseudo : dictionnaire {id cible: classe} ou None ; les points cibles du lot qui y figurent rejoignent
             les sources dans le SupCon sur la fusion.
    normaliser_z : SupCon de la fusion sur z normalisé L2 (option phase1.normaliser_z), la géométrie que lit la
             phase 2 ; par défaut z entre tel quel.
    cmcfa_cible : False = CMCFA sur les sources seulement (la cible n'entre dans aucune perte, comme l'époque 0 de
             M2CAN) ; True (défaut) = sur tous les domaines.
    Renvoie (tetes, cmcfa, cdcfa, sc_fusion, im) ; im porte sur les têtes du lot cible, sans étiquette.
    """
    n_src = len(lots) - 1
    feats, feats_aug, logits = [], [], []
    for lot in lots:
        f, fa = modele.encoder(lot["x"]), modele.encoder(lot["x_aug"])
        feats.append(f); feats_aug.append(fa); logits.append(modele.logits_tetes(f))

    l_cmcfa = sum(cmcfa(feats[d], feats_aug[d], logits[d]) for d in range(len(lots) if cmcfa_cible else n_src))

    ys = [lots[d]["y"] for d in range(n_src)]
    l_tetes = sum(F.cross_entropy(logits[d][m], ys[d]) for d in range(n_src) for m in range(modele.M)) \
        / (modele.M * n_src)

    y_src = torch.cat(ys)
    l_cdcfa = sum(supcon(torch.cat([feats[d][m] for d in range(n_src)]), y_src, tau) for m in range(modele.M))

    z = [modele.fusionner(feats[d]) for d in range(n_src)]
    y_fus = [y_src]
    if pseudo:
        ids = lots[-1]["ids"]
        sel = [k for k, i in enumerate(ids) if i in pseudo]
        if sel:
            z.append(modele.fusionner(feats[-1])[sel])
            y_fus.append(torch.tensor([pseudo[ids[k]] for k in sel], device=y_src.device, dtype=y_src.dtype))
    z = torch.cat(z)
    l_sc = supcon(F.normalize(z, dim=-1) if normaliser_z else z, torch.cat(y_fus), tau)
    l_im = sum(info_max(l) for l in logits[-1]) / modele.M
    return l_tetes, l_cmcfa, l_cdcfa, l_sc, l_im
