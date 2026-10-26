"""Outils de transport optimal : coût cosinus, Sinkhorn en domaine log, barycentre de Wasserstein à support libre.

Reprise à l'identique de wbhr.py (fusion_geometrique_code), sans dépendance au code M2CAN.
"""
import torch
import torch.nn.functional as F


def cosine_cost(X, Y):
    """C[i, j] = ||x_i/|x_i| - y_j/|y_j|||^2 = 2 - 2 cos(x_i, y_j), dans [0, 4]."""
    Xn = F.normalize(X, dim=-1)
    Yn = F.normalize(Y, dim=-1)
    return (2.0 - 2.0 * (Xn @ Yn.t())).clamp_min(0.0)


def sinkhorn_log(C, a, b, eps, n_iter):
    """Sinkhorn en domaine log (stable). Renvoie le plan P (|a|, |b|)."""
    log_a, log_b = torch.log(a), torch.log(b)
    f, g = torch.zeros_like(a), torch.zeros_like(b)
    for _ in range(n_iter):
        f = eps * (log_a - torch.logsumexp((-C + g.unsqueeze(0)) / eps, dim=1))
        g = eps * (log_b - torch.logsumexp((-C + f.unsqueeze(1)) / eps, dim=0))
    return torch.exp((-C + f.unsqueeze(1) + g.unsqueeze(0)) / eps)


def free_support_barycenter(clouds, support=16, eps=0.05, sinkhorn_iters=20, outer_iters=3, init=None):
    """Barycentre de Wasserstein à support libre de N mesures empiriques uniformes (point fixe barycentrique).

    Renvoie (B, plans, couts).
    """
    device, dtype = clouds[0].device, clouds[0].dtype
    if init is not None:
        B = init.detach().clone().to(device=device, dtype=dtype)
    else:
        pool = torch.cat([c.detach() for c in clouds], dim=0)
        B = pool[torch.randint(0, pool.shape[0], (support,), device=device)].clone()
    a = torch.full((support,), 1.0 / support, device=device, dtype=dtype)
    weights = [torch.full((c.shape[0],), 1.0 / c.shape[0], device=device, dtype=dtype) for c in clouds]
    plans, costs = [], []
    for t in range(outer_iters):
        plans, costs = [], []
        for c, b in zip(clouds, weights):
            C = cosine_cost(B, c)
            plans.append(sinkhorn_log(C, a, b, eps, sinkhorn_iters))
            costs.append(C)
        if t < outer_iters - 1:
            B = sum(P @ c for P, c in zip(plans, clouds)) / (len(clouds) * a.unsqueeze(1))
    return B, plans, costs


def plan_transport(B, X, eps, iters):
    """Plan de Sinkhorn P (|B|, |X|) entre deux nuages uniformes, coût cosinus. Renvoie (P, C)."""
    C = cosine_cost(B, X)
    a = torch.full((B.shape[0],), 1.0 / B.shape[0], device=B.device, dtype=B.dtype)
    b = torch.full((X.shape[0],), 1.0 / X.shape[0], device=B.device, dtype=B.dtype)
    return sinkhorn_log(C, a, b, eps, iters), C


def cout_transport(B, X, eps, iters):
    """Coût de Sinkhorn <P, C> entre deux nuages uniformes, coût cosinus."""
    P, C = plan_transport(B, X, eps, iters)
    return float((P * C).sum())


def projection_barycentrique(B, X, P):
    """Transport des points de B vers X par le plan P (marges uniformes sur B) : B~ = |B| P X."""
    return B.shape[0] * (P @ X)
