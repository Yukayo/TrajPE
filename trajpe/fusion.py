from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

SIGMA_FLOOR = 1.0


def vote_prior(noisy_votes: np.ndarray, rho: float) -> np.ndarray:
    p = np.maximum(noisy_votes, 0.0)
    p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1.0 / len(p))
    if rho > 0:
        p = (1.0 - rho) * p + rho / len(p)
    return p


def fuse(fams, N: float, p: np.ndarray, gamma: float = 1.0, max_iter: int = 300, log=None):
    supp = np.flatnonzero(p > 0)
    ps = p[supp] / p[supp].sum()
    logp = np.log(ps)
    mats = []
    for f in fams:
        A = f.Phi[supp].T.tocsr()
        c = N / np.maximum(np.asarray(f.sigma, np.float64), SIGMA_FLOOR) ** 2
        mats.append((f.name, A, A.T.tocsr(), np.asarray(f.y, np.float64) / N, c))

    def obj(z):
        z = z - z.max()
        e = np.exp(z)
        w = e / e.sum()
        val, g = 0.0, np.zeros_like(w)
        for _, A, AT, y, c in mats:
            r = A @ w - y
            val += 0.5 * float((c * r) @ r)
            g += AT @ (c * r)
        if gamma > 0:
            lw = np.log(np.maximum(w, 1e-300))
            val += gamma * float(w @ (lw - logp))
            g += gamma * (lw - logp + 1.0)
        return val, w * (g - float(w @ g))

    z0 = logp.copy()
    res = minimize(obj, z0, jac=True, method="L-BFGS-B", options={"maxiter": max_iter, "maxcor": 20})
    z = res.x - res.x.max()
    ws = np.exp(z)
    ws /= ws.sum()
    w = np.zeros(len(p))
    w[supp] = ws
    info = {"iters": int(res.nit), "obj0": float(obj(z0)[0]), "obj": float(res.fun),
            "ess_prior": float(1.0 / (ps ** 2).sum()), "ess": float(1.0 / (ws ** 2).sum()),
            "kl": float(ws @ (np.log(np.maximum(ws, 1e-300)) - logp))}
    for name, A, _, y, c in mats:
        info[f"res_{name}"] = [round(float(np.abs(A @ ps - y).sum()), 4), round(float(np.abs(A @ ws - y).sum()), 4)]
    if log:
        log({"fusion": info})
    return w, info


def systematic(w: np.ndarray, n: int, rng) -> np.ndarray:
    c = np.cumsum(w)
    c /= c[-1]
    u = (rng.random() + np.arange(n)) / n
    idx = np.minimum(np.searchsorted(c, u, side="right"), len(w) - 1)
    return rng.permutation(idx)
