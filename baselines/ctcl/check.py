from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ctcl_conf as C

METRIC_KEYS = ("composite_vs_train", "composite_vs_test", "vs_train", "vs_test", "fidelity", "metrics", "bench")


def _num(side: dict, key: str, problems: list, positive: bool = True):
    v = side.get(key)
    if v is None:
        problems.append(f"missing {key}")
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        problems.append(f"{key} is not a number: {v!r}")
        return None
    if not math.isfinite(v):
        problems.append(f"{key} is not finite: {v}")
        return None
    if positive and v <= 0:
        problems.append(f"{key} must be > 0: {v}")
        return None
    return v


def prv_subprocess(sigma, q, T, gauss) -> dict:
    cmd = [C.DPLORA_PY, "-I", "-B", str(HERE / "prv_ctcl.py"), "--sigma", repr(sigma), "--q", repr(q), "--steps",
           str(int(T)), "--delta", repr(C.DELTA)] + [a for s, c in gauss for a in ("--gauss", f"{s!r}:{int(c)}")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"prv_ctcl.py failed: {r.stderr[-500:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def check_sidecar(side: dict, prv=prv_subprocess, method: str | None = None) -> dict:
    import calibrate as CB
    P, res = [], {}
    method = method or side.get("method")
    if method not in C.METHODS:
        return {"problems": [f"unknown method {method!r}"], "ok": False}
    variant, config, param = C.METHODS[method]
    cfg = C.CONFIGS[config]
    for k in ("variant", "config", "param"):
        if side.get(k) != {"variant": variant, "config": config, "param": param}[k]:
            P.append(f"{k} {side.get(k)!r} != {method}'s")
    eps = _num(side, "eps_target", P)
    if eps is not None and side.get("eps") not in (None, eps):
        P.append(f"eps {side.get('eps')} != eps_target {eps}")
    sigma = _num(side, "noise_multiplier", P)
    q = _num(side, "sample_rate", P)
    T = _num(side, "steps", P)
    clip = _num(side, "clip", P)
    b_nom = _num(side, "b_nom", P)
    delta = _num(side, "delta", P)
    s_h = _num(side, "sigma_topic_hist", P)
    n_h = _num(side, "n_topic_hist", P)
    K = _num(side, "K_topics", P)
    want = {"sample_rate": (q, cfg["batch"] / C.N_REF), "steps": (T, cfg["steps"]), "clip": (clip, cfg["clip"]),
            "b_nom": (b_nom, cfg["batch"]), "delta": (delta, C.DELTA), "sigma_topic_hist": (s_h, C.SIGMA_TOPIC_HIST),
            "n_topic_hist": (n_h, C.N_TOPIC_HIST), "K_topics": (K, C.K_TOPICS)}
    for k, (got, exp) in want.items():
        if got is not None and got != exp:
            P.append(f"{k} {got} != frozen {exp}")
    s_prior, n_prior = None, 0
    if variant == "vp":
        s_prior = _num(side, "sigma_prior", P)
        n_prior = _num(side, "n_prior", P)
        if eps is not None and s_prior is not None:
            exp = CB.sigma_prior_for(eps, C.DELTA)
            if abs(s_prior - exp) > 1e-9:
                P.append(f"sigma_prior {s_prior} != sqrt(2 / (0.1 mu(eps)^2)) = {exp}")
        if n_prior is not None and n_prior != C.N_PRIOR:
            P.append(f"n_prior {n_prior} != {C.N_PRIOR}")
    else:
        if side.get("sigma_prior") is not None or side.get("n_prior") not in (0, None):
            P.append("a plain-variant sidecar carries prior noise")
        if "n_prior" not in side:
            P.append("missing n_prior")
    s_eq = _num(side, "sigma_eq", P)
    if s_eq is not None:
        exp = CB.sigma_eq(s_prior if variant == "vp" else None)
        if abs(s_eq - exp) > 1e-9:
            P.append(f"sigma_eq {s_eq} != {exp}")
    vpc = side.get("vp_calibration") or {}
    if s_eq is not None and (vpc.get("sigma_hist") != s_eq or vpc.get("n_hist") != 1):
        P.append(f"vp_calibration {vpc} != {{sigma_hist: sigma_eq, n_hist: 1}}")
    if side.get("complete") is not True:
        P.append("training incomplete")
    if T is not None and side.get("steps_run") != int(T):
        P.append(f"steps_run {side.get('steps_run')} != T {T}")
    if (side.get("private_stats") or {}).get("oom_retries", 0):
        P.append("OOM retry recorded")
    for k in METRIC_KEYS:
        if k in side:
            P.append(f"the sidecar holds a metric ({k})")
    if eps is not None and sigma is not None and float(eps).is_integer():
        ref = C.SIGMA_REF.get((config, variant), {}).get(int(eps))
        if ref is not None and abs(sigma - ref) > C.SIGMA_REF_TOL:
            P.append(f"sigma_SGD {sigma} differs from the spec's reference {ref} by more than {C.SIGMA_REF_TOL}")
    if not P:
        e = CB.composed_eps(sigma, q, int(T), s_h, int(n_h), s_prior, int(n_prior or 0), delta)
        e_cal = _num(side, "eps_composed_pld", P)
        res["eps_pld_recomposed"] = e
        if not math.isfinite(e):
            P.append(f"composed PLD epsilon is not finite: {e}")
        elif e > eps:
            P.append(f"composed PLD epsilon {e} > eps {eps}")
        if e_cal is not None and math.isfinite(e) and abs(e - e_cal) > 1e-6:
            P.append(f"recomposed PLD epsilon {e} != the calibration's {e_cal}")
        gauss = [(s_h, n_h)] + ([(s_prior, n_prior)] if variant == "vp" else [])
        pr = prv(sigma, q, int(T), gauss)
        res["eps_prv_recomputed"], res["eps_rdp_recomputed"] = pr["eps_prv"], pr["eps_rdp"]
        if not (math.isfinite(pr["eps_prv"]) and pr["eps_prv"] <= eps):
            P.append(f"Opacus PRV epsilon {pr['eps_prv']} is not finite or > eps {eps}")
        rec = side.get("eps_prv")
        if rec is None or abs(float(rec) - pr["eps_prv"]) > 1e-6:
            P.append(f"recorded eps_prv {rec} != recomputed {pr['eps_prv']}")
    res["problems"] = P
    res["ok"] = not P
    return res


def main():
    rows = []
    for f in sys.argv[1:]:
        side = json.loads(Path(f).read_text())
        r = check_sidecar(side)
        r["file"] = f
        rows.append(r)
        print(json.dumps(r, default=str), flush=True)
    sys.exit(0 if rows and all(r["ok"] for r in rows) else 1)


if __name__ == "__main__":
    main()
