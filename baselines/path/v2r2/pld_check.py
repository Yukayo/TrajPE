from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pathfmt as PF

sys.path.insert(0, str(PF.PROJECT))
from trajpe.dp import mu_for

PRIVATE_KEYS = {"private_stats", "log", "train_sec", "train_total_sec", "train_loop_sec", "n_private_trails",
                "n_private_rows", "embed_private_sec", "knn_sec", "vote_sec", "raw_votes", "tokenize_sec",
                "response_tokens", "losses"}
ROW_MODEL = {"path_v2r2": "0.5b", "path_7b_v2r2": "7b"}


def accountants(sigma_sgd: float, q: float, T: int, sigma_sel, delta: float) -> dict:
    import dp_accounting
    from dp_accounting.pld import pld_privacy_accountant
    from dp_accounting.rdp import rdp_privacy_accountant
    pld = pld_privacy_accountant.PLDAccountant(
        neighboring_relation=dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE, value_discretization_interval=1e-4)
    rdp = rdp_privacy_accountant.RdpAccountant(neighboring_relation=dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE)
    if math.isfinite(sigma_sgd):
        ev = dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(sigma_sgd))
        pld.compose(ev, int(T))
        rdp.compose(ev, int(T))
    if sigma_sel is not None and math.isfinite(sigma_sel):
        pld.compose(dp_accounting.GaussianDpEvent(sigma_sel), 1)
        rdp.compose(dp_accounting.GaussianDpEvent(sigma_sel), 1)
    return {"pld": float(pld.get_epsilon(delta)), "rdp": float(rdp.get_epsilon(delta))}


def all_keys(o) -> set:
    if isinstance(o, dict):
        return set(o) | set().union(*(all_keys(v) for v in o.values()))
    if isinstance(o, (list, tuple)):
        return set().union(*(all_keys(v) for v in o)) if o else set()
    return set()


def check(f: Path) -> dict:
    s = json.loads(f.read_text())
    P = []
    a = s.get("accounting") or {}
    res = {"file": str(f), "method": s.get("method"), "row": s.get("row"), "eps": s.get("eps"), "problems": P}
    if s.get("method") != "path" or s.get("row") not in ROW_MODEL:
        P.append("not a PATH release sidecar")
        res["ok"] = False
        return res
    mk = ROW_MODEL[s["row"]]
    eps, delta = float(a["eps_total"]), float(a["delta"])
    if eps != float(s["eps"]) or delta != PF.DELTA:
        P.append("eps_total / delta differ from the release")
    sg, sel = a["dp_sgd"], a["selection"]
    q, T, sig = float(sg["q"]), int(sg["T"]), float(sg["sigma_sgd"])
    if abs(q - PF.sample_rate()) > 1e-15 or T != PF.n_steps() or float(sg["clip"]) != PF.CLIP or \
            int(sg["b_nom"]) != PF.BATCH:
        P.append("DP-SGD constants differ from SPEC section 3")
    k = int(sel["k"])
    if k != PF.VOTE_K or abs(float(sel["l2_sensitivity"]) - math.sqrt(k)) > 1e-12:
        P.append("selection k / sensitivity differ")
    unit = float(sel["noise_std"]) / float(sel["l2_sensitivity"])
    es = PF.eps_select(eps)
    sigma_rule = 1.0 / mu_for(es, delta / 2)
    if abs(float(sel["eps_select"]) - es) > 1e-12 or abs(float(sel["delta_select"]) - delta / 2) > 0:
        P.append("eps_select / delta_select differ from the A11 rule")
    if abs(float(sel["sigma_unit"]) - sigma_rule) > 1e-9 * sigma_rule or abs(unit - sigma_rule) > 1e-9 * sigma_rule:
        P.append(f"selection noise {sel['sigma_unit']} / {unit} differs from the calibration's {sigma_rule}")
    joint = accountants(sig, q, T, unit, delta)
    alone_sgd = accountants(sig, q, T, None, delta)
    alone_sel = accountants(float("inf"), q, T, unit, delta)
    res.update({"eps_joint_pld_recomputed": joint["pld"], "eps_joint_rdp_dp_accounting": joint["rdp"],
                "eps_joint_pld_recorded": a.get("eps_joint_pld"), "eps_joint_prv_recorded": a.get("eps_joint_prv"),
                "eps_sgd_only_pld": alone_sgd["pld"], "eps_sel_only_pld": alone_sel["pld"],
                "sigma_sgd": sig, "sigma_sel": unit})
    if not joint["pld"] <= eps:
        P.append(f"joint PLD epsilon {joint['pld']:.5f} exceeds {eps}")
    if abs(joint["pld"] - float(a["eps_joint_pld"])) > 1e-6:
        P.append(f"recorded eps_joint_pld {a['eps_joint_pld']} != recomputed {joint['pld']}")
    if not float(a["eps_joint_prv"]) <= eps:
        P.append(f"recorded Opacus PRV epsilon {a['eps_joint_prv']} exceeds {eps}")
    if not joint["rdp"] >= joint["pld"] - 1e-6:
        P.append("RDP below PLD (accountant inconsistency)")
    lit = a.get("paper_literal") or {}
    basic = accountants(sig, q, T, None, delta / 2)["pld"] + es
    if abs(basic - float(lit.get("eps_basic_reading_of_release", -1))) > 1e-6:
        P.append("paper_literal basic reading differs from its recomputation")
    if s.get("settings") != json.loads(json.dumps(PF.settings(mk))):
        P.append("settings differ from SPEC section 3" + (" / A17" if mk == "7b" else "") + " / appendix R2")
    gen = (s.get("sampling") or {}).get("generation") or {}
    if s.get("version") != PF.VERSION or gen.get("decode") != PF.DECODE or gen.get("coord_decimals") != PF.DECIMALS:
        P.append(f"not a v2r2 release (version {s.get('version')}, decode {gen.get('decode')}, "
                 f"decimals {gen.get('coord_decimals')}; SPEC appendix R2)")
    tr = s.get("training") or {}
    want = {"model": PF.MODELS[mk]["id"], "model_revision": PF.MODELS[mk]["revision"], "noise_multiplier": sig,
            "sample_rate": q, "steps": T, "clip": PF.CLIP, "lr": PF.LR, "lora_rank": PF.RANK, "lora_alpha": PF.ALPHA,
            "lora_dropout": PF.DROPOUT, "lora_targets": PF.TARGETS, "trainable_params": PF.MODELS[mk]["trainable_params"],
            "epochs": PF.EPOCHS, "b_nom": PF.BATCH, "batch_ref": PF.BATCH, "complete": True, "steps_run": T}
    for kk, v in want.items():
        if tr.get(kk) != v:
            P.append(f"training.{kk} = {tr.get(kk)!r}, expected {v!r}")
    if int(s.get("n_trails", -1)) != PF.N_OUT or int(s.get("n_out", -1)) != PF.N_OUT:
        P.append("n_trails / n_out != 50,000")
    if (s.get("sampling") or {}).get("m") != PF.POOL_M:
        P.append("pool size != M")
    leak = PRIVATE_KEYS & all_keys(s)
    if leak:
        P.append(f"work-dir-only fields in the release sidecar: {sorted(leak)}")
    work = Path(s.get("work", ""))
    if work.is_dir() and (work / "train.json").exists():
        tj = json.loads((work / "train.json").read_text())
        ps = tj.get("private_stats") or {}
        if ps.get("oom_retries", 0) != 0 or tj.get("complete") is not True:
            P.append(f"training release gate (oom_retries={ps.get('oom_retries')}, complete={tj.get('complete')})")
        cal = json.loads((work / "calib.json").read_text())
        if (cal["sigma_sgd"], cal["sigma_sel"], cal["sample_rate"], cal["steps"], cal["eps_total"]) != \
                (sig, float(sel["sigma_unit"]), q, T, eps):
            P.append("calib.json differs from the sidecar")
        res["work_checked"] = True
    res["ok"] = not P
    return res


def main():
    rows = [check(Path(f)) for f in sys.argv[1:] if f.endswith(".json") and not f.endswith(".eval.json")]
    for r in rows:
        print(json.dumps(r))
    bad = [r for r in rows if not r["ok"]]
    for r in bad:
        print("FAIL", r["file"], r["problems"])
    sys.exit(3 if bad or not rows else 0)


if __name__ == "__main__":
    main()
