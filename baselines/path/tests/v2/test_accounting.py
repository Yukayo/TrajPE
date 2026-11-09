import os as _tp_os
def _tp_root():
    p = _tp_os.path.dirname(_tp_os.path.abspath(__file__))
    while not _tp_os.path.isfile(_tp_os.path.join(p, "trajpe", "__init__.py")) and _tp_os.path.dirname(p) != p:
        p = _tp_os.path.dirname(p)
    return p
_TP_ROOT = _tp_os.environ.get("TRAJPE_ROOT", _tp_root())
_TP_WORK = _tp_os.environ.get("TRAJPE_WORK", _tp_os.path.join(_TP_ROOT, "work"))
_TP_PY = _tp_os.environ.get("TRAJPE_PYTHON", "python")
_TP_PY_DPLORA = _tp_os.environ.get("TRAJPE_PYTHON_DPLORA", _TP_PY)
_TP_PY_AIMSEQ = _tp_os.environ.get("TRAJPE_PYTHON_AIMSEQ", _TP_PY)
_TP_PY_GRID = _tp_os.environ.get("TRAJPE_PYTHON_GRID", _TP_PY)
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = Path(__file__).resolve().parents[2] / "v2"
sys.path.insert(0, str(HERE))

import calibrate as CB
import pathfmt as PF
import pld_check as PC
from trajpe.dp import eps_for, gdp_delta

PREVIEW = {1.0: (16.010, 0.8846, 0.9849, 0.9948), 2.0: (8.348, 0.7370, 1.9845, 1.9952),
           4.0: (4.365, 0.6059, 3.9845, 3.9957), 8.0: (4.365, 0.4874, 7.9824, 7.9945)}
LITERAL = {1.0: (0.9794, 0.731, 1.289), 2.0: (0.8067, 1.424, 2.598), 4.0: (0.6678, 2.910, 5.120),
           8.0: (0.5144, 6.704, 9.327)}
CALS = {}


def cal(eps):
    if eps not in CALS:
        CALS[eps] = CB.calibrate(eps)
    return CALS[eps]


def test_eps_select_rule():
    assert [PF.eps_select(e) for e in (0.5, 2.0, 4.0, 10.0)] == [0.25, 0.5, 1.0, 1.0]
    assert [PF.eps_select(e) for e in (1.0, 2.0, 4.0, 8.0)] == [0.25, 0.5, 1.0, 1.0]
    assert PF.sample_rate() == 256 / 74_685 and PF.n_steps() == 875


def test_calibration_all_eps():
    for eps, (ss, s, pld, _) in PREVIEW.items():
        c = cal(eps)
        assert abs(c["sigma_sel"] - ss) < 5e-4 and abs(c["sigma_sgd"] - s) < 2e-4, (eps, c["sigma_sel"], c["sigma_sgd"])
        assert c["eps_composed_pld"] <= eps - PF.PRV_MARGIN + 1e-12 and c["eps_composed_pld"] <= eps
        assert abs(c["eps_composed_pld"] - pld) < 5e-4
        assert eps - PF.PRV_MARGIN - c["eps_composed_pld"] < PF.CALIB_TOL
        tight = CB.hybrid.composed_eps(c["sigma_sgd"] - 2e-4, c["sample_rate"], c["steps"], c["mu_sel"], c["delta"])
        assert tight > eps - PF.PRV_MARGIN
        assert abs(eps_for(1.0 / c["sigma_sel"], PF.DELTA / 2) - c["eps_select"]) < 1e-9
        assert abs(gdp_delta(1.0 / c["sigma_sel"], c["eps_select"]) - PF.DELTA / 2) < 1e-15
        assert c["sigma_hist"] == c["sigma_sel"] and c["n_hist"] == 1
        lit = c["paper_literal"]
        sb, jb, br = LITERAL[eps]
        assert abs(lit["sigma_sgd_basic"] - sb) < 2e-4 and abs(lit["eps_joint_pld_of_sigma_sgd_basic"] - jb) < 2e-3
        assert abs(lit["eps_basic_reading_of_release"] - br) < 2e-3
        assert lit["eps_joint_pld_of_sigma_sgd_basic"] < c["eps_composed_pld"] and lit["sigma_sgd_basic"] > c["sigma_sgd"]
        print(f"   eps={eps:g}: eps_sel={c['eps_select']} sigma_sel={c['sigma_sel']:.4f} sigma_SGD={c['sigma_sgd']:.4f}"
              f" joint PLD={c['eps_composed_pld']:.4f}", flush=True)


def test_second_accountants():
    rows = []
    for eps in PREVIEW:
        c = cal(eps)
        a = PC.accountants(c["sigma_sgd"], c["sample_rate"], c["steps"], c["sigma_sel"], PF.DELTA)
        assert abs(a["pld"] - c["eps_composed_pld"]) < 1e-9 and a["pld"] <= eps
        assert a["rdp"] >= a["pld"]
        k = PF.VOTE_K
        unit = (math.sqrt(k) * c["sigma_sel"]) / math.sqrt(k)
        assert abs(unit - c["sigma_sel"]) < 1e-12
        rows.append([c["sigma_sgd"], c["sample_rate"], c["steps"], c["sigma_sel"], eps])
    code = ("import json,sys; sys.path.insert(0, %r); import accounting as A; out=[]\n"
            "for s,q,T,ss,e in json.loads(sys.argv[1]): out.append(A.epsilons_composed(s,q,T,ss,1,1e-6))\n"
            "print(json.dumps(out))") % str(PF.PROJECT / "baselines" / "dplora")
    r = subprocess.run([f"{_TP_PY_DPLORA}", "-I", "-B", "-c", code, json.dumps(rows)],
                       capture_output=True, text=True, cwd="/tmp", env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stderr[-2000:]
    prv = json.loads(r.stdout.strip().splitlines()[-1])
    for (s, q, T, ss, eps), p in zip(rows, prv):
        assert p["eps_prv"] <= eps and abs(p["eps_prv"] - PREVIEW[eps][3]) < 2e-3, (eps, p)
        assert p["eps_rdp"] >= p["eps_prv"]
        print(f"   eps={eps:g}: Opacus PRV {p['eps_prv']:.4f}, RDP {p['eps_rdp']:.4f}", flush=True)


def test_finer_pld_third_route():
    import dp_accounting
    from dp_accounting.pld import pld_privacy_accountant
    for eps in PREVIEW:
        c = cal(eps)
        acc = pld_privacy_accountant.PLDAccountant(
            neighboring_relation=dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE,
            value_discretization_interval=1e-5)
        acc.compose(dp_accounting.PoissonSampledDpEvent(c["sample_rate"], dp_accounting.GaussianDpEvent(c["sigma_sgd"])),
                    int(c["steps"]))
        acc.compose(dp_accounting.GaussianDpEvent(c["sigma_sel"]), 1)
        e = float(acc.get_epsilon(PF.DELTA))
        assert e <= eps and abs(e - c["eps_composed_pld"]) < 2e-3, (eps, e, c["eps_composed_pld"])
        print(f"   eps={eps:g}: PLD at discretization 1e-5: {e:.4f} (1e-4: {c['eps_composed_pld']:.4f})", flush=True)


def fake_sidecar(eps: float, mk: str = "0.5b") -> dict:
    import release as RL
    c = cal(eps)
    tj = {"eps_prv": PREVIEW[eps][3], "eps_rdp": 9.9, "eps_sgd_part_prv": 1.0, "eps_sgd_part_rdp": 1.0}
    m = PF.MODELS[mk]
    tr = {"model": m["id"], "model_revision": m["revision"], "noise_multiplier": c["sigma_sgd"],
          "sample_rate": c["sample_rate"], "steps": c["steps"], "clip": PF.CLIP, "lr": PF.LR, "lora_rank": PF.RANK,
          "lora_alpha": PF.ALPHA, "lora_dropout": PF.DROPOUT, "lora_targets": PF.TARGETS,
          "trainable_params": m["trainable_params"], "epochs": PF.EPOCHS, "b_nom": PF.BATCH, "batch_ref": PF.BATCH,
          "complete": True, "steps_run": c["steps"]}
    return {"method": "path", "version": PF.VERSION, "row": m["row"], "eps": eps, "n_trails": PF.N_OUT,
            "n_out": PF.N_OUT,
            "settings": PF.settings(mk), "accounting": RL.accounting_block(c, tj), "training": tr,
            "sampling": {"m": PF.POOL_M}, "work": "/nonexistent"}


def test_pld_check_accepts_and_rejects():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "x_eps4_s0.json"
        good = fake_sidecar(4.0)
        f.write_text(json.dumps(good))
        assert PC.check(f)["ok"], PC.check(f)["problems"]
        for mutate in (lambda s: s["accounting"]["dp_sgd"].__setitem__("sigma_sgd", 0.55),
                       lambda s: s["accounting"]["selection"].__setitem__("noise_std", 10.0),
                       lambda s: s["accounting"].__setitem__("eps_joint_prv", 4.01),
                       lambda s: s["settings"].__setitem__("lora_rank", 64),
                       lambda s: s["training"].__setitem__("lr", 1e-3),
                       lambda s: s["sampling"].__setitem__("private_stats", {"n_trails": 1}),
                       lambda s: s.__setitem__("train_sec", 10.0),
                       lambda s: s["settings"].update(model="Qwen/Qwen2.5-1.5B", version="v1"),
                       lambda s: s["training"].__setitem__("model", "Qwen/Qwen2.5-1.5B"),
                       lambda s: s.__setitem__("row", "path_1p5b")):
            bad = json.loads(json.dumps(good))
            mutate(bad)
            f.write_text(json.dumps(bad))
            assert not PC.check(f)["ok"]
        f7 = Path(td) / "y_eps1_s0.json"
        f7.write_text(json.dumps(fake_sidecar(1.0, "7b")))
        assert PC.check(f7)["ok"], PC.check(f7)["problems"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name, flush=True)
    print("all accounting tests passed")
