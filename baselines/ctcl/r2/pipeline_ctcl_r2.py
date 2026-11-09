from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

R2 = Path(__file__).resolve().parent
sys.path.insert(0, str(R2))
import r2conf
import pipeline_ctcl as PC

C = r2conf.C
EXIT_BROKEN = PC.EXIT_BROKEN


class JobR2(PC.Job):
    def __init__(self, a, runner=None, log=None):
        super().__init__(a, runner=runner, log=log)
        self.failed = self.w / "texts.failed_r2.json"
        if self.table.parent != r2conf.TOPIC_DIR and not a.test_data_ok:
            raise SystemExit(f"r2 releases use the r2 topic tables ({r2conf.TOPIC_DIR}), not {self.table}")
        for old in ("texts.failed.json", "texts.failed_r1.json"):
            if (self.w / old).exists():
                raise SystemExit(f"{self.w / old}: a v1/r1 marker in an r2 work dir")

    def do_m2(self):
        st = self.m2_status()
        if st in ("done", "consumed"):
            self.say(f"M2: train.json complete ({st}); training skipped")
            return
        if st == "unusable":
            s = PC.stamp()
            for p in (self.tj, self.wdir, self.texts, self.texts.with_suffix(".json"), self.failed):
                if p.exists():
                    p.rename(p.with_name(f"{p.name}.unusable_{s}"))
            self.say(f"M2: an unusable earlier run was renamed aside (*.unusable_{s}); restarting from step 0")
        self.wait_for_disk()
        cmd = [C.DPLORA_PY, "-I", "-B", R2 / "train_ctcl_r2.py", "--dataset", self.a.dataset, "--eps",
               C.eps_tag(self.a.eps), "--variant", self.variant, "--config", self.config, "--param", self.param,
               "--seed", self.a.seed, "--calib", self.calib, "--table", self.table, "--out_dir", self.w,
               "--mem_cap_gb", self.a.mem_cap_gb]
        if self.a.log_private:
            cmd.append("--log_private")
        cmd += self.a.extra_train.split()
        self.timed("m2", cmd)
        if self.m2_status() != "done":
            raise PC.StateError("training finished without a complete train.json and weights")
        tj = json.loads(self.tj.read_text())
        if (tj.get("coord_decimals"), tj.get("max_resp_tokens")) != (r2conf.COORD_DECIMALS, r2conf.MAX_RESP_TOKENS):
            raise PC.StateError("train.json was not produced with the r2 format")

    def do_sample(self) -> int:
        if self.failed.exists():
            self.say(f"{self.failed.name} exists: the sampling failed for good (r2); exit 3 without re-sampling")
            return EXIT_BROKEN
        if self.texts.exists() and self.texts.with_suffix(".json").exists():
            self.say("sampling: texts.parquet exists (reused)")
            return 0
        if self.m2_status() == "consumed":
            raise PC.StateError("weights were deleted but the texts are missing")
        src = ["--adapter", self.wdir] if self.param == "lora" else ["--model_dir", self.wdir]
        rc = self.timed("sample", [C.TRAJPE_PY, "-I", "-B", R2 / "sample_ctcl_r2.py", "--dataset", self.a.dataset]
                        + src + ["--hist", self.hist, "--table", self.table, "--out", self.texts, "--seed", self.a.seed,
                                 "--n_out", self.a.n_out], ok_codes=(0, EXIT_BROKEN))
        return EXIT_BROKEN if rc == EXIT_BROKEN else 0

    def augment(self, rj: Path):
        super().augment(rj)
        side = json.loads(rj.read_text())
        blk = side["ctcl"]
        blk["spec"] = ("docs/baselines/CTCL_SPEC.md (pre-registration, frozen/ctcl_v1.sha256) + r1 amendment (sampler "
                       "gates) + r2 amendment (unified text format; frozen/ctcl_v1r2.sha256)")
        blk["rev"] = r2conf.REV
        blk["r2"] = r2conf.summary()
        blk["code_sha256"].update(r2conf.code_hashes())
        C.write_json_atomic(rj, side)

    def finish(self):
        self.note("weights_deleted", deleted=True)
        if self.wdir.is_symlink():
            tgt = self.wdir.resolve()
            if tgt.exists():
                shutil.rmtree(tgt)
            self.wdir.unlink()
        elif self.wdir.exists():
            shutil.rmtree(self.wdir)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.rel, self.out)
        os.replace(self.rel.with_suffix(".json"), self.out.with_suffix(".json"))
        self.note("released", out=str(self.out), rev=r2conf.REV)


def main(argv=None, runner=None) -> int:
    return JobR2(PC.parse(argv), runner=runner)()


if __name__ == "__main__":
    sys.exit(main())
