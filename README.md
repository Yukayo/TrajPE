# TrajPE: Differentially Private Check-in Synthesis through Venue-Aware Private Evolution

TrajPE releases a synthetic table of check-in trails over a public venue catalog under trail-level
$(\varepsilon,\delta)$-differential privacy. A small language model fine-tuned with DP-SGD proposes candidate trails,
nineteen noisy histograms measure population statistics, and Private Evolution with noise-matched cluster votes
calibrates the candidates before a final union fit produces the release. All mechanisms share one budget composed with
privacy loss distributions.

This repository contains the method, the baseline adapters, the evaluation benchmark (58 metrics in 26 families) and
the next-POI evaluators used in the paper.

## Repository layout

| Path | Content |
|---|---|
| `trajpe/` | Method: catalog, trails, DP mechanisms, workload histograms, embedding, Private Evolution, fusion, benchmark metrics |
| `scripts/run_trajpe.sh`, `scripts/v5_job.sh` | End-to-end TrajPE release (calibration, DP-SGD generator, PE, post-processing, privacy sidecar) |
| `scripts/run_hybrid.py`, `scripts/postprocess_release.py` | PE stage and release post-processing |
| `scripts/prepare_data.py`, `scripts/build_public_catalog.py`, `scripts/map_categories_v2_to_fsq.py` | Data preparation |
| `scripts/eval_bench.py`, `scripts/final_tables.py` | Fidelity metrics and the paired statistical comparison |
| `scripts/run_ablation.py`, `scripts/eff_run.py` | Ablations and runtime / GPU-memory measurement |
| `downstream/` | Next-POI evaluators (Transformer, Markov) trained on a release and tested on real trails |
| `baselines/` | Baseline adapters: DP-LoRA (+vp), CTCL, PATH, PateGAIL++, AIM-seq, LDPTrace |
| `configs/` | Frozen TrajPE configurations (`v5_small_eps.json` for $\varepsilon<4$, `v5_large_eps.json` otherwise), dataset definitions, benchmark families |
| `frozen/` | Public category leaf list used by the category mapping |
| `data/processed/category_map_*.json` | Public category mappings (derived from public taxonomies only) |
| `tests/` | Unit and integration tests |

## Installation

The experiments use three Python 3.12 environments because vLLM, Opacus and JAX pin different dependencies.

| Environment | Used for | Requirements |
|---|---|---|
| main | data preparation, PE, sampling with vLLM, evaluation | `requirements/main.txt` |
| dplora | DP-SGD fine-tuning (Opacus + PEFT) | `requirements/dplora.txt` |
| aimseq | AIM-seq (JAX, private-pgm) | `requirements/aimseq.txt` |

```bash
python -m venv envs/main   && envs/main/bin/pip install -r requirements/main.txt
python -m venv envs/dplora && envs/dplora/bin/pip install -r requirements/dplora.txt
python -m venv envs/aimseq && envs/aimseq/bin/pip install -r requirements/aimseq.txt
```

The code locates the repository and its working directory through environment variables:

```bash
export TRAJPE_ROOT=$PWD                    # repository root (default: detected automatically)
export TRAJPE_WORK=$PWD/work               # raw data, model cache, runs (default: $TRAJPE_ROOT/work)
export TRAJPE_PYTHON=$PWD/envs/main/bin/python
export TRAJPE_PYTHON_DPLORA=$PWD/envs/dplora/bin/python
export TRAJPE_PYTHON_AIMSEQ=$PWD/envs/aimseq/bin/python
export TRAJPE_PYTHON_GRID=$TRAJPE_PYTHON   # interpreter for the grid baselines
```

Download the language models into the model cache (sampling and training run offline). The pinned
`huggingface_hub==2.2.0` installs the `hf` command (it no longer provides `huggingface-cli`); the revisions are the ones
used in the paper:

```bash
export HF_HOME=$TRAJPE_WORK/hf_cache
HF=envs/main/bin/hf
$HF download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775
$HF download Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28   # naive PE and the 7B DP-LoRA variants
$HF download sentence-transformers/all-MiniLM-L6-v2 --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41   # naive PE (--embed text)
$HF download Qwen/Qwen3-Embedding-0.6B --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3   # PATH row embeddings
```

Every loader requests exactly these commits (`trajpe/hf_revisions.py`; PATH and CTCL pin the same ones), so the offline
runs read the downloaded snapshots directly and need no `main` reference in the cache
(`tests/test_hf_revisions.py` checks this on a fresh cache).

## Data

No data is redistributed. Place the public sources under `$TRAJPE_WORK/data/raw/`:

| Directory | Source |
|---|---|
| `massive_steps_<city>/{train,validation,test}.parquet` | Massive-STEPS (Hugging Face, `CRUISEResearchGroup/Massive-STEPS-*`) |
| `tsmc2014/extracted/dataset_tsmc2014/dataset_TSMC2014_{TKY,NYC}.txt` | TSMC2014 Foursquare check-ins (Yang et al., IEEE TSMC 2015) |
| `gowalla_snap/loc-gowalla_totalCheckins.txt.gz` | Gowalla check-ins (SNAP) |
| `gowalla_liu/gowalla_spots_subset1.csv`, `gowalla_liu/gowalla_category_structure.json` | Gowalla spot categories (Liu et al., CIKM 2014) |
| `fsq_os_places/foursquare_places.parquet` | Foursquare Open Source Places, release 2024-11-19 |
| `semantic_trails/categories.csv`, `fsq_v2_taxonomy/` | Public category taxonomies (Semantic Trails; Foursquare v2 hierarchy) |

Then build the public venue catalogs and the trail tables:

```bash
$TRAJPE_PYTHON scripts/build_public_catalog.py istanbul tokyo sao_paulo kuwait_city petaling_jaya moscow bandung gw_stockholm gw_san_francisco
$TRAJPE_PYTHON scripts/prepare_data.py --datasets istanbul tokyo sao_paulo kuwait_city petaling_jaya moscow bandung gw_stockholm gw_san_francisco
```

Each dataset is written to `data/processed/<dataset>/` (`pois.parquet`, `train/val/test.parquet`, `meta.json`). Trails are
segmented at eight-hour gaps and capped at 20 check-ins; every raw check-in is mapped to the public catalog by a
record-wise rule that depends only on the catalog. `data/processed/category_map_*.json` are included, so
`scripts/map_categories_v2_to_fsq.py` is only needed to rebuild them.

## Running TrajPE

```bash
bash scripts/run_trajpe.sh istanbul 4        # dataset, epsilon, [seed]
```

The release is written to `$TRAJPE_WORK/runs/<dataset>/trajpe_e<eps>_s<seed>div/synth.parquet` (50,000 trails) with a
JSON sidecar that records the noise scales and the composed $(\varepsilon,\delta)$ ($\delta=10^{-6}$). The release
path runs in release mode: no statistic of the private splits that is not part of the DP mechanism (such as a fidelity
comparison with the training or test trails) is computed, written to the sidecar or printed. `scripts/run_pe.py` and
`scripts/postprocess_release.py` compute such comparisons only with an explicit `--metrics` (internal evaluation; refused
when `TRAJPE_GRID_NOMETRICS=1`); evaluate a release with `scripts/eval_bench.py` (below) instead. The run trains the
DP-SGD generator with `TRAJPE_PYTHON_DPLORA` and runs Private Evolution with `TRAJPE_PYTHON` on one GPU (measured peak
memory about 22 GB). All hyperparameters are read from `configs/v5_*_eps.json`.

## Privacy accounting

The noise scales and the composed budget of every release depend only on public constants. They can be recomputed without
any data, for example for the four budgets of the paper's privacy table ($\delta=10^{-6}$; batch 2048 for $\varepsilon<4$,
1024 otherwise):

```bash
for e in 1 2; do $TRAJPE_PYTHON baselines/dplora/hybrid.py calibrate --eps $e --sgd_share 0.3 --batch 2048 --epochs 3 --out calib_e$e.json; done
for e in 4 8; do $TRAJPE_PYTHON baselines/dplora/hybrid.py calibrate --eps $e --sgd_share 0.3 --batch 1024 --epochs 3 --out calib_e$e.json; done
```

Each JSON lists the DP-SGD noise multiplier `sigma_sgd`, sampling rate `sample_rate`, number of steps `steps`, the Gaussian-DP
parameter of the measurements and votes `mu_pe`, and the PLD-composed `eps_composed_pld`. The guarantee is per trail; a user
with $k$ trails is covered by group privacy at $k\varepsilon$.

## Baselines

| Method | Entry point | Notes |
|---|---|---|
| DP-LoRA, DP-LoRA+vp | `baselines/dplora/pipeline_default.py --dataset D --eps E --seed S --batch 1024 --epochs 10 --lr 0.0003 --rank 8 --clip 1.0 --out OUT --max_new 512 --lora_alpha 32 --lr_schedule constant --max_target_tokens 512 --defaults_cfg baselines/dplora/frozen_default.json [--variant vp]` | authors' settings; `--stage gpu` with the dplora environment, then `--stage cpu` with the main environment |
| CTCL | `baselines/ctcl/r2/pipeline_ctcl_r2.py --method ctcl --dataset D --eps E --seed S` | re-implementation from the paper; the public topic tables in `baselines/ctcl/r2/topics/` are tied to the catalog hash; if your catalog differs, rebuild them with `baselines/ctcl/r2/topics_r2.py table --datasets D` (public catalog and the Qwen2.5-0.5B-Instruct tokenizer only) |
| PATH | `baselines/path/v2r2fx/pipeline.py --dataset D --eps E --model_key 0.5b --seed S` | re-implementation from the paper |
| PateGAIL++ | `baselines/pategail/run_pategail.py --dataset D --eps E --seed S --arm main --out OUT` | needs the upstream repository (below) |
| AIM-seq | `baselines/aimseq/run_aimseq.py --config baselines/aimseq/frozen_v2.json --dataset D --eps E --seed S --out OUT` | aimseq environment; needs private-pgm |
| LDPTrace | `baselines/run_ldptrace.py --dataset D --eps E --seed S --out OUT` | adapted from the authors' code |
| naive PE | `scripts/run_pe.py --dataset D --api llm --sim_frac 0 --eps E --seed S --n_syn 10000 --T 3 --threshold_sigmas 1.0 --marginals off --venue_aware off --transitions off --embed text --llm_variation llm --tag TAG` | zero-shot Qwen2.5-7B-Instruct |

PrivTrace, AdaTrace and HRNet are not included because their upstream code has no licence or a GPL licence; use the
authors' repositories. Clone the upstream repositories used by the adapters into `$TRAJPE_WORK/baselines_src/<name>/repo`
at the pinned commits:

| Name | Repository | Commit |
|---|---|---|
| `pategail` | github.com/yingjie1234/PATEGAIL-PlusPlus-Utility-Optimized-Private-Trajectory-Generation | `fe8f78c64db2424d52178d608d3294f57646d980` |
| `private-pgm` | github.com/ryan112358/private-pgm | `59e378de766bbc5403b8679a813ac42dde1a6909` |
| `ldptrace` | github.com/yuntaod/LDPTrace | `2d30e4135db11fd50d1fb98f59a1e84ebc61b218` |
| `privtrace` | github.com/DpTrace/PrivTrace | `b06cef7d8df0305b10f309e8b75660949946f22a` |
| `adatrace` | github.com/git-disl/AdaTrace | `55fc77b7255b8871c4ac6f95f6c12a141a07bdfc` |
| `hrnet` | github.com/Emory-AIMS/priv_traj_gen | `9495566799da391f6bdf127756e22b2742db12ae` |

All baselines release 50,000 trails under the same trail-level guarantee. Baselines use their authors' settings; the
values are in the `frozen*.json` files next to each adapter.

For batch execution, `scripts/queue_baselines.py` supports LDPTrace, DP-LoRA v2 (plain / venue-prior, 0.5B / 7B),
and naive PE. Its DP-LoRA jobs use the legacy `frozen_v2.json` settings; use `pipeline_default.py` above for the
authors' default configuration. Planning and dry runs do not need data or start training:

```bash
$TRAJPE_PYTHON scripts/queue_baselines.py plan --datasets istanbul --methods dplora_plain --eps 4
$TRAJPE_PYTHON scripts/queue_baselines.py queue --datasets istanbul --methods dplora_plain --eps 4 --dry
```

After preparing the dataset and downloading the models, omit `--dry` to add tasks to `$TRAJPE_WORK/gpuq/jobs.jsonl`.
Adding tasks does not start a worker. Inspect the queue or start one explicitly with:

```bash
$TRAJPE_PYTHON scripts/gpuq.py status --state "$TRAJPE_WORK/gpuq"
$TRAJPE_PYTHON scripts/gpuq.py worker --state "$TRAJPE_WORK/gpuq" --gpu 0 --max_jobs 1
```

Use `--gpu cpu` for LDPTrace tasks. Queueing checks that the prepared training data, public catalog and metadata
exist. `queue_baselines.py --require-frozen` additionally requires the original experiment's private hash records;
those records are not distributed. Methods whose adapters are not included are excluded from the batch queue.

## Evaluation

Fidelity metrics of a release (against the real training split, and the test split as a second reference):

```bash
$TRAJPE_PYTHON scripts/eval_bench.py --dataset istanbul --synth $TRAJPE_WORK/runs/istanbul/trajpe_e4_s0div/synth.parquet \
    --method trajpe --eps 4 --seed 0 --out-dir results/bench/istanbul
```

Next-POI utility (train on the release, test on real trails; Acc@1/5/10, NDCG@10, MRR):

```bash
$TRAJPE_PYTHON downstream/train_eval.py --dataset istanbul --train $TRAJPE_WORK/runs/istanbul/trajpe_e4_s0div/synth.parquet \
    --model transformer --out results/downstream/istanbul/trajpe_eps4_s0_transformer.json --bench-metrics
```

`scripts/final_tables.py` computes the family-level scores $D$, win/tie/loss counts and Holm-corrected Wilcoxon tests
from an inventory of releases and their metric files (see `--help`; use `--no-freeze-check` outside the original
experiment). `scripts/eff_run.py` records wall-clock time per stage and peak GPU memory for any command.

## Tests

```bash
for t in tests/test_*.py; do $TRAJPE_PYTHON $t; done
```

Tests of the DP-SGD components run with `TRAJPE_PYTHON_DPLORA`, and `baselines/aimseq/test_aimseq.py` with
`TRAJPE_PYTHON_AIMSEQ`. Some tests need the processed datasets.

Run the public CTCL tests in their required order (accounting, sampling, pipeline, training, release):

```bash
$TRAJPE_PYTHON baselines/ctcl/tests/run_tests.py
```

They use the prepared public catalogs and cached Qwen tokenizer/model; the training checks run on CPU with a tiny
model. `--only test_pipeline.py` runs a subset. The runner returns zero when the selected tests pass and records
`full_suite` separately in `baselines/ctcl/tests/RESULTS.json`; a missing required test is an error.

## Third-party code

`baselines/pategail/` adapts the PateGAIL++ code (MIT, Copyright (c) 2022 FIB LAB, Tsinghua University; see
`baselines/pategail/LICENSE`). `baselines/run_ldptrace.py` is a modified version of LDPTrace's `main.py`
(Apache-2.0; see `baselines/LICENSE-LDPTrace`), changed for check-in trails and trail-level accounting. The benchmark
metrics in `trajpe/bench_metrics.py` are re-implemented from the definitions in the source papers.
