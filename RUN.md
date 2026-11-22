<p align="right"><b>English</b> | <a href="RUN.zh-CN.md">中文</a></p>

# How to run the code

This file covers the environment, the data, a smoke test that checks the setup, and the commands for every group. The project overview and the results are in [`README.md`](README.md).

## 1. What is in the package

| Folder | Contents | Entry point |
|---|---|---|
| `G1G2/` | G1 LoRA, G2 magnitude control; the controls and H4 analyses in `jobs/` | `src/train.py` |
| `G3G4/` | G3 EVA, G4 parameter-cost allocation; the shared data split and training loop | `src/train.py` |
| `G5G6/` | G5 / G6 GW-EVA (package `gw_eva`) | `python -m gw_eva.study` |
| `G7/` | G7 LoRA-GA | `src/train.py` |
| `G8/` | G8 EVA calibrated on wikitext-103 | `src/train.py` |
| `G9/` | G9 EVA calibrated on another task | `src/train.py` |
| `H1 Diagnosis/` | H1 subspace diagnosis (GW-EVA vs EVA) | `scripts/run_h1.py` |
| `summary/` | Cross-group tables, figures, report draft, results page | `collect.py`, `figures.py`, `site/build_data.py` |
| `tools/` | Smoke test of every group (`smoke_test.py`, `smoke_test.sbatch`) | `smoke_test.py` |

Every `*/results/` folder already holds the run JSONs behind the report, so the tables, figures and results page can be rebuilt without any GPU (section 6).

## 2. Hardware

- **Training needs an NVIDIA GPU with bf16** (Ampere or newer). All runs in the report used one GPU with at most 24 GB of memory; the shared runs ran on L40S, A40, RTX 4090 D and an RTX 3050 Ti laptop GPU.
- G5/G6, G7 and H1 call CUDA directly and only run on a GPU. G1–G4, G8 and G9 also run on CPU, but only the smoke test is practical there.
- Time per run on one L40S: 1–4 min for MRPC / RTE, 3–8 min for SST-2. Two runs fit side by side on one GPU.
- The summary scripts and the results page need no GPU.

## 3. Environment

One environment runs every group. It is the one most groups used for the report (torch 2.11.0, transformers 5.21.0, peft 0.21.2, datasets 5.1.0).

```bash
python3.11 -m venv .venv            # Python 3.11 or 3.12; conda works too
source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # pick the wheel for your CUDA driver
pip install -r requirements.txt
```

On a CPU-only machine or macOS, install `torch==2.11.0` from PyPI instead. [`requirements.txt`](requirements.txt) lists every third-party package the code imports. G9's notebook `src/overlap_analysis.ipynb` additionally needs `pandas` and Jupyter.

**Verified on 2026-11-22** with the smoke test (section 5) from this package:

| Where | Python / torch / transformers / peft / datasets | Result |
|---|---|---|
| NTU CCDS TC2, NVIDIA L40S (Slurm) | 3.11.17 / 2.11.0+cu128 / 5.21.0 / 0.21.2 / 5.1.0 | Smoke test: G1–G4, G5/G6, G7, G8, G9 and H1 all **passed** (H1 about 20 min, the others 9–70 s each) |
| Same node, two full runs | same | G1, RTE, seed 0, lr 5e-4: **reproduces the published run exactly** (74.01, identical step-0 gradient and first loss). G7, RTE, seed 0: runs to completion with 70.76 (published 66.43, see below) |
| macOS, CPU only | 3.12.13 / 2.11.0 / 5.21.0 / 0.21.2 / 5.1.0 | Smoke test: G1–G4, G8, G9 **passed**; G5/G6 passed with the CUDA calls stubbed; G7 and H1 skipped (GPU only). The groups' own offline tests (`G3G4/tests/test_tiny.py`, `G5G6/tests/`) pass |

So this one environment runs every group. Groups whose published runs used it (G1–G4, G8 and all controls) reproduce exactly on the same GPU model. G7, G5/G6 and H1 were run with other library versions (table below); in this environment they give similar but not identical numbers: G7 above, and H1 on RTE seed 0 gives EVA–GW 0.421 and GW–GW 0.783 against the published 0.479 and 0.853, the same conclusion. To match their published numbers exactly, use the versions they recorded.

**Versions the published runs used.** Results can shift slightly with library versions and GPUs (README, "Known limitations"). For a bit-for-bit rerun of one group, use the versions recorded in its run JSONs (`versions`, `gpu`):

| Groups | Python | torch | transformers | peft | datasets | Recorded in |
|---|---|---|---|---|---|---|
| G1–G4, G8, all controls | 3.11 / 3.12 | 2.11.0+cu128 | 5.21.0 | 0.21.2 | 5.1.0 | `G3G4/requirements.txt`, `G3G4/env_gweva.yml`, `G3G4/pip_freeze.txt` |
| G9 | – | 2.11.0+cu128 | 5.20.2 | 0.21.2 | – | run JSONs |
| G5 / G6, H1 | ≥ 3.11 | 2.8.0 | 4.45.1 | 0.12.0 | 2.20.0 | `G5G6/requirements.txt` |
| G7 | 3.9 | 2.4.0 | 4.52.4 | 0.17.1 | 2.20.0 | `G7/requirements.txt` |

## 4. Data

Compute nodes on most clusters have no internet, so download once on a machine that has it (a login node or your own computer) and copy the folders over if needed. Training jobs then run with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`.

**4.1 Shared data folder (G1–G4, G7, G8, G9).** RoBERTa-base, GLUE SST-2 / MRPC / RTE and wikitext-103 (for G8), about 1 GB:

```bash
export GWEVA_DATA=$HOME/gweva-data     # any folder; every group reads this variable (default ~/gweva-data)
python G8/jobs/prefetch.py
```

This writes `$GWEVA_DATA/roberta-base/`, `$GWEVA_DATA/glue/{sst2,mrpc,rte}/` and `$GWEVA_DATA/wikitext103/`. Set `GWEVA_DATA` the same way in every shell and job that trains.

**4.2 G5/G6 and H1.** They download pinned revisions of the same model and GLUE parquet files, checked by SHA-256, into `G5G6/models/` and `G5G6/data/` (about 500 MB):

```bash
cd G5G6
PYTHONPATH=src python jobs/prefetch.py --tasks rte mrpc sst2
cd ..
```

H1 reuses these files when `GWEVA_ROOT` points at `G5G6/` (section 7.6).

## 5. Check the setup: smoke test

The smoke test runs each group's real entry point on a few dozen examples for one epoch, then checks the result JSON it wrote (final metric, step-0 gradient on B, loss curve, LoRA parameter count). It takes a few minutes and proves that code, environment and data work together; its numbers mean nothing.

```bash
python tools/smoke_test.py                    # G1–G4, G8, G9, G7, G5G6; on CPU, G7 is skipped and G5G6 runs with CUDA calls stubbed
python tools/smoke_test.py --groups G3 G8     # selected groups
python tools/smoke_test.py --groups H1        # H1 on a GPU, about 10–30 min
```

On a Slurm cluster, from the repository root (partition, QoS and node flags are cluster-specific; this is NTU CCDS TC2):

```bash
sbatch -p MGPU-TC2 --qos=normal --nodelist=TC2N08 --export=ALL,VENV=$PWD/.venv,GWEVA_DATA=$HOME/gweva-data tools/smoke_test.sbatch
```

Each group prints `[PASSED]`, `[SKIPPED]` with the reason, or `[FAILED]` with the end of its error output.

## 6. Rebuild tables, figures and the results page (no GPU)

```bash
python summary/collect.py           # summary/tables.md, summary/all_runs.csv
python summary/figures.py           # summary/figures/*.png, *.pdf
python summary/site/build_data.py   # summary/site/data.js
python -m http.server 8000 --directory summary/site   # then open http://localhost:8000
```

These read only the JSONs under `*/results/` and `H1 Diagnosis/results/`. Figures can differ in rendering details with another matplotlib version; the numbers do not.

## 7. Run the experiments

**Before rerunning:** `train.py` skips a run whose result JSON already exists, and every `results/` folder already holds the published runs. Either pass `--out` with a new folder for single runs, or move the published folder aside for a full rerun (for example `mv results results_published`). The `main` and `lowres` queues read the learning rates from `results/tune/`, so keep or rerun the tuning stage first.

Learning rates chosen on the dev split (low resource reuses the SST-2 choice; G8 and G9 reuse G3's):

| Group | SST-2 | MRPC | RTE |
|---|---|---|---|
| G1 | 5e-4 | 2e-4 | 5e-4 |
| G2 | 5e-4 | 1e-3 | 2e-4 |
| G3, G8, G9 | 1e-3 | 2e-4 | 2e-4 |
| G4 | 5e-4 | 2e-4 | 2e-4 |
| G5 | 1e-3 | 2e-4 | 2e-4 |
| G6 | 5e-4 | 2e-4 | 2e-4 |
| G7 | 2e-4 | 2e-4 | 2e-4 |

Seeds: SST-2 0–2, MRPC and RTE 0–4, low resource 0–4 (G5/G6 0–2). Stages: `tune` (seed 0, learning-rate grid 2e-4 / 5e-4 / 1e-3, evaluated on the 10% dev split), `main` (official validation), `lowres` (1,000-example SST-2 subset).

### 7.1 G1 / G2 (`G1G2/`) and G3 / G4 (`G3G4/`)

```bash
cd G1G2                                        # or G3G4 with --group G3 / G4
python src/train.py --stage main --task rte --group G1 --lr 5e-4 --seed 0 --out results_new
```

A whole stage, two runs at a time on one GPU (Linux):

```bash
python jobs/make_queue.py tune                 # writes jobs/queues/tune.txt; then main, lowres
bash jobs/run_queue.sh jobs/queues/tune.txt 2
python jobs/summarize.py --g3 ../G3G4/results  # G1G2: tables, paired with G3 by seed
```

On Slurm: `sbatch --export=ALL,STAGE=tune,NPAR=2 jobs/g12.sbatch` (G3G4: `jobs/g34.sbatch`). The templates hold the TC2 partition and paths; adjust them for another cluster.

The controls behind the report's low-resource, noise-floor and per-layer H4 analyses are extra stages of `G1G2/jobs/make_queue.py` (`lowres_ctl`, `seeds10`, `g3_lowres_ctl`, `quick4`, `quick_ad`; see its docstring) and the scripts `jobs/g3_control.py`, `jobs/eva_noise_floor.py` and `jobs/h4_layers.py`. They import `../G3G4/src` and `../G8/src`, which the package layout provides.

### 7.2 G5 / G6 (`G5G6/`)

```bash
cd G5G6
export PYTHONPATH=$PWD/src
python -m gw_eva.study --tasks rte             # tuning on seed 0, then all seeds, for G5 and G6
python -m gw_eva.study                         # all tasks: rte mrpc sst2 sst2_lowres (sst2_lowres needs sst2 first)
python jobs/summarize.py                       # results table from results/main
```

Outputs go to `G5G6/outputs/gw_eva/shared_study/`, which also keeps each seed's head checkpoint that H1 needs. `jobs/parallel_worker.py` runs two task queues side by side in the same study folder, as in the original rerun.

### 7.3 G7 (`G7/`)

```bash
cd G7
python src/train.py --stage main --task rte --lr 2e-4 --seed 0 --out results_new
python jobs/run_experiments.py                 # tuning then main for all tasks, two processes on one GPU
python jobs/summarize.py
```

`jobs/run_g7.sbatch` is the Slurm version.

### 7.4 G8 (`G8/`)

```bash
cd G8
python src/train.py --stage main --task rte --lr 2e-4 --seed 0 --out results_new
python jobs/make_queue.py main && bash jobs/run_queue.sh jobs/queues/main.txt 2
```

G8 has no tuning stage; it uses G3's learning rates.

### 7.5 G9 (`G9/`)

```bash
cd G9
python src/train.py --stage main --task rte --lr 2e-4 --seed 0 --out results_new
```

The 13 main runs are listed in `run_main.bat` (Windows). On Linux / macOS:

```bash
for s in 0 1 2; do python src/train.py --stage main --task sst2 --lr 1e-3 --seed $s; done
for t in mrpc rte; do for s in 0 1 2 3 4; do python src/train.py --stage main --task $t --lr 2e-4 --seed $s; done; done
```

The overlap analysis is `src/overlap_analysis.ipynb`; `G1G2/jobs/h4_layers.py` rebuilds the same overlaps per layer from the repository.

### 7.6 H1 (`H1 Diagnosis/`)

H1 reuses the head checkpoints of the G5/G6 study, so run G5/G6 (7.2) for the task first, at least up to its initialization step.

```bash
cd "H1 Diagnosis"
export GWEVA_ROOT=$PWD/../G5G6                 # reuse G5G6's model and data
export PYTHONPATH=$PWD/src:$PWD/scripts
python scripts/run_h1.py --task rte --seed 0 --checkpoint-root ../G5G6/outputs/gw_eva/shared_study --output results_new
python scripts/summarize_h1.py results_new    # needs all 4 tasks x 3 seeds
```

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `skip <run>: done` and nothing trains | The JSON exists already; use `--out` with a new folder (section 7) |
| `FileNotFoundError` under `gweva-data` | `GWEVA_DATA` is unset or points elsewhere; rerun section 4.1 or export the right path |
| Network errors on a compute node | Download on a machine with internet (section 4) and set `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` |
| `G5G6/models/roberta-base is missing` | Run section 4.2 |
| G7 / G5G6 / H1 stop on a CPU machine | They need CUDA by design |
| Slightly different numbers from the report | Different GPU or library versions change bf16 numerics; see section 3 for the versions used |
| `run_queue.sh`: `xargs: invalid option -- 'd'` | macOS's xargs; use Linux or run the commands in `jobs/queues/*.txt` one by one |

## 9. Changes made for this package

The experiment code is as in the repository, with these additions so that the folders run together from one environment:

- `requirements.txt`, `RUN.md` / `RUN.zh-CN.md` and `tools/` (smoke test) are new.
- `G7/src/train.py` and `G7/jobs/run_experiments.py` no longer require a Slurm job; they still require a CUDA GPU. `slurm_job` in the JSON is `null` outside Slurm. The training itself is unchanged.
