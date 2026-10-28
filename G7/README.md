<p align="right"><b>English</b> | <a href="README.zh-CN.md">中文</a></p>

# G7: LoRA-GA under the shared GW-EVA protocol

Owner: Liyouran. This folder contains G7 code, Slurm execution and results on branch `liyouran`.

The implementation follows the shared settings in the repository's `main` README and the data/training pipeline from `yinchenyu_g3g4:G3G4/src/`. The adopted settings and source details are documented in [the protocol audit](docs/protocol-audit.zh-CN.md).

## Method

After one epoch of classification-head warm-up, G7 estimates the mean loss gradient of each of the 72 attention/FFN weight matrices on 1,024 labelled training examples. Calibration uses FP32 and evaluation mode, with the warmed-up head frozen. Each example has equal weight, including incomplete batches.

For a gradient SVD `G = U S Vᵀ`, the authors' `ArB2r` initialisation uses `A = c V[:, :r]ᵀ` and `B = c U[:, r:2r]`, with `c = d_out**0.25 / sqrt(16)`. Both factors use singular vectors; the singular values are not multiplied into B. The base weight becomes `W₀ − (α/r)B₀A₀`, preserving the effective initial weight. The implementation records the compensation error and stores the initial factors with the final adapter in the local cache for reconstruction.

Source: [LoRA-GA authors' code](https://github.com/Outsider565/LoRA-GA/tree/c4cd5372c75b290924214b348008891f744512ef), `peft/src/peft/tuners/lora/layer.py::lora_ga_init`. G7 keeps `B ≠ 0`; the GW-EVA groups G5/G6 instead initialise only A and keep B zero.

## Shared settings

| Setting | G7 |
|---|---|
| Model / tasks | RoBERTa-base / SST-2, MRPC, RTE |
| LoRA | All 72 attention and FFN linear layers; r=8, α=8, dropout=0 |
| Trainable parameters | 1,327,104 LoRA + 592,130 classification head |
| Learning-rate grid | 2e-4, 5e-4, 1e-3; tune on seed 0 |
| Main seeds | SST-2: 0–2; MRPC/RTE: 0–4 |
| Epochs | SST-2: 3; MRPC/RTE: 10 |
| Batch / maximum length | 32 / 128; dynamic padding |
| Optimiser | AdamW, fine-tuning weight decay 0, linear LR schedule, 6% warm-up; no clipping |
| Precision | FP32 weights with BF16 mixed-precision training |
| Head warm-up | 1 epoch, LR 1e-3, frozen encoder; `model.train()` as in the reference |
| Data split | NumPy permutation with seed 2026; 10% dev, remaining 90% train |
| Calibration | 1,024 training examples, batch 16, NumPy sampling seed equal to the run seed; labels retained for G7 |
| Selection | Last dev epoch; MRPC F1, other tasks accuracy; ties choose the smaller LR |
| Final evaluation | Last training epoch on the official validation split |

The head warm-up reuses the reference's AdamW default weight decay of 0.01; fine-tuning uses the explicitly specified weight decay 0. The training loader's seeded generator is reused continuously across head warm-up and fine-tuning. Epoch diagnostics use dev, and the official validation split is evaluated once at the end of each main run. `per_epoch_eval_split` identifies the diagnostic split in each JSON.

There are 9 tuning runs and 13 separate main runs. No low-resource G7 run is included: the proposal assigns low-resource SST-2 to G1/G3/G5.

## Run

Use the project virtual environment and dependencies listed in this folder. Data and pretrained weights are local, outside Git. `GWEVA_DATA` points to a directory containing `roberta-base/` and `glue/{sst2,mrpc,rte}/` saved with Hugging Face datasets.

```bash
cd G7
export GWEVA_PYTHON=/path/to/project/.venv/bin/python
export GWEVA_DATA=/path/to/gweva-data
"$GWEVA_PYTHON" jobs/prepare_data.py
mkdir -p logs
sbatch jobs/run_g7.sbatch
"$GWEVA_PYTHON" jobs/summarize.py
```

The default preparation reuses this workspace's downloaded parquet files and model. On a new machine, add `--download` to the preparation command to fetch the pinned public revisions. The batch job runs tuning first, selects each task's LR, then runs the main stage; two processes share its one allocated GPU.

Each run writes the shared-format JSON under `results/tune/` or `results/main/`. Finished runs are skipped when resubmitted. Caches, weights, raw data and logs stay outside Git.

## Results and comparison

See [the current summary](results/summary.md) for completion counts, LR selection, mean ± sample SD, costs, loss curves and same-seed G7−G3/G4 differences with 95% t intervals. [Scalar reference metrics](results/reference_metrics.json) preserve the other members' measurements and their source commit, allowing summary regeneration without their Git branches.

The G7 environment is Python 3.9, torch 2.4.0, Transformers 4.52.4 and PEFT 0.17.1 on an L40S. The shared G3/G4 results were produced on an A40 with torch 2.11.0, Transformers 5.21.0 and PEFT 0.21.2. Per-run JSONs record the exact versions and GPU. The paired comparison shares seeds and training settings and retains these environment differences; G3's rank redistribution also changes its actual adapter parameter count.

## Files

- `src/data.py`: the shared split, tokenisation and loader implementation.
- `src/train.py`: the shared training sequence with G7 initialisation and final-only official evaluation.
- `src/init_g7.py`: labelled mean-gradient SVD and residual-weight compensation.
- `jobs/`: preparation, Slurm execution and result aggregation.
- `results/`: compact run JSONs, CSV, summary and scientific figures.
- `docs/protocol-audit.zh-CN.md`: old-versus-shared protocol comparison.
