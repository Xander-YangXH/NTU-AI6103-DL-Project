<p align="right"><b>English</b> | <a href="README.zh-CN.md">中文</a></p>

# G7: LoRA-GA under the shared GW-EVA protocol

Owner: Liyouran. This folder contains G7 code, Slurm execution and results on branch `liyouran`.

The implementation follows the shared settings in the repository's `main` README. The data and training pipeline is adapted from the shared source at commit `26616379899f0abce7d693e227ad4e03138312ab`. [The experiment protocol](docs/experiment-protocol.zh-CN.md) documents G7's method, data and execution settings.

## Method

After one epoch of classification-head warm-up, G7 estimates the mean loss gradient of each of the 72 attention/FFN weight matrices on 1,024 labelled training examples. Calibration uses FP32 and evaluation mode, with the warmed-up head frozen. Each example has equal weight, including incomplete batches.

For a gradient SVD `G = U S Vᵀ`, the authors' `ArB2r` initialisation uses `A = c V[:, :r]ᵀ` and `B = c U[:, r:2r]`, with `c = d_out**0.25 / sqrt(16)`. Both factors use singular vectors; the singular values are not multiplied into B. The base weight becomes `W₀ − (α/r)B₀A₀`, preserving the effective initial weight. The implementation records the compensation error and stores the initial factors with the final adapter in the local cache for reconstruction.

Source: [LoRA-GA authors' code](https://github.com/Outsider565/LoRA-GA/tree/c4cd5372c75b290924214b348008891f744512ef), `peft/src/peft/tuners/lora/layer.py::lora_ga_init`. Both A and B are trainable from the first update.

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

There are 9 tuning runs and 13 separate main runs on the three full-data tasks.

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

## Results and analysis

| Task | Seeds | Accuracy (%) | F1 (%) |
|---|---:|---:|---:|
| SST-2 | 3 | 93.96 ± 0.46 | — |
| MRPC | 5 | 88.53 ± 0.74 | 91.83 ± 0.46 |
| RTE | 5 | 56.68 ± 8.85 | — |

All tasks select LR 2e-4. Increasing the rate to 1e-3 reduces seed-0 dev primary by 2.44/8.63/18.07 percentage points on SST-2/MRPC/RTE. RTE's final accuracy spans 47.29–66.43%; seed 1 predicts label 1 on all 277 validation examples. The stored loss traces remain finite, documenting a classification failure rather than a numerical exception.

[The full G7 analysis](results/summary.md) includes per-task interpretations, mean confidence intervals, precision/recall, all LR candidates, per-seed predictions, initialisation errors, costs and the first-500-step curves. It is regenerated exclusively from G7's own JSONs. [中文结果解读](docs/analysis.zh-CN.md).

G7 uses Python 3.9, torch 2.4.0, Transformers 4.52.4 and PEFT 0.17.1 on an L40S. Exact versions and GPU are recorded in each run JSON. Slurm job 48482 completed all 22 runs in 1,179 seconds with two processes sharing one GPU.

## Files

- `src/data.py`: the shared split, tokenisation and loader implementation.
- `src/train.py`: the shared training sequence with G7 initialisation and final-only official evaluation.
- `src/init_g7.py`: labelled mean-gradient SVD and residual-weight compensation.
- `jobs/`: preparation, Slurm execution and result aggregation.
- `results/`: compact run JSONs, CSV, summary and scientific figures.
- `docs/experiment-protocol.zh-CN.md`: method, data and execution protocol.
- `docs/analysis.zh-CN.md`: G7 results and interpretation in Chinese.
