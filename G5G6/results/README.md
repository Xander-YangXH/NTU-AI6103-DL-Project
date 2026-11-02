# Exported results

This folder follows the one-JSON-per-run organization of G3G4/G7, with explicit protocol flags. `main/` has26 completed formal runs, `tune/` has18 runs on the final common LR grid. Raw datasets, caches, checkpoints and long logs are kept in the separately verified local backup, not git.

## Schema

`run_id`, `stage`, `group`, `task`, `lr`, `seed`, `eval_split`, `final.acc`, `final.f1`, `final.primary`, `steps`, `lora_params`, `head_params`, `ranks`, `time_s`, `peak_mem_gb`, `cfg`, `versions` follow neighboring branches. MRPC `final.primary` is F1 for reporting; **historical LR selection actually used mean(acc,F1)**, recorded separately in `selection_rule` and `source_result.metrics.selection_score`.

- `protocol_id=hanyanze_independent_20261030`, `shared_protocol_compliant=false`: measurements are not shared-protocol G3/G4 comparators.
- `per_epoch=[]`: no per-epoch validation measurements were collected; not zero-filled data.
- `step0_grad_B`: first training backward before the first update, unclipped B-only norm. The definition is attached; not an independently evaluated probe.
- `loss_first_steps`, `b_grad_first_steps`: first500 actual optimizer-step records.
- `time_s.total`: fine-tune/eval/reload job duration only. Head/calibration are shared, separately labeled; unmeasured load/allocation times are not invented.
- `host`, `slurm_job`: null (private machine identity omitted; run did not use Slurm).
- `source_result`: original numerical fields with machine-specific absolute paths made relative. `source_result_sha256`/`source_predictions_sha256` identify the untouched backup sources.
- `predictions`: actual labels, predictions and logits, allowing metric recomputation without data downloads.

`initialization/` holds13 per-task/seed weighted spectrum/gradient/rank summaries, shared acrossG5/G6. `configs/splits/` records the precise fit/selection membership. `chosen_lr.json` records final-grid tuning. `verification.json` is the server artifact audit; `diagnostics/` adds numerical and real-hook checks. `provenance.json` records source hashes.

The original study evaluated final epochs only; summary SD uses ddof1 and paired accuracy intervals use Student t. Parameter-budget differences and protocol mismatch are discussed in `../docs/protocol.zh-CN.md`.
