"""Summarise team-protocol G7 runs and same-seed differences from G3/G4.

Run from any directory: python G7/jobs/summarize.py
The small reference_metrics.json preserves the source metrics needed when this
branch is downloaded without the other members' Git refs.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


G7 = Path(__file__).resolve().parents[1]
REPO = G7.parent
TASK_SEEDS = {"sst2": list(range(3)), "mrpc": list(range(5)), "rte": list(range(5))}
LR_GRID = [2e-4, 5e-4, 1e-3]
REFERENCE_BRANCH = "origin/yinchenyu_g3g4"
T95 = {1: 12.7062047364, 2: 4.3026527297, 3: 3.1824463053, 4: 2.7764451052}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def describe(values, percentage_points=False):
    scale = 100 if percentage_points else 1
    values = [float(value) * scale for value in values]
    if not values:
        return {"n": 0, "mean": None, "sd": None, "ci95": None}
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    half = T95[len(values) - 1] * sd / math.sqrt(len(values)) if sd is not None else None
    return {"n": len(values), "mean": mean, "sd": sd,
            "ci95": [mean - half, mean + half] if half is not None else None}


def fmt(stat, scale=1, signed=False):
    if stat["mean"] is None:
        return "pending"
    sign = "+" if signed else ""
    value = f"{stat['mean'] * scale:{sign}.2f}"
    return value + (f" ± {stat['sd'] * scale:.2f}" if stat["sd"] is not None else " (n=1)")


def interval(stat, signed=False):
    if stat["ci95"] is None:
        return "pending"
    sign = "+" if signed else ""
    lo, hi = stat["ci95"]
    return f"[{lo:{sign}.2f}, {hi:{sign}.2f}]"


def reference_snapshot(results):
    destination = results / "reference_metrics.json"
    if destination.exists():
        return json.loads(destination.read_text())
    commit = subprocess.check_output(["git", "rev-parse", REFERENCE_BRANCH], cwd=REPO, text=True).strip()
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, "G3G4/results/main"], cwd=REPO, text=True,
    ).splitlines()
    runs = []
    for path in paths:
        if not path.endswith(".json"):
            continue
        run = json.loads(subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPO, text=True))
        if run.get("group") not in {"G3", "G4"}:
            continue
        runs.append({key: run[key] for key in (
            "run_id", "group", "task", "seed", "lr", "final", "lora_params", "budget_dev", "gpu", "versions"
        )} | {"source_path": path})
    snapshot = {"repository": "https://github.com/Xander-YangXH/NTU-AI6103-DL-Project",
                "branch": "yinchenyu_g3g4", "commit": commit,
                "attribution": "G3/G4 measurements by the owner of yinchenyu_g3g4; copied as scalar references only",
                "runs": runs}
    write_json(destination, snapshot)
    return snapshot


def read_runs(results, stage):
    runs = []
    for path in sorted((results / stage).glob(f"{stage}_G7_*.json")):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and "final" in payload and payload.get("stage") == stage:
            runs.append(payload)
    return runs


def tuning_findings(tune):
    by = {(run["task"], run["lr"]): run for run in tune}
    findings = {}
    for task in TASK_SEEDS:
        if not all((task, lr) in by for lr in LR_GRID):
            continue
        candidates = [by[(task, lr)] for lr in LR_GRID]
        findings[task] = {
            "final_primary_pct": {f"{run['lr']:g}": run["final"]["primary"] * 100 for run in candidates},
            "lowest_minus_highest_lr_pp": 100 * (candidates[0]["final"]["primary"] - candidates[-1]["final"]["primary"]),
            "epoch_primary_pct": {f"{run['lr']:g}": [epoch["primary"] * 100 for epoch in run["per_epoch"]]
                                  for run in candidates},
            "stored_losses_finite": {f"{run['lr']:g}": all(math.isfinite(value) for value in run["loss_first_steps"])
                                     for run in candidates},
            "last50_stored_loss_mean": {f"{run['lr']:g}": statistics.mean(run["loss_first_steps"][-50:])
                                       for run in candidates},
        }
    return findings


def initialization_diagnostics(runs):
    diagnostics = {}
    for task in TASK_SEEDS:
        task_runs = [run for run in runs if run["task"] == task]
        if not task_runs:
            continue
        initializations = [run["init_details"] for run in task_runs]
        layers = [layer for item in initializations for layer in item["layers"]]
        diagnostics[task] = {
            "runs": len(task_runs),
            "calibration_samples": sorted({item["calibration_samples"] for item in initializations}),
            "target_layer_counts": sorted({len(item["layers"]) for item in initializations}),
            "ranks": sorted({layer["rank"] for layer in layers}),
            "scalings": sorted({layer["scaling"] for layer in layers}),
            "logits_max_abs_error": max(item["initial_logits_max_abs_error"] for item in initializations),
            "weight_compensation_max_abs_error": max(layer["weight_compensation_max_abs_error"] for layer in layers),
            "lora_params": sorted({run["lora_params"] for run in task_runs}),
            "head_params": sorted({run["head_params"] for run in task_runs}),
            "max_abs_budget_deviation": max(abs(run["budget_dev"]) for run in task_runs),
        }
    return diagnostics


def make_plot(runs, results):
    if not runs:
        return None
    os.environ.setdefault("MPLCONFIGDIR", str(G7 / "cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharex=True)
    for ax, task in zip(axes, TASK_SEEDS):
        for run in sorted((r for r in runs if r["task"] == task), key=lambda r: r["seed"]):
            y = np.asarray(run["loss_first_steps"][:500], dtype=float)
            x = np.arange(1, len(y) + 1)
            color = f"C{run['seed']}"
            ax.plot(x, y, color=color, alpha=0.18, linewidth=0.5)
            window = min(20, len(y))
            if window:
                ax.plot(x[window - 1:], np.convolve(y, np.ones(window) / window, mode="valid"),
                        color=color, linewidth=1, label=f"seed {run['seed']}")
        ax.set(title=task.upper(), xlabel="Optimizer step", xlim=(0, 500))
        ax.grid(alpha=0.2)
        if any(r["task"] == task for r in runs):
            ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Training cross-entropy")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(results / f"first500-loss.{extension}", dpi=200)
    plt.close(fig)
    return "first500-loss.png"


def write_report(summary, tune, main, results):
    lines = ["# G7 LoRA-GA: team-protocol results", "",
             f"Completed: **{summary['main_completed']}/13 main runs**, **{len(tune)}/9 tuning runs**.", "",
             "Results use the last epoch. Accuracy and F1 below are percentages; ± denotes sample standard deviation across seeds. "
             "MRPC's primary metric is F1. Hyperparameters are selected on the internal dev split.", "",
             "| Task | Seeds | LR | Accuracy | F1 |", "|---|---:|---:|---|---|"]
    for task, item in summary["tasks"].items():
        lr = item["chosen_lr"]
        lines.append(f"| {task.upper()} | {item['completed']}/{len(TASK_SEEDS[task])} | "
                     f"{f'{lr:g}' if lr is not None else 'pending'} | {fmt(item['acc'], 100)} | "
                     f"{fmt(item['f1'], 100) if task == 'mrpc' else '—'} |")
    lines.extend(["", "## Same-seed differences", "",
                  "Differences are G7 minus the reference, in percentage points. The two-sided 95% interval uses "
                  "Student t with n−1 degrees of freedom. Completed seed intersections are listed explicitly.", "",
                  "| Reference | Task | Paired seeds | Primary difference | 95% CI |",
                  "|---|---|---|---|---|"])
    for group in ("G3", "G4"):
        for task in TASK_SEEDS:
            item = summary["paired"][group][task]
            lines.append(f"| {group} | {task.upper()} | {', '.join(map(str, item['seeds'])) or 'pending'} | "
                         f"{fmt(item['primary_pp'], signed=True)} | {interval(item['primary_pp'], signed=True)} |")
    for group in ("G3", "G4"):
        categories = {"include zero": [], "lie above zero": [], "lie below zero": []}
        for task, seeds in TASK_SEEDS.items():
            item = summary["paired"][group][task]
            if len(item["seeds"]) != len(seeds):
                continue
            lo, hi = item["primary_pp"]["ci95"]
            category = "lie above zero" if lo > 0 else "lie below zero" if hi < 0 else "include zero"
            categories[category].append(task.upper())
        clauses = [f"{', '.join(tasks)} intervals {category}" for category, tasks in categories.items() if tasks]
        if clauses:
            lines.extend(["", f"**G7−{group}:** " + "; ".join(clauses) + "."])
    ref = summary["reference"]
    lines.extend(["", f"Reference measurements: `{ref['branch']}` at `{ref['commit']}`. "
                  "The scalar source metrics and file paths are preserved in [reference_metrics.json](reference_metrics.json). "
                  "The experimental protocol is shared; hardware and library versions are recorded for each run. "
                  "G3/G4 used an A40 and torch 2.11 / Transformers 5.21 / PEFT 0.21; this G7 run uses an L40S "
                  "and the project's Python 3.9 environment (torch 2.4 / Transformers 4.52 / PEFT 0.17). "
                  "G3 redistributes ranks and its actual adapter count varies; this comparison retains those recorded counts.", "",
                  "## Tuning", "",
                  "All candidates use seed 0 and the final dev metric. Equal primary scores select the smaller learning rate.", "",
                  "| Task | LR | Final dev primary (%) | Chosen |", "|---|---:|---:|---|"])
    by_tune = {(r["task"], r["lr"]): r for r in tune}
    for task in TASK_SEEDS:
        chosen = summary["tasks"][task]["chosen_lr"]
        for lr in LR_GRID:
            run = by_tune.get((task, lr))
            value = f"{run['final']['primary'] * 100:.2f}" if run else "pending"
            lines.append(f"| {task.upper()} | {lr:g} | {value} | "
                         f"{'yes' if chosen == lr else ('no' if chosen is not None else 'pending')} |")
    findings = summary["tuning_findings"]
    if len(findings) == len(TASK_SEEDS):
        decreases = ", ".join(f"{task.upper()} {findings[task]['lowest_minus_highest_lr_pp']:.2f} pp" for task in TASK_SEEDS)
        lines.extend(["", "### Learning-rate sensitivity", "",
                      f"The 2e-4 candidate has the highest final dev primary score on all three tasks. "
                      f"Increasing LR from 2e-4 to 1e-3 reduced that score by {decreases}. "
                      "This is the observed learning-rate sensitivity of G7 under the shared α/r=1, no-clipping setting."])
        rte = findings["rte"]
        high = [by_tune[("rte", lr)] for lr in LR_GRID[1:]]
        scores = sorted({epoch["primary"] * 100 for run in high for epoch in run["per_epoch"]})
        if len(scores) <= 3:
            values = "/".join(f"{value:.2f}" for value in scores)
            losses = "/".join(f"{rte['last50_stored_loss_mean'][f'{lr:g}']:.4f}" for lr in LR_GRID[1:])
            if all(rte["stored_losses_finite"][f"{lr:g}"] for lr in LR_GRID[1:]):
                lines.extend(["", f"For RTE, the 5e-4 and 1e-3 runs stayed at {values}% dev accuracy across epochs; "
                              f"their last 50 stored-step mean losses were {losses}, with finite losses throughout the stored trace. "
                              "The 48.19/51.81% plateau is consistent with single-class prediction on the fixed dev split "
                              "(an inference from aggregate scores). The selected 2e-4 run reached "
                              f"{rte['final_primary_pct']['0.0002']:.2f}% at the required last epoch. "
                              "These records show a finite-loss learning plateau at the higher rates."])
    lines.extend(["", "## Initialisation and parameter checks", "",
                  "The following measurements cover completed tuning and main runs. Logit error compares the warmed-up base "
                  "with the compensated LoRA-GA model on the same calibration batch; weight error is the maximum across "
                  "`W_residual + (α/r)B₀A₀ − W₀` over all target layers.", "",
                  "| Task | Runs | Calibration samples | Layers / rank / scaling | LoRA / head parameters | Max budget deviation | Max logits error | Max weight error |",
                  "|---|---:|---|---|---|---:|---:|---:|"])
    for task, item in summary["initialization_diagnostics"].items():
        count = ", ".join(map(str, item["calibration_samples"]))
        geometry = " / ".join(",".join(map(str, item[key])) for key in ("target_layer_counts", "ranks", "scalings"))
        parameters = " / ".join(",".join(f"{value:,}" for value in item[key]) for key in ("lora_params", "head_params"))
        lines.append(f"| {task.upper()} | {item['runs']} | {count} | {geometry} | {parameters} | "
                     f"{item['max_abs_budget_deviation']*100:.2f}% | {item['logits_max_abs_error']:.3e} | "
                     f"{item['weight_compensation_max_abs_error']:.3e} |")
    lines.extend(["", "## Per-run results", "",
                  "Times are seconds; peak memory is PyTorch's maximum allocated GPU memory in GiB. "
                  "Total includes loading, head warm-up, initialisation, training and evaluation.", "",
                  "| Task | Seed | Accuracy (%) | F1 (%) | Init s | Total s | Peak GiB | Step-0 B gradient |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for run in sorted(main, key=lambda r: (list(TASK_SEEDS).index(r["task"]), r["seed"])):
        f1 = f"{run['final']['f1'] * 100:.2f}" if "f1" in run["final"] else "—"
        lines.append(f"| {run['task'].upper()} | {run['seed']} | {run['final']['acc'] * 100:.2f} | {f1} | "
                     f"{run['time_s']['init']:.2f} | {run['time_s']['total']:.2f} | "
                     f"{run['peak_mem_gb']:.2f} | {run['step0_grad_B']:.4g} |")
    lines.extend(["", "| Task | Mean init s | Mean total s | Mean peak GiB | LoRA parameters |",
                  "|---|---|---|---|---|"])
    for task, item in summary["tasks"].items():
        lines.append(f"| {task.upper()} | {fmt(item['init_s'])} | {fmt(item['total_s'])} | "
                     f"{fmt(item['peak_mem_gb'])} | {fmt(item['lora_params'])} |")
    rte_predictions = summary["rte_prediction_breakdown"]
    if rte_predictions:
        lines.extend(["", "## RTE prediction breakdown", "",
                      "These are measured predictions from the selected-LR main runs on all 277 official validation examples.", "",
                      "| Seed | Predicted label 0 | Predicted label 1 | Accuracy (%) |",
                      "|---|---:|---:|---:|"])
        for item in rte_predictions:
            lines.append(f"| {item['seed']} | {item['predicted_0']} | {item['predicted_1']} | {item['acc']*100:.2f} |")
        constant = [item for item in rte_predictions if min(item["predicted_0"], item["predicted_1"]) == 0]
        if constant:
            for item in constant:
                label = 1 if item["predicted_1"] else 0
                lines.extend(["", f"At the selected LR 2e-4, seed {item['seed']} predicted label {label} for every "
                              f"validation example ({item['predicted_0'] + item['predicted_1']}/277), yielding "
                              f"{item['acc']*100:.2f}% accuracy. This is a directly observed single-class prediction failure. "
                              f"Its stored first-500-step losses are finite, with a mean of {item['last50_stored_loss_mean']:.4f} "
                              "over the final 50 stored steps."])
        if len(rte_predictions) == len(TASK_SEEDS["rte"]):
            ordered = sorted(rte_predictions, key=lambda item: item["acc"])
            stat = summary["tasks"]["rte"]["acc"]
            lines.extend(["", f"RTE accuracy spans {ordered[0]['acc']*100:.2f}–{ordered[-1]['acc']*100:.2f}%, "
                          f"with mean ± SD {fmt(stat, 100)}%. The two lowest results are seed {ordered[0]['seed']} "
                          f"({ordered[0]['acc']*100:.2f}%) and seed {ordered[1]['seed']} ({ordered[1]['acc']*100:.2f}%). "
                          "These measured low-performing runs contribute to the low mean and large seed variation."])
    if summary.get("scheduler"):
        scheduler = summary["scheduler"]
        lines.extend(["", "## Execution and verification", "",
                      f"Slurm job **{scheduler['job_id']}** finished **{scheduler['state']}**, exit code "
                      f"`{scheduler['exit_code']}`, in **{scheduler['elapsed_seconds']} seconds**. "
                      f"{scheduler['processes_on_single_gpu']} processes shared {scheduler['gpus']} allocated "
                      f"{scheduler['gpu']} GPU, with {scheduler['cpus']} CPUs and {scheduler['memory_gib']} GiB host memory. "
                      "[Scheduler record](scheduler.json)."])
    if summary.get("verification"):
        verification = summary["verification"]
        lines.extend(["", f"Verification recomputed all **{verification['predictions_recomputed']:,}** predictions "
                      f"from **{verification['main_runs']}** main runs, checked label order and the shared configuration, "
                      f"and confirmed finite first-500-step traces for all **{verification['finite_first500_traces']}** runs. "
                      "[Verification record](verification.json)."])
    if summary["loss_plot"]:
        lines.extend(["", "## First 500 steps", "",
                      "Thin lines show individual batches; thick lines show a 20-step moving mean. "
                      "Each task has its own y-axis. Curves include main runs at the selected learning rate.", "",
                      "![First 500 training steps](first500-loss.png)", "",
                      "[PDF figure](first500-loss.pdf)"])
    lines.extend(["", "[Protocol and reproduction instructions](../README.md)", ""])
    (results / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=G7 / "results")
    args = parser.parse_args()
    results = args.results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    references = reference_snapshot(results)
    tune, main_runs = read_runs(results, "tune"), read_runs(results, "main")
    chosen_path = results / "chosen_lr.json"
    chosen = json.loads(chosen_path.read_text()) if chosen_path.exists() else {}
    summary = {"group": "G7", "main_completed": len(main_runs), "main_expected": 13,
               "tune_completed": len(tune), "tune_expected": 9,
               "status": "completed" if len(main_runs) == 13 and len(tune) == 9 else "in_progress",
               "reference": {key: references[key] for key in ("repository", "branch", "commit", "attribution")},
               "tuning_findings": tuning_findings(tune),
               "initialization_diagnostics": initialization_diagnostics(tune + main_runs),
               "tasks": {}, "paired": {"G3": {}, "G4": {}}}
    summary["rte_prediction_breakdown"] = []
    for run in sorted((item for item in main_runs if item["task"] == "rte"), key=lambda item: item["seed"]):
        predictions = run["predictions"]["predictions"]
        summary["rte_prediction_breakdown"].append({
            "seed": run["seed"], "predicted_0": predictions.count(0), "predicted_1": predictions.count(1),
            "acc": run["final"]["acc"],
            "losses_finite": all(math.isfinite(value) for value in run["loss_first_steps"]),
            "last50_stored_loss_mean": statistics.mean(run["loss_first_steps"][-50:]),
        })
    for name in ("scheduler", "verification"):
        path = results / f"{name}.json"
        if path.exists():
            summary[name] = json.loads(path.read_text())
    for task, seeds in TASK_SEEDS.items():
        runs = sorted((r for r in main_runs if r["task"] == task), key=lambda r: r["seed"])
        by_seed = {r["seed"]: r for r in runs}
        if len(by_seed) != len(runs) or not set(by_seed).issubset(seeds):
            raise ValueError(f"Unexpected or duplicate seeds for {task}")
        summary["tasks"][task] = {
            "completed": len(runs), "seeds": sorted(by_seed), "chosen_lr": chosen.get(f"G7_{task}"),
            "acc": describe([r["final"]["acc"] for r in runs]),
            "f1": describe([r["final"]["f1"] for r in runs if "f1" in r["final"]]),
            "primary": describe([r["final"]["primary"] for r in runs]),
            "init_s": describe([r["time_s"]["init"] for r in runs]),
            "total_s": describe([r["time_s"]["total"] for r in runs]),
            "peak_mem_gb": describe([r["peak_mem_gb"] for r in runs]),
            "lora_params": describe([r["lora_params"] for r in runs]),
        }
        for group in ("G3", "G4"):
            baseline = {r["seed"]: r for r in references["runs"] if r["group"] == group and r["task"] == task}
            paired_seeds = sorted(set(by_seed) & set(baseline))
            metrics = ("acc", "f1", "primary") if task == "mrpc" else ("acc", "primary")
            summary["paired"][group][task] = {
                "seeds": paired_seeds,
                **{metric + "_pp": describe([by_seed[s]["final"][metric] - baseline[s]["final"][metric]
                                             for s in paired_seeds], percentage_points=True) for metric in metrics},
            }
    fields = ["run_id", "stage", "group", "task", "lr", "seed", "eval_split", "acc", "f1", "primary",
              "step0_grad_B", "lora_params", "head_params", "budget_dev", "init_s", "total_s", "peak_mem_gb", "slurm_job"]
    with (results / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run in tune + main_runs:
            row = {key: run[key] for key in fields if key in run}
            row.update(run["final"])
            row.update(init_s=run["time_s"]["init"], total_s=run["time_s"]["total"])
            writer.writerow(row)
    summary["loss_plot"] = make_plot(main_runs, results)
    write_json(results / "summary.json", summary)
    write_report(summary, tune, main_runs, results)
    print(json.dumps({key: summary[key] for key in ("status", "main_completed", "tune_completed")}))


if __name__ == "__main__":
    main()
