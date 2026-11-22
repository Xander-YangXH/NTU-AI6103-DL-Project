"""Run the team's G7 tuning and main stages, two processes on one Slurm GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GRID = [2e-4, 5e-4, 1e-3]
SEEDS = {"sst2": range(3), "mrpc": range(5), "rte": range(5)}


def run_stage(stage, settings):
    pending = iter(settings)
    active = []
    exhausted = False
    failed = []
    while active or not exhausted:
        while len(active) < 2 and not exhausted and not failed:
            try:
                task, lr, seed = next(pending)
            except StopIteration:
                exhausted = True
                break
            run_id = f"{stage}_G7_{task}_lr{lr:g}_s{seed}"
            if (ROOT / "results" / stage / (run_id + ".json")).exists():
                continue
            stream = (ROOT / "logs" / (run_id + ".log")).open("w")
            command = [sys.executable, "-u", "src/train.py", "--stage", stage, "--task", task,
                       "--group", "G7", "--lr", str(lr), "--seed", str(seed)]
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            active.append((process, stream, run_id))
            print(f"Started {run_id} pid={process.pid}", flush=True)
        remaining = []
        for process, stream, run_id in active:
            code = process.poll()
            if code is None:
                remaining.append((process, stream, run_id))
            else:
                stream.close()
                print(f"Finished {run_id} exit={code}", flush=True)
                if code:
                    failed.append((run_id, code))
        active = remaining
        if failed and not active:
            raise RuntimeError(f"G7 runs failed: {failed}; inspect logs")
        if active:
            time.sleep(1)


def main():
    (ROOT / "logs").mkdir(exist_ok=True)
    run_stage("tune", [(task, lr, 0) for task in SEEDS for lr in GRID])
    chosen = {}
    for task in SEEDS:
        runs = [json.loads((ROOT / "results/tune" / f"tune_G7_{task}_lr{lr:g}_s0.json").read_text())
                for lr in GRID]
        best = max(runs, key=lambda run: (run["final"]["primary"], -run["lr"]))
        chosen["G7_" + task] = best["lr"]
    (ROOT / "results/chosen_lr.json").write_text(json.dumps(chosen, indent=2) + "\n")
    print(f"Selected learning rates: {chosen}", flush=True)
    run_stage("main", [(task, chosen["G7_" + task], seed) for task, seeds in SEEDS.items() for seed in seeds])
    subprocess.run([sys.executable, "jobs/summarize.py"], cwd=ROOT, check=True)
    (ROOT / "results/complete.json").write_text(json.dumps({
        "status": "completed", "tune_runs": 9, "main_runs": 13,
        "slurm_job": os.environ["SLURM_JOB_ID"], "chosen_lr": chosen,
    }, indent=2) + "\n")
    print("G7 completed: 9 tuning runs and 13 main runs", flush=True)


if __name__ == "__main__":
    main()
