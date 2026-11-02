"""Write the command queue for one stage of G3/G4 to jobs/queues/<stage>.txt.

    python jobs/make_queue.py pilot | tune | main | lowres
main and lowres read the tuning results to pick each (group, task)'s learning rate.
"""
import glob
import json
import os
import sys

LR_GRID = [2e-4, 5e-4, 1e-3]          # team-wide grid: confirm with the whole team before the tuning stage
TUNE_SEED = 0
SEEDS = {"sst2": [0, 1, 2], "mrpc": [0, 1, 2, 3, 4], "rte": [0, 1, 2, 3, 4]}   # shared with all groups
LOWRES_SEEDS = [0, 1, 2, 3, 4]
TASKS = ["sst2", "mrpc", "rte"]        # SST-2 first: the longest runs start first
GROUPS = ["G3", "G4"]


def cmd(stage, task, group, lr, seed):
    rid = f"{stage}_{group}_{task}_lr{lr:g}_s{seed}"
    return (f"python src/train.py --stage {stage} --task {task} --group {group} --lr {lr:g} --seed {seed}"
            f" > logs/{rid}.log 2>&1")


def pick_best(runs):
    """Selection rule used everywhere: highest dev primary metric; a tie goes to the smaller learning rate."""
    return max(runs, key=lambda r: (r["final"]["primary"], -r["lr"]))


def best_lr(group, task):
    runs = [json.load(open(p)) for p in glob.glob(f"results/tune/tune_{group}_{task}_*.json")]
    if len(runs) != len(LR_GRID):
        sys.exit(f"tuning incomplete for {group} {task}: {len(runs)}/{len(LR_GRID)} runs")
    return pick_best(runs)["lr"]   # dev split only


def main(stage):
    if stage == "pilot":
        lines = [cmd("pilot", t, "G3", LR_GRID[1], TUNE_SEED) for t in TASKS] + [cmd("pilot", "rte", "G4", LR_GRID[1], TUNE_SEED)]
    elif stage == "tune":
        lines = [cmd("tune", t, g, lr, TUNE_SEED) for t in TASKS for g in GROUPS for lr in LR_GRID]
    elif stage == "main":
        lrs = {f"{g}_{t}": best_lr(g, t) for g in GROUPS for t in TASKS}
        os.makedirs("results", exist_ok=True)
        json.dump(lrs, open("results/chosen_lr.json", "w"), indent=1)
        print("chosen learning rates:", lrs)
        lines = [cmd("main", t, g, lrs[f"{g}_{t}"], s) for t in TASKS for g in GROUPS for s in SEEDS[t]]
    elif stage == "lowres":
        lines = [cmd("lowres", "sst2", "G3", best_lr("G3", "sst2"), s) for s in LOWRES_SEEDS]
    else:
        sys.exit(f"unknown stage {stage}")
    os.makedirs("jobs/queues", exist_ok=True)
    with open(f"jobs/queues/{stage}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"jobs/queues/{stage}.txt: {len(lines)} runs")


if __name__ == "__main__":
    main(sys.argv[1])
