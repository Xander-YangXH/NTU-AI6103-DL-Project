"""Write the command queue for one stage of G8 to jobs/queues/<stage>.txt.

    python jobs/make_queue.py pilot | main | lowres
G8 has no tuning stage: it reuses G3's learning rates (G3G4/results/chosen_lr.json on branch yinchenyu_g3g4),
as the shared settings in the main README require.
lowres (optional, beyond the proposal): SST-2 1k subset, 5 seeds, G3's SST-2 rate, exactly as G3's lowres stage.
"""
import os
import sys

G3_LR = {"sst2": 1e-3, "mrpc": 2e-4, "rte": 2e-4}   # copied from G3G4/results/chosen_lr.json
SEEDS = {"sst2": [0, 1, 2], "mrpc": [0, 1, 2, 3, 4], "rte": [0, 1, 2, 3, 4]}   # shared with all groups
LOWRES_SEEDS = [0, 1, 2, 3, 4]
TASKS = ["sst2", "mrpc", "rte"]        # SST-2 first: the longest runs start first


def cmd(stage, task, lr, seed):
    rid = f"{stage}_G8_{task}_lr{lr:g}_s{seed}"
    return f"python src/train.py --stage {stage} --task {task} --group G8 --lr {lr:g} --seed {seed} > logs/{rid}.log 2>&1"


def main(stage):
    if stage == "pilot":
        lines = [cmd("pilot", "rte", G3_LR["rte"], 0)]
    elif stage == "main":
        lines = [cmd("main", t, G3_LR[t], s) for t in TASKS for s in SEEDS[t]]
    elif stage == "lowres":
        lines = [cmd("lowres", "sst2", G3_LR["sst2"], s) for s in LOWRES_SEEDS]
    else:
        sys.exit(f"unknown stage {stage}")
    os.makedirs("jobs/queues", exist_ok=True)
    with open(f"jobs/queues/{stage}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"jobs/queues/{stage}.txt: {len(lines)} runs")


if __name__ == "__main__":
    main(sys.argv[1])
