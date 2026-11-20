"""Write the command queue for one stage of G1/G2 to jobs/queues/<stage>.txt (copied from G3G4/jobs/make_queue.py).

    python jobs/make_queue.py lowres_ctl   # extra: G1 low-resource at lr 1e-3 -> results/control/lowres/
    python jobs/make_queue.py seeds10      # extra: G1 seeds 5-9 on MRPC, RTE, low resource -> results/seeds10/
    python jobs/make_queue.py g3_lowres_ctl  # extra: why EVA hurts in low resource (G3 runs, needs ../G3G4/src)

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
EXTRA_SEEDS = [5, 6, 7, 8, 9]   # seeds10 stage: brings MRPC, RTE and low resource to 10 seeds
TASKS = ["sst2", "mrpc", "rte"]        # SST-2 first: the longest runs start first
GROUPS = ["G1", "G2"]


def cmd(stage, task, group, lr, seed, out="results"):
    rid = f"{stage}_{group}_{task}_lr{lr:g}_s{seed}"
    return (f"python src/train.py --stage {stage} --task {task} --group {group} --lr {lr:g} --seed {seed}"
            f" --out {out} > logs/{rid}.log 2>&1")


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
        lines = [cmd("pilot", "mrpc", g, LR_GRID[1], TUNE_SEED) for g in GROUPS]
    elif stage == "tune":
        lines = [cmd("tune", t, g, lr, TUNE_SEED) for t in TASKS for g in GROUPS for lr in LR_GRID]
    elif stage == "main":
        lrs = {f"{g}_{t}": best_lr(g, t) for g in GROUPS for t in TASKS}
        os.makedirs("results", exist_ok=True)
        json.dump(lrs, open("results/chosen_lr.json", "w"), indent=1)
        print("chosen learning rates:", lrs)
        lines = [cmd("main", t, g, lrs[f"{g}_{t}"], s) for t in TASKS for g in GROUPS for s in SEEDS[t]]
    elif stage == "lowres":
        # proposal: low-resource SST-2 for G1, G3, G5 only; lr = the one G1 picked on full SST-2
        lines = [cmd("lowres", "sst2", "G1", best_lr("G1", "sst2"), s) for s in LOWRES_SEEDS]
    elif stage == "lowres_ctl":
        # learning-rate control, outside the shared protocol: G1 low-resource at G3's lr (1e-3), so that the
        # G1-vs-G3 low-resource gap is not confounded by lr. Kept in results/control/ to stay apart from lowres.
        lines = [cmd("lowres", "sst2", "G1", 1e-3, s, out="results/control") for s in LOWRES_SEEDS]
    elif stage == "seeds10":
        # more seeds for G1 where the intervals are widest, at the learning rates already chosen (no new tuning).
        # Kept in results/seeds10/ so the shared tables stay on seeds 0-4 until the other groups catch up.
        lrs = json.load(open("results/chosen_lr.json"))
        lines = [cmd("main", t, "G1", lrs[f"G1_{t}"], s, out="results/seeds10") for t in ("mrpc", "rte") for s in EXTRA_SEEDS]
        lines += [cmd("lowres", "sst2", "G1", lrs["G1_sst2"], s, out="results/seeds10") for s in EXTRA_SEEDS]
    elif stage == "g3_lowres_ctl":
        # Low-resource G3 (EVA) controls, run with G3G4's code through jobs/g3_control.py:
        #   repro: G3 as in the shared protocol (lr 1e-3, rho 2), rerun on L40S as the reference for the two below
        #   lr:    G3 at lr 5e-4 (G1's choice) and 2e-4, rho 2: is the gap a learning-rate effect for EVA?
        #   rho1:  EVA directions with uniform r = 8 (rho 1) at lr 1e-3: direction vs rank redistribution
        def g3(lr, s, rho, out, cache):
            rid = f"lowres_G3_sst2_lr{lr:g}_s{s}_rho{rho:g}"
            return (f"python jobs/g3_control.py --rho {rho:g} --stage lowres --task sst2 --group G3 --lr {lr:g} --seed {s}"
                    f" --out {out} --cache {cache} > logs/{rid}.log 2>&1")
        lines = [g3(1e-3, s, 2, "results/control_repro", "cache_g3") for s in LOWRES_SEEDS]
        lines += [g3(lr, s, 2, "results/control", "cache_g3") for lr in (5e-4, 2e-4) for s in LOWRES_SEEDS]
        lines += [g3(1e-3, s, 1, "results/control_rho1", "cache_g3_rho1") for s in LOWRES_SEEDS]
    else:
        sys.exit(f"unknown stage {stage}")
    os.makedirs("jobs/queues", exist_ok=True)
    with open(f"jobs/queues/{stage}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"jobs/queues/{stage}.txt: {len(lines)} runs")


if __name__ == "__main__":
    main(sys.argv[1])
