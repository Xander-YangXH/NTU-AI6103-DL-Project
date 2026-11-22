"""Smoke test: run every group's real training entry point end to end on a small slice of the data.

    python tools/smoke_test.py                       # every group this machine can run
    python tools/smoke_test.py --groups G3 G8        # selected groups
    python tools/smoke_test.py --groups H1           # H1 diagnosis (GPU, about 10-30 minutes)

What it does. Each group's own code is imported unchanged and run through its normal main(), in a subprocess per
group (the folders share module names such as data.py and train.py). Only the amount of work is reduced: the data
splits are cut to a few dozen examples, every stage runs one epoch, the calibration set is small and the result
JSON goes to a temporary folder. The check then reads the JSON the run wrote and verifies its key fields
(final metric, step-0 gradient on B, LoRA parameter count, loss curve). Numbers from a smoke run are meaningless;
it only shows that the code, the environment and the data folder work together.

Requirements: the environment in requirements.txt (see RUN.md) and the data prepared as described there.
  - G1-G4, G8, G9 run on CPU or GPU and read GWEVA_DATA.
  - G7, G5/G6 and H1 are GPU-only by design. On a machine without CUDA, G5/G6 run with the CUDA-only calls stubbed
    (API check only) and G7/H1 are skipped.
  - G5/G6 read G5G6/models and G5G6/data (python G5G6/jobs/prefetch.py); H1 reuses them via GWEVA_ROOT.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = {"G1": "G1G2", "G2": "G1G2", "G3": "G3G4", "G4": "G3G4", "G7": "G7", "G8": "G8", "G9": "G9",
          "G5G6": "G5G6", "H1": "H1 Diagnosis"}
DEFAULT = ["G1", "G2", "G3", "G4", "G8", "G9", "G7", "G5G6"]
SIZES = {"train": 160, "dev": 48, "val": 64, "lowres": 160}   # examples kept per split in a smoke run


def has_cuda():
    import torch
    return torch.cuda.is_available()


def stub_cuda():
    """Make the CUDA-only bookkeeping calls harmless on a CPU machine (G5/G6 API check without a GPU)."""
    import torch
    for name, value in dict(reset_peak_memory_stats=None, max_memory_allocated=0, synchronize=None,
                            empty_cache=None, manual_seed_all=None, get_rng_state_all=[], set_rng_state_all=None,
                            get_device_name="cpu (CUDA calls stubbed)").items():
        setattr(torch.cuda, name, lambda *a, _v=value, **k: _v)
    torch.Tensor.cuda = lambda self, *a, **k: self


def cut(ds, n):
    return ds.select(range(min(n, len(ds))))


# ---------------------------------------------------------------- groups sharing G3G4's pipeline (G1-G4, G8, G9)
def run_shared(group, task, out):
    src = ROOT / FOLDER[group] / "src"
    sys.path.insert(0, str(src))
    import data
    import train
    original = data.load_splits

    def small_splits(task, tokenizer, max_len):
        sp = original(task, tokenizer, max_len)
        return {k: (cut(v, SIZES[k]) if k in SIZES else v) for k, v in sp.items()}
    data.load_splits = small_splits
    train.CFG.update(epochs={k: 1 for k in train.CFG["epochs"]}, calib_size=64, num_workers=0)
    lr = "2e-4"
    sys.argv = ["train.py", "--stage", "main", "--task", task, "--group", group, "--lr", lr, "--seed", "0",
                "--out", str(out / "results"), "--cache", str(out / "cache")]
    train.main()
    path = next((out / "results" / "main").glob("*.json"))
    return json.loads(path.read_text())


def run_g7(task, out):
    if not has_cuda():
        return "skip: G7 (LoRA-GA) calibrates full-weight gradients on the GPU by design and refuses to run on CPU"
    return run_shared("G7", task, out)


# ---------------------------------------------------------------- G5/G6: the gw_eva study pipeline
def run_g5g6(task, out):
    g56 = ROOT / "G5G6"
    if not (g56 / "models/roberta-base/model.safetensors").exists():
        raise RuntimeError("G5G6/models/roberta-base is missing: run `python G5G6/jobs/prefetch.py` first (see RUN.md)")
    if not has_cuda():
        stub_cuda()
    sys.path.insert(0, str(g56 / "src"))
    from gw_eva import study
    original = study.prepare

    def small_prepare(t):
        data, collator, split, sha = original(t)
        return data, collator, {"fit": split["fit"][:SIZES["train"]], "selection": split["selection"][:SIZES["dev"]]}, sha
    study.prepare = small_prepare
    study.STUDY = out / "shared_study"
    study.TASKS = {task: dict(epochs=1, batch=32, seeds=1, selection_epochs=1)}
    study.LRS = [2e-4]
    sys.argv = ["study", "--tasks", task]
    study.main()
    results = sorted(study.STUDY.glob(f"{task}/formal/*/seed0/final/result.json"))
    assert {p.parts[-4] for p in results} == {"G5", "G6"}, results
    rows = [json.loads(p.read_text()) for p in results]
    for r in rows:   # map onto the shared JSON fields checked below
        r.update(final={"primary": r["metrics"]["selection_score"]}, lora_params=None)
    return rows


# ---------------------------------------------------------------- H1: needs G5/G6's head checkpoint for the task
def run_h1(task, out):
    if not has_cuda():
        return "skip: H1 sketches 3 x 1,024 examples with per-token gradients; run it on a GPU"
    g56, h1 = ROOT / "G5G6", ROOT / "H1 Diagnosis"
    os.environ["GWEVA_ROOT"] = str(g56)   # reuse G5G6's pinned model and data instead of a second download
    sys.path[:0] = [str(h1 / "src"), str(h1 / "scripts")]
    from gw_eva import study
    import run_h1
    study.STUDY = out / "shared_study"
    study.context(task, 0, "fit")          # one-epoch head warm-up on the full split, as in the shared runs
    run_h1.run(task, 0, study.STUDY, out / "h1")
    r = json.loads((out / "h1" / f"{task}_seed0.json").read_text())
    return {"final": {"primary": r["mean_eva_gw"]}, "h1": {"eva_gw": r["mean_eva_gw"], "gw_gw": r["mean_gw_gw"],
                                                            "layers": len(r["per_layer"])}}


def check(group, res):
    rows = res if isinstance(res, list) else [res]
    lines = []
    for r in rows:
        if "h1" in r:
            assert r["h1"]["layers"] == 72 and 0 <= r["h1"]["eva_gw"] <= 1 and 0 <= r["h1"]["gw_gw"] <= 1, r["h1"]
            lines.append(f"EVA-GW {r['h1']['eva_gw']:.3f}, GW-GW {r['h1']['gw_gw']:.3f} over 72 layers")
            continue
        p = r["final"]["primary"]
        assert 0 <= p <= 1, p
        g0 = r.get("step0_grad_B")
        assert g0 is not None and math.isfinite(g0) and g0 > 0, g0
        assert r.get("loss_first_steps") and all(math.isfinite(x) for x in r["loss_first_steps"])
        if r.get("lora_params") is not None:
            assert r["lora_params"] > 0
        name = r.get("group", group)
        lines.append(f"{name}: primary {p:.3f}, step-0 |dL/dB| {g0:.3g}, {len(r['loss_first_steps'])} loss points"
                     + (f", {r['lora_params']:,} LoRA params" if r.get("lora_params") else ""))
    return "; ".join(lines)


def child(group, task, out):
    import warnings
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    out = Path(out)
    if group == "G7":
        res = run_g7(task, out)
    elif group == "G5G6":
        res = run_g5g6(task, out)
    elif group == "H1":
        res = run_h1(task, out)
    else:
        res = run_shared(group, task, out)
    if isinstance(res, str):
        print("SMOKE_RESULT", json.dumps({"status": "skipped", "detail": res}))
        return
    print("SMOKE_RESULT", json.dumps({"status": "passed", "detail": check(group, res)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=DEFAULT, choices=list(FOLDER))
    ap.add_argument("--task", default="rte", choices=["sst2", "mrpc", "rte"])
    ap.add_argument("--keep", action="store_true", help="keep the temporary output folders")
    ap.add_argument("--child", nargs=3, metavar=("GROUP", "TASK", "OUT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return child(*args.child)

    import torch
    print(f"python {sys.version.split()[0]}, torch {torch.__version__}, "
          f"device {'cuda: ' + torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}")
    print(f"GWEVA_DATA = {os.path.expanduser(os.environ.get('GWEVA_DATA', '~/gweva-data'))}\n")
    summary, failed = [], False
    for g in args.groups:
        out = Path(tempfile.mkdtemp(prefix=f"smoke_{g}_"))
        started = time.time()
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", g, args.task, str(out)], cwd=ROOT / FOLDER[g],
                              capture_output=True, text=True)
        secs = time.time() - started
        line = next((l for l in proc.stdout.splitlines() if l.startswith("SMOKE_RESULT ")), None)
        if proc.returncode == 0 and line:
            r = json.loads(line[len("SMOKE_RESULT "):])
            status, detail = r["status"].upper(), r["detail"]
        else:
            failed = True
            status = "FAILED"
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-15:]
            detail = "\n      ".join(tail)
        summary.append((g, status, secs, detail))
        print(f"[{status}] {g} ({secs:.0f}s): {detail}", flush=True)
        if not args.keep:
            import shutil
            shutil.rmtree(out, ignore_errors=True)
    print("\nSummary:")
    for g, status, secs, _ in summary:
        print(f"  {g:5s} {status:8s} {secs:6.0f}s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
