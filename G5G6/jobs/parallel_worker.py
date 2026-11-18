import fcntl
import sys
from pathlib import Path
from gw_eva import study

name = sys.argv.pop(1)
write = study.write
summarize = study.summarize

def write_worker(path, value):
    path = Path(path)
    if path.name == "active.json":
        path = path.with_name(f"active_{name}.json")
    if path.name == "pipeline.json":
        path = path.with_name(f"pipeline_{name}.json")
    write(path, value)

def summarize_locked():
    with (study.STUDY / "summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        summarize()

study.write = write_worker
study.summarize = summarize_locked
study.main()
