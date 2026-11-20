"""Zip everything to send back: result JSONs, G8's initial A matrices (cache/g8A_*) and the logs.

    python jobs/pack_results.py g8_all_<jobid>.zip      # g8.sbatch calls this at the end of every job
"""
import glob
import sys
import zipfile

files = sorted(glob.glob("results/**/*.json", recursive=True) + glob.glob("cache/g8A_*.pt") + glob.glob("logs/*"))
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p)
print(f"wrote {sys.argv[1]} ({len(files)} files)")
