"""Run G3G4's unchanged train.py with a different EVA rho, for the G3 low-resource controls.

    python jobs/g3_control.py --rho 1 --stage lowres --task sst2 --group G3 --lr 1e-3 --seed 0 --out results/control_rho1

Everything except --rho is passed on to ../G3G4/src/train.py (the copy on main / summary). rho = 1 keeps r = 8 on every
layer, so the run uses EVA's directions without its rank redistribution and hits the parameter budget exactly.
The value used ends up in the JSON's cfg.rho.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "G3G4", "src"))
import train  # noqa: E402  (G3G4's train.py)

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--rho" in argv:
        i = argv.index("--rho")
        train.CFG["rho"] = float(argv[i + 1])
        del argv[i:i + 2]
    sys.argv = [train.__file__] + argv
    train.main()
