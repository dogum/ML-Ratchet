"""__NAME__ on a few real exam examples (a fixed random draw within strata — never picked by result).
Runs in seconds with numpy only:  python demo.py"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from __ID__ import __NAME__  # noqa: E402

if __name__ == "__main__":
    model = __NAME__()
    files = sorted(glob.glob(os.path.join(HERE, "examples", "*.npz")))
    if not files:
        print("No examples yet: add a few real exam examples to examples/ (see examples/README.md).")
    for f in files:
        print(os.path.basename(f), model.read(f))
