"""Round 2: run the harness's evaluate.py unchanged, with the frozen sim-val set taken from a round-2 bank.

usage: python eval2.py <bank> <arm_dir> [evaluate.py options ...]
       e.g. eval2.py b2a work/cand/b2a_lstm_r1_s0 --tag own --out work/eval/b2a_lstm_r1_s0
The dev gate is the harness's (52 dev basins, WY2001-2014); only the sim-val bank changes.
"""
import sys

import r2

r2.use_bank(sys.argv[1])
import evaluate  # noqa: E402  (harness/evaluate.py)

sys.argv = ['evaluate.py'] + sys.argv[2:]
evaluate.main()
