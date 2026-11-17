#!/bin/sh
# eval_rung.sh WEIGHTS TAG [SHIELD]: policy on val and dev
python3 scripts/evaluate.py val policy $1 $2 $3
python3 scripts/evaluate.py dev policy $1 $2 $3
