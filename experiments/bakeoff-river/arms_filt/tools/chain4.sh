#!/bin/bash
source /home/claude/bakeoff/river/harness/env.sh
cd /home/claude/bakeoff/river/arms_filt
python3 tools/make_cand.py r2 '{"sigma_S": 0.1, "inflation": 0.9, "sigma_obs": 0.3, "sigma_P": 0.2, "param_scatter": 1.0, "augment": ["pmult", "FC", "BETA", "K1", "K2", "PERC", "K0", "LP", "UZL"], "augment_init_sd": {"pmult": 0.15}, "augment_rw_sd": {"pmult": 0.003, "FC": 0.005, "BETA": 0.005, "K1": 0.005, "K2": 0.005, "PERC": 0.005, "K0": 0.005, "LP": 0.005, "UZL": 0.005}, "c_frac": 0.2}' --eval > logs/eval_r2.log 2>&1
python3 tools/make_cand.py openloop '{"sigma_S": 0.1, "inflation": 0.9, "sigma_obs": 1000000.0, "sigma_P": 0.2}' > logs/eval_openloop.log 2>&1
python3 /home/claude/bakeoff/river/harness/evaluate.py enkf_watersim/work/cand_openloop --sets dev --tag openloop --out enkf_watersim/work/eval >> logs/eval_openloop.log 2>&1
