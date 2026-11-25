#!/bin/bash
source /home/claude/bakeoff/river/harness/env.sh
cd /home/claude/bakeoff/river/arms_filt
R0='{"sigma_S": 0.1, "inflation": 0.9, "sigma_obs": 0.3, "sigma_P": 0.2}'
python3 tools/make_cand.py r0 "$R0" --eval > logs/eval_r0.log 2>&1
V='[{"augment": ["pmult"], "augment_init_sd": {"pmult": 0.15}, "augment_rw_sd": {"pmult": 0.005}},
 {"augment": ["pmult"], "augment_init_sd": {"pmult": 0.25}, "augment_rw_sd": {"pmult": 0.02}},
 {"augment": ["pmult", "K2"], "augment_init_sd": {"pmult": 0.15}, "augment_rw_sd": {"pmult": 0.005, "K2": 0.01}},
 {"augment": ["pmult", "K1", "K2"], "augment_init_sd": {"pmult": 0.15}, "augment_rw_sd": {"pmult": 0.005, "K1": 0.01, "K2": 0.01}},
 {"augment": ["pmult", "FC", "BETA", "K1", "K2"], "augment_init_sd": {"pmult": 0.15}, "augment_rw_sd": {"pmult": 0.005, "FC": 0.01, "BETA": 0.01, "K1": 0.01, "K2": 0.01}},
 {"augment": ["pmult", "FC", "BETA", "K1", "K2"], "augment_init_sd": {"pmult": 0.15}, "augment_rw_sd": {"pmult": 0.002, "FC": 0.003, "BETA": 0.003, "K1": 0.003, "K2": 0.003}},
 {"augment": ["pmult", "FC", "BETA", "K1", "K2"], "augment_init_sd": {"pmult": 0.15}, "augment_rw_sd": {"pmult": 0.02, "FC": 0.03, "BETA": 0.03, "K1": 0.03, "K2": 0.03}}]'
python3 tools/tune.py aug1 "$R0" "$V" 250 > logs/tune_aug1.log 2>&1
