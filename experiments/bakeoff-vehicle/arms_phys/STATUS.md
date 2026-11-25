# arms_phys (agent C: physics, formulas, conformal layer): STATUS

Updated 2026-11-23 17:05. DONE (agent C). Nothing running. Source `arms_phys/env.sh` (harness env + own NUMBA_CACHE_DIR + src/ on PYTHONPATH).
Nothing of mine needs the exam; I never ran it.

## Registered (hash prefix)
- `phys_anchor` c646d0fb9e424d1b (table + trace): E_phys alone + sim bands. r0 released; r1 sim scale / r2 dev scale
  rejected (D16 worse by 0.024 / 0.014).
- `formula` a5691cba86f7ee5c (table + trace): r1 released (s1 most accurate per group): sim val skill 0.323, MdAPE 0.160,
  dev D16 0.1752. r0 knees 0.231 / 0.206 base; r2 (+dyn) 0.322 / 0.1518 rejected (skill +0.000).
- conformal wrappers: `enet_v21_conformal` 95a0d343, `phys_anchor_conformal` 454013bd, `formula_conformal` 22f1b89d,
  `tab_gbm_conformal` dae72a03, `tab_mlp_conformal` ad2f61e5. `python3 conformal/summary.py` prints all coverages.
- `tab_tabpfn_conformal` e07f73a7 (its own r1 dev file failed verification, max rel diff 0.018 on smoke rows, so the
  wrapper used a fresh evaluate.py dev pass).
- Still to wrap when they register: tab_gp (agent A), cnn / gru / transformer / s4d (agent B), anything later.
  If a base arm is re-registered (--replace), its wrapper's hash changes: wrap_pending.py refits and re-registers it.

## For the coordinator: wrap later arms
```
source /home/claude/bakeoff/vehicle/arms_phys/env.sh
python3 /home/claude/bakeoff/vehicle/arms_phys/conformal/wrap_pending.py          # list
python3 /home/claude/bakeoff/vehicle/arms_phys/conformal/wrap_pending.py --run    # fit + register each (one at a time)
```
Reuses an arm's own `work/**/<released>_<track>_dev_preds.parquet` after verifying it on the 72 smoke records,
else one dev pass with evaluate.py. Failed wrappers move to conformal/failed/ (cannot block the exam).

## LawFinder export: lawfinder_road/ (MERGE_NOTES.md first)

## Layout
- src/: lawvars (law variables, table == trace), lawsearch (vendored + log_target), search_law, ladder, build_formula,
  rungs_formula, fit_anchor, fscore (in-process scorer = evaluate.py's numbers), build_lawfinder, run_s{1,2,3}.sh.
- work/search/<tag>_<group>.json fronts; work/ladder_<tag>_<group>.json scored fronts; work/rungs_formula.json.
- conformal/: lib/vcf_mondrian.py, fit_wrapper.py, verify_preds.py, wrap_pending.py, summary.py, base_preds/, <x>_conformal/.
