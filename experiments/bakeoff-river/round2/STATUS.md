# River Bake-off round 2: STATUS

Dated entries (local time, Windows 11, 16 cores; at most 6 processes). Plan and pass marks: `PREREG.md`.
Exam 2 has NOT been run and must not be run by an agent: the owner runs `exam2.py --run` once.

## Layout
- `r2.py` path layer (harness on sys.path, `config.BANK` -> `work/banks/<name>`); `bank2.py` bank builder (p1 via
  harness `build_bank.make_sample`, b2 options in `make_sample_b2`); `phys/<bank>.json` recipes.
- `train_seq.py` (path-only copy of `arms_seq/train/train.py`), `eval2.py` (harness `evaluate.py` on a round-2 bank).
- `work/` (git-ignored): banks, checkpoints, candidates, eval outputs, logs.

## If the session restarted
- Banks: `python bank2.py build <name>` is resumable per shard. Trainings resume from `work/ckpt/<run>.resume.pt`.

## Log
- 2026-11-24 22:45 worktree fast-forwarded to 5a4b6da (bake-off import); water data junctions (raw, work, logs, proc/*)
  point at the main checkout's `shared/water/data` (git status clean).
- 23:20 `bank2.py selftest` OK (b2 code with every option off = p1 code). p1 rebuilt in 2.5 min (6 processes):
  31 of 36 arrays byte-identical to `harness/bank/MANIFEST.json`; q and states (both splits) and train qo differ
  (last-bit float output of WaterSim on Windows; qo rounded to 0.001 mm/day). See PREREG section 1.
- 22:52 (correction: the entry above says 23:20; the machine clock said about 22:52). Check (2) of PREREG section 1
  passed: the v1.0 `seq_lstm` arm scores sim-val 0.4255 and dev 0.4408 on the rebuilt p1 (its stored numbers).
- 22:57 b2a built (2.5 min, 5 processes): runs with reservoirs 16.6 % -> 19.9 % (p1 had 3.3 % of runs with a storage
  attribute and no reservoir; b2a has 0). Mean simulated flow 1.556 -> 1.562 mm/day.
- 22:58 b2c built (mosaic). calib_prior.py (dev only): p1's 4 coefficients reproduced (max |diff| 0.0005). LOO robust
  RMSE of log m*: M0 constant 0.153, M1 snow 0.177, M2 FlowCast 0.159 -> M0 chosen as pre-registered:
  log m = -0.026 + N(0, 0.153) (median m 0.97). Note: the plain LOO RMSE ranks them the other way (M2 0.206, M0
  0.231): the attributes help on the few outlying basins and not on the typical one. The pre-registered robust
  criterion stands. b2b building (1 process).
- 22:59 control queue started (5 processes): p1, b2a, b2c x seeds 0-2. About 0.24 s/step, ~24 min per run.
- 23:12 b2b built (1 process): median m 0.95 -> 0.97, 5-95 % 0.65-1.35 -> 0.76-1.25; mean simulated flow
  1.556 -> 1.411 mm/day (-9 %). b2b control seeds queued (separate 1-slot queues).
- 23:20 first control wave (sim val own bank / dev median NSE): p1 s0 0.428 / 0.444, s1 0.417 / 0.425,
  s2 0.434 / 0.417 (dev 3-seed mean 0.429; round 1's cloud seed 0 was 0.4255 / 0.4408, so the local p1 control
  reproduces round 1 within seed noise: PREREG section 1 check (3)). b2a s0 0.417 / 0.443 (p1 val 0.426);
  b2c s0 0.465 / 0.440 (p1 val 0.429). Training ~19 CPU-min per run.
- 23:42 bank rungs (control = LSTM r1 recipe, seeds 0/1/2; dev median NSE, 52 basins):
  - p1: dev 0.444 / 0.425 / 0.417, mean 0.4288; sim val 0.428 / 0.417 / 0.434.
  - b2a (reservoirs): dev 0.443 / 0.413 / 0.436, mean 0.4307, gain +0.002: REJECTED. Sim val own 0.415, on p1
    val 0.421 (p1 control 0.426). Paired bootstrap (seed-averaged per basin) -0.036..+0.013.
  - b2c (mosaic structures): dev 0.440 / 0.428 / 0.442, mean 0.4364, gain +0.008: REJECTED (under the 0.01
    margin). Sim val own 0.458, on p1 val 0.423. Bootstrap -0.004..+0.033.
- 23:43 v1.0 member arms (dev, --save-preds) and E1 (p1 sim val + dev) evaluating, 4 processes.
- 23:54 v1.0 members on dev reproduce round 1 exactly: flowcast_v1 0.4248, seq_lstm 0.4408, seq_tcn 0.4196,
  seq_ssm 0.4349, tab_gbm 0.4209. **Best dev member (E1's pre-declared exam comparator): seq_lstm 0.4408.**
- 23:55 E1 (ens_e1): sim val on p1 0.435 (members 0.398-0.426), dev 0.445 (KGE 0.435, FHV -32.5, cov50/90
  0.59 / 0.97, CRPS/Q 0.383). Gate 0.445 >= 0.95 x 0.4408: PASSES (registered as an arm). Ensembling helps on dev
  only if > best member + 0.01: +0.004, so NO such claim.
- 00:00 b2b (precipitation prior, constant m): dev 0.439 / 0.441 / 0.443, mean 0.4408, gain +0.012 over p1:
  **KEPT**. Sim val own 0.446, on p1 val 0.403 (p1 control 0.426: the b2b-trained net is worse on p1's frozen set,
  as expected when the multiplier prior changes). Seed-averaged paired bootstrap of dev medians -0.014..+0.076
  (covers 0: the gain is at the margin and not statistically clear). By the pre-registered rule: one change kept,
  so **b2 = b2b** (no combination bank). `work/banks/b2` is a junction to `work/banks/b2b`; runs are named b2b_*.
- 00:01 family retrains on b2 started: S4D (ssm_r2), TCN (tcn_r1), trees (gbm r2 recipe; lightgbm 4.7.0 locally,
  round 1 used 4.6.0). The LSTM's b2 rung is the control's seed-0 run on b2b (same recipe, same seed): not re-run.
  v1.0 arms being scored on b2's frozen val.
- Clock note: the two entries above marked 00:00 and 00:01 happened at about 23:54-23:57 (queue log times).
- 23:58 E1 registered (ens_e1, hash 42733b91...). Its conformal wrapper (round 1's Mondrian layer, fitted on E1's
  dev residuals) registered as ens_e1_conformal (d41da9d2...); dev mean coverage cov90 0.88 -> 0.88 (LOBO),
  cov50 0.52 -> 0.50, CRPS/Q (mean over basins) 0.520 -> 0.497.
- 23:59 disagreement on E1's members (descriptive extra; the pre-registered test is on E2's members): Spearman rho
  0.29, 95 % CI 0.02..0.50; with |volume bias| 0.51 (0.27..0.70).
- 00:00 LSTM family rung on b2 (= the control's seed-0 run on b2b): sim val on b2's frozen set 0.445 vs v1.0
  seq_lstm 0.394 (+0.051); dev 0.4386 vs 0.4408 (x 0.995) -> **KEPT**, registered as seq_lstm_b2 (20c85ab7...).
  By class it moves a lot: snow 0.535 -> 0.589, arid 0.182 -> -0.100, humid 0.573 -> 0.569, regulated 0.279 ->
  0.311, large 0.307 -> -0.100. The constant multiplier helps snowy basins and hurts dry ones; the overall median
  barely moves. TCN (0.71 s/step, ~71 min) and S4D (1.08 s/step, ~1.8 h) are training; trees training.
- 00:03 register2 now converts arm text files to LF before hashing (the repo's .gitattributes checks out eol=lf;
  Python on Windows wrote CRLF, so a fresh checkout would not have matched the hash). seq_lstm_b2, ens_e1 and
  ens_e1_conformal re-registered with --replace (logged in registry history; ens_e1's hash did not change).
- 00:06 trees: the first b2 training imported `arms_tab/src/feats.py`, which is round 1's rejected r3 module (54
  features); the released r2 arm uses `src/feats_v1.py` = `tab_gbm/feats.py` (47 features). It crashed at
  evaluation (feature index out of range), unscored, and was moved to work/gbm/b2b_failed_54feat. gbm2.py now loads
  the released feats.py; retrained.
- 00:10 exam2.py rehearsal tested on seq_lstm_b2 alone: full pipeline OK (v1.0 rows scored live on dev, dev refs,
  paired comparisons). Fixed: worker folder names cannot hold ':' on Windows (v1__<row>). Test output deleted.
- 00:24 trees family rung on b2: sim val on b2's frozen set 0.415 vs v1.0 tab_gbm 0.366 (+0.049); dev 0.430 vs
  0.421 (x 1.022) -> **KEPT**, registered as tab_gbm_b2 (0326ced5...). numpy trees == LightGBM 4.7.0 to 1.6e-14.
  v1.0 arms on b2's frozen val: seq_lstm 0.394, seq_tcn 0.386, seq_ssm 0.383, tab_gbm 0.366, flowcast_v1 0.370
  (they scored 0.398-0.426 on p1's val: b2's val is harder for nets trained on p1).
- 00:27 conformal wrappers registered: seq_lstm_b2_conformal (dev mean cov90 0.86 -> 0.88 LOBO),
  tab_gbm_b2_conformal (0.84 -> 0.88).
- 00:30 exam2.py gained `--jobs N` (arm workers in parallel processes; chooses nothing). Estimated exam time with
  one process is several hours (each ensemble re-runs its five members).
- 00:57 TCN family rung on b2: sim val on b2 0.436 vs v1.0 seq_tcn 0.386 (+0.050); dev 0.433 vs 0.420
  (x 1.032) -> **KEPT**, registered as seq_tcn_b2 (807ed6e0...); sim val on p1's frozen set 0.401. Conformal
  wrapper seq_tcn_b2_conformal registered (dev mean cov90 0.89 -> LOBO see its card).
- 01:17 S4D family rung on b2: sim val on b2 0.424 vs v1.0 seq_ssm 0.383 (+0.041); dev 0.4316 vs 0.4349
  (x 0.992) -> **KEPT**, registered as seq_ssm_b2 (c009c95f...), with seq_ssm_b2_conformal. All four families'
  b2 rungs are kept; every one passes on the sim criterion by +0.04-0.05 (b2's own val) while dev moves by
  -0.003..+0.013 only.
- 01:20 E2 = tab_gbm_b2, flowcast_v1, seq_lstm_b2, seq_tcn_b2, seq_ssm_b2 (round2/arms/ens_e2).
- 01:20 **Disagreement test (PREREG section 6, E2's members, dev only):** Spearman rho(relative spread, 1 - NSE of
  E2's mean) = 0.21, 95 % CI -0.10..0.48 (10,000 basin bootstraps): **does not pass** (CI covers 0). Extras: with
  |volume bias| rho 0.57 (0.33..0.74); per member 1 - NSE rho 0.24 (trees) and others in
  results/disagreement_e2_members.json. For E1's members the same test gave 0.29 (0.02..0.50): the signal is weak
  and its sign is stable, but it does not clear the pre-registered bar for the round-2 entries.
