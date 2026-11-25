# arms_trace (Phase 2, B: trace learners): STATUS

Agent B. Four trace-track arms, one shared input pipeline: `cnn` (1-D dilated CNN), `gru` (bidirectional GRU),
`transformer` (patch-token encoder), `s4d` (bidirectional diagonal state space). The exam is NOT ours to run.

## Layout
- `common/tracelib.py`: the shared pipeline (5-s block means -> 12 channels, 57 statics, stem MLP, mean pooling,
  chunking > 1,440 blocks, quantile head) + numpy inference of all four cores (hashed with every arm, extra_files).
- `train/nets.py` (torch nets + loss + export), `train/train.py` (resumable, one thread), `train/configs/<arm>_<rung>.json`,
  `train/run_rung.sh` (train -> candidate -> harness evaluate.py), `train/queue.sh` (sequential, skips evaluated rungs),
  `train/bench.py` (speed + numpy==torch check).
- `ckpt/` torch checkpoints (resume files), `logs/` per-rung logs, `<arm>/work/cand_<rung>/` candidate folders,
  `<arm>/work/eval/` harness evaluate outputs, `<arm>/work/train_<rung>.json` (CPU, export check).

## Plan (rungs; rule = HARNESS.md's: keep if sim val skill > incumbent + 0.01 AND dev D16 <= incumbent + 0.01)
- r0 (base): pure trace learner, anchor A0 = train-bank median MJ/km (arch x class) x distance; no E_phys. 12 epochs.
- r1: physics anchor: target log(y / E_phys) with E_phys = bake.e_phys_from_trace (declared), log(E_phys/A0) as an input.
- r2: arm-specific (transformer: more steps, per the River lesson; others decided from r0/r1).
- Shared recipe: 5-s blocks, tokens 8192 / <= 64 trips per batch, AdamW 2e-3 one-cycle cosine, pinball on 5 log-quantiles
  + 0.2 x component MSE, fresh tiers per epoch, last epoch used.

## If the container restarted
- Check `ps -eo pid,args | grep -E "train.py|queue.sh|evaluate.py"` (own PIDs only; logs/*.pid).
- Re-launch queues (resume from ckpt/*.resume.pt; evaluated rungs are skipped):
  `cd arms_trace/train; setsid nohup nice -n 5 ./queue.sh qA <configs...> > ../logs/qA.log 2>&1 < /dev/null &`

## Progress
- 14:00 read briefs; harness READY. Pipeline + nets written; numpy == torch (padded batch vs unpadded numpy) to 2e-7 in
  log space on all four. Params: cnn 86k, gru 53k, transformer 86k, s4d 49k (N2 16). Est. CPU per epoch:
  cnn ~160 s, gru ~150-200 s (padded GRU + per-trip reversal, 3.5x faster than packed), transformer ~55 s, s4d ~300 s.
- 14:05 smoke run (transformer, 1 epoch, 0.015 CPU-h, pilot): end to end OK; sim val skill 0.426, dev D16 0.147
  (pure trace, no physics anchor).
- 14:15 first launch killed by me after 2 min: RSS 1.5-1.6 GB (memory-mapped trace pages from the one-time block
  precompute). Fixed (madvise/fadvise DONTNEED every 2,000 trips): peak RSS 0.65 GB at precompute, ~0.9-1.05 GB training.
- 14:18 queues running: qA = s4d_r0 -> s4d_r1 -> cnn_r0; qB = gru_r0 -> transformer_r0 -> gru_r1 -> transformer_r1 -> cnn_r1.
  Machine shared (load ~5 on 2 cores): wall ~2.8x CPU. Est. CPU: s4d ~1.0 h/run, gru ~0.6, cnn ~0.5, transformer ~0.2.
- r0 results (full sim val 3,985 / dev D16): gru 0.605 / 0.119 (0.58 CPU-h); transformer 0.608 / 0.121 (0.19);
  s4d 0.616 / 0.115 (1.0). Pure trace learners, no E_phys. EnergyNet v2.1: 0.680 / 0.113.
- Plan for r2 (all arms): 2x training = warm restart from the incumbent for 12 more epochs (peak lr 1e-3, data seed 1).
- r1 (physics anchor) results: gru 0.639 / 0.1089 kept (+0.034, CI 0.019-0.046); transformer 0.630 / 0.1087 kept
  (+0.022); s4d 0.645 / 0.1200 kept (+0.028; D16 +0.005 within tolerance); cnn r0 0.617 / 0.1170, r1 0.633 / 0.1134 kept (+0.016).
- r2: gru (warm restart +12 epochs) 0.647 / 0.1102: +0.008 (CI -0.001..0.016) -> rejected; GRU released = r1.
  s4d r2 (+12 epochs) 0.655 / 0.1187: +0.0103 (CI 0.003..0.020), D16 better -> kept, released (decisions.json).
  transformer r2 (+24 epochs) 0.644 / 0.1135: +0.0145 (CI 0.004..0.025), D16 +0.0048 (within 0.01) -> kept.
  transformer r3 queued (qF): 10-s patches (patch 2), 36 epochs from scratch; vs incumbent r2.
- 19:44 CONTAINER RESTART (nothing running). cnn_r2 training had finished (0.509 CPU-h) but its harness eval was killed:
  re-run as eval only (train/eval_only.sh cnn r2). transformer_r3 resumed from ckpt (step ~7,800 / 33,444) via qF.
  Lost to the restart: ~0.01 CPU-h (r3 steps since last 200-step save + partial cnn eval).
- 19:51 cnn r2 (+12 epochs) 0.641 / 0.1085: +0.008 (CI -0.001..0.019) < 0.01 -> rejected; CNN released = r1 (decisions.json).
  Waiting on transformer_r3 (qF, ETA ~20:30 incl. eval); then release transformer, release.py cnn, freeze tracelib, register 4.
- Tools: train/ladder.py (ladder + paired sim-val bootstrap), train/npcheck.py (numpy vs torch, incl. chunked 37,919 s
  ZTBus leg: 9e-10 f64 / 1.7e-7 f32), train/release.py (arm folder + ratchet.json + card.md from train/decisions.json).
- 19:55 agent B resumed after an interruption (no second container restart): qF / transformer_r3 still running
  (PIDs 220/226/229; step 13,000 / 33,444 at 19:54, ~100 steps / 9 s wall). Not duplicated.
- 19:55 release.py re-run for gru (r1, 1.648 CPU-h) and s4d (r2, 2.75); cnn released 19:52 (r1, 1.572).
  common/tracelib.py FROZEN (chmod a-w; unchanged since 14:05; sha256 d64d133881a5285ea393ef568e9922c5acc39c4acf630862f24316ef783be25a).
  Registered (train/register_bg.sh): trace_cnn d430085a6092, trace_gru ef837cb12962, trace_s4d 280230f3ffa5.
- 19:57 arm C's conformal/wrap_pending.py --run (train/wrap_bg.sh; dev residuals only): tab_gp, trace_cnn, trace_gru, trace_s4d.
  Wrapped + registered (D16-mean cov50/cov90 raw -> cross-fit): tab_gp_conformal 0.61/0.93 -> 0.46/0.85 (97b339478708);
  trace_cnn_conformal 0.41/0.77 -> 0.49/0.85 (d16fa8e571f7); trace_gru_conformal 0.38/0.79 -> 0.48/0.86 (eda5b385436c);
  trace_s4d_conformal 0.36/0.72 -> 0.51/0.86 (151a00f02227). release.py: transformer text now follows the released patch size.
- 20:32 transformer_r3 done: 0.896 CPU-h, sim val 0.6404 / D16 0.1122; vs r2 d_skill -0.0037 (CI -0.017..+0.010) -> REJECTED.
  Transformer released = r2 (decisions.json), release.py transformer (1.635 CPU-h; npcheck r2 2e-9 f64).
- 20:38 registered trace_transformer d42add9c9877. 20:39 wrap_pending.py --run: trace_transformer_conformal 46ec6b2433f5
  (D16-mean cov50/cov90 raw 0.39/0.74 -> cross-fit 0.48/0.85). wrap_pending now lists nothing pending.
- 20:40 harness/exam.py --check: CHECK PASSED, 23 arms (all 4 trace arms + 4 trace wrappers + tab_gp_conformal);
  est. predict CPU 1,125 min (trace arms + wrappers ~175 min; tab_tabpfn + wrapper ~937 min).
## DONE (agent B). Released: cnn r1, gru r1, s4d r2, transformer r2. Group training compute 7.61 CPU-h.
   Exam NOT run (Phase 3). Do not touch arm folders or common/ (registered hashes).
