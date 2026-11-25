# arms_seq (Phase 2, A: sequence learners): STATUS

Agent A. Four ungauged arms, one shared input pipeline: `lstm` (FlowCast's net retrained, the in-bake-off control),
`transformer`, `ssm` (S4D), `tcn`. The exam is NOT ours to run.

**REGISTERED (2026-11-23 07:50 EST): all four arms, frozen. Do not touch the arm folders or common/ (except work/).**

| Arm id | Released | Sim val NSE | Dev NSE | Train CPU-h | Registry hash | Exam est. |
|---|---|---|---|---|---|---|
| seq_lstm | r1 (6000 steps) | 0.4255 | 0.4408 | 0.867 (incl. 2 seed checks) | 93421517ac8df0b76f96842d6540f381afaf1014b9fe0ea01bfa6015ff6e8501 | 1.3 min |
| seq_tcn | r1 (6000 steps) | 0.4164 | 0.4196 | 1.251 | d85e8142dafdc01cef6d90aea75706547b7985720ae9104ea847cad43e0a37f8 | 6.5 min |
| seq_ssm | r2 (pre-MLP + S4D) | 0.4220 | 0.4349 | 3.693 | d93736f3c9c12f38397986d92ec29e62edc769481165dfd765d0a963723305bd | 7.3 min |
| seq_transformer | r3 (12000 steps, 2 heads) | 0.4205 | 0.4109 | 3.553 | 58490e2c52a25844432906dafe78e4b192bb7204b71ca528a73f43ceb95b2cbe | 16.5 min |

`common/seqlib.py` frozen (read-only) sha256 d53f7d73f085fd0546b4fb55d8d4dc513e4f53b3460d3617f70b39dcf32f3dda
(`common/SEQLIB_FROZEN.sha256`). Arm files chmod a-w. `register.py --verify` OK for all four.
Conformal (done 07:55 by agent C's watcher, Mondrian, dev residuals; all four wrappers registered, wrap_pending.py shows nothing pending):
seq_lstm_conformal 847a3c71..., seq_tcn_conformal 0ada27ab..., seq_ssm_conformal a2fe7765..., seq_transformer_conformal 61c41d94...
Dev cov90 raw -> LOBO: lstm 0.962 -> 0.948, tcn 0.972 -> 0.952, ssm 0.972 -> 0.947, transformer 0.970 -> 0.939.
Agent A's work is complete; nothing of ours is running.

## Layout
- `common/seqlib.py`: the shared pipeline + numpy inference of all four cores (hashed with every arm, extra_files).
- `train/nets.py` (torch nets + FlowCast loss + export), `train/train.py` (resumable, one thread),
  `train/configs/<arm>_<rung>.json`, `train/run_rung.sh` (train -> candidate -> harness evaluate.py),
  `train/queue.sh` (sequential rungs, skips evaluated ones), `train/ladder.py` (ladder + paired bootstrap).
- `ckpt/` torch checkpoints (resume files), `logs/` per-rung logs, `<arm>/work/cand_<rung>/` candidate arm folders
  (evaluation only), `<arm>/work/eval/` harness evaluate outputs, `<arm>/work/train_<rung>.json` (CPU, export check).

## If the container restarted
- Re-launch the queues (they resume from ckpt/*.resume.pt and skip evaluated rungs):
  `cd arms_seq; setsid nohup nice -n 10 train/queue.sh qA <configs...> > logs/qA.log 2>&1 < /dev/null &`
- Check `ps -eo pid,args | grep train.py` first; kill only own PIDs.

## Progress
- 23:35 shared pipeline + 4 nets written; numpy == torch checked on smoke runs (rel. diff <= 5e-6);
  params lstm 24,710 / transformer 26,854 / ssm 25,734 / tcn 26,266.
- 23:37 r0 of all four launched (queues qA: lstm_r0, tcn_r0; qB: ssm_r0, transformer_r0), nice 10.
- 00:20-00:55 r0 results (harness evaluate, all 800 sim val runs + 52 dev basins):
  lstm r0 sim 0.410 / dev 0.391; ssm r0 0.398 / 0.410; tcn r0 0.401 / 0.412. Seed check lstm seed1 0.410 / 0.431
  (dev seed noise ~0.04!). transformer r0 training.
- run_rung.sh had a /usr/bin/time call that does not exist: lstm/tcn/ssm r0 evals were re-run by train/evalq.sh (fixed).
- 00:56 SSM FFT length 2T -> next power of 2 (2048; exact linear convolution either way), numpy and torch.
- Queued: qA lstm_seed2 -> lstm_r1 -> tcn_r1 (6000 steps); qB transformer_r0 -> ssm_r1.
- 01:30 lstm seed2: sim 0.409 / dev 0.404 -> LSTM sim val seed noise ~0.001, dev seed noise ~0.02 sd.
- 02:00 lstm r1 (6000 steps): sim 0.426 / dev 0.441 -> kept (+0.016 sim).
- 02:35 transformer r0: sim 0.364 / dev 0.362 (learns slowly: quick val 0.30 / 0.34 / 0.39 at 1k/2k/3k steps).
- 02:40 running: tcn_r1, ssm_r1, transformer_r1 (all 6000 steps); queued: lstm_r2 (12000 steps) after tcn_r1.
  Prepared: tcn_r2 (gated WaveNet activations, c=20, 26,866 params), ssm_r2 (pointwise MLP encoder before the SSM,
  30,438 params). Both 6000 steps if r1 is kept.
- 02:45 tcn r1 (6000 steps): sim 0.416 / dev 0.420 -> kept. tcn_r2 (gated) queued after lstm_r2.
- 03:05 transformer r2 written: causal PATCHED tokens (7-day patches, 105 tokens, 2 blocks; each day reads the
  summary of all earlier complete weeks + a depthwise causal conv over its last 7 days); 25,798 params; 10x faster.
  Causality checked (outputs before day 400 unchanged when inputs after it change). Launched (6000 steps).
- 04:05 lstm r2 (12000 steps): sim 0.428 / dev 0.440 -> rejected (+0.003). LSTM RELEASED = r1 (sim 0.4255, dev 0.441):
  lstm/{predict.py, weights.npz, ratchet.json, card.md} written (train/release.py + train/finalize.py from
  train/decisions.json). numpy == torch float64 to 3e-15. Registration dry run (register.check_arm) clean.
  NOT yet registered: register all four at the end (seqlib.py is shared; any edit would change every hash).
- 03:40 transformer r2 (patched, 6000 steps): sim 0.341 / dev 0.353 (worse than daily r0 0.364).
- 05:00 ssm r1 (6000 steps): sim 0.399 / dev 0.428 -> rejected (+0.001: S4D does not gain from more training,
  unlike LSTM/TCN). ssm_r2 (pre-MLP, 6000 steps) running; its incumbent is r0 (r1 shows steps alone add nothing).
- 05:10 ssm r3 written: Mamba-lite (selective SSM, Mamba-2 scalar decay per head, chunked SSD in torch, exact
  day-by-day recurrence in numpy; torch f64 == numpy to 1.6e-15; causal). 28,926 params. Queued after tcn_r2 (6000 steps).
- 05:30 tcn r2 (gated, 6000 steps): sim 0.405 / dev 0.425 -> rejected. TCN RELEASED = r1 (sim 0.416, dev 0.420);
  tcn/ folder finalized (not registered yet).
- 06:15 transformer r1 (6000 steps): sim 0.401 / dev 0.389 -> kept (+0.037). r2 (patched) therefore judged against
  r1: rejected. transformer_r3 launched: 12000 steps, 2 heads x 16 (same params; halves attention cost).
- Running now: ssm_r2 (near end), ssm_r3 (Mamba-lite), transformer_r3.
- 07:20 ssm r2 (pre-MLP + S4D): sim 0.422 / dev 0.435 -> kept; ssm r3 (Mamba-lite): sim 0.415 / dev 0.429 -> rejected.
  SSM RELEASED = r2; ssm/ folder finalized. Waiting for transformer_r3 (12000 steps, 2 heads).
- 07:39 transformer r3 (12000 steps, 2 heads x 16): sim 0.4205 / dev 0.411 vs incumbent r1 0.401 / 0.389 -> kept
  (sim +0.020, bootstrap +0.002..+0.032). TRANSFORMER RELEASED = r3. Compute 3.55 CPU-h (r0 0.59, r1 1.27, r2 0.14, r3 1.55).
- 07:50 seqlib.py frozen; all four arms registered (table above). Training queues finished; no own jobs running.
