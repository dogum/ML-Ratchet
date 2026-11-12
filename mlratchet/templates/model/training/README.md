# Training: __NAME__

Run every script from this folder. Outputs go to `../work/` (git-ignored) or `$ML_RATCHET_WORK/__ID__/`.

| Step | Script | Writes |
|---|---|---|
| 1. Physics | `simulator.py` — the forward model | — |
| 2. Check it | `physics_check.py` — against an independent reference | `physics_check.json` (commit it) |
| 3. Real noise | `noise.py` — fetch and screen real background recordings | `work/noise/` |
| 4. Split | `split.py` — dev / exam units by a fixed seed | `split.json` (commit it) |
| 5. Banks | `simbank.py train1 50000 1` and `simbank.py val1 5000 99` (frozen) | `work/bank/` |
| 6. Train | `train.py --name r1 --bank train1` (resumable checkpoints) | `work/ckpt/` |
| 7. Ratchet | `ratchet.py r1 "what changed"` — frozen-set gain AND dev gate | `../ratchet.yaml` (commit it) |
| 8. Exam | `exam.py r1` — once, at the end, against baselines | `exam.json` (commit it) |
| 9. Export | `export.py r1` — fold BatchNorm, numpy parity, `.npz` | `../__ID___weights.npz` |

Keep the network small (≤ ~500k parameters is usually plenty), banks in float16, and write a resumable
checkpoint every few hundred steps.
