# Vehicle Bake-off round 2: STATUS

Pre-registration: `PREREG.md` (committed d1a98a4 before any training). Exam-2 is NOT run (the owner runs it once).

## Layout
- `paths.py` the path layer (import first): round-2 banks, work/, registry/, results/; points the harness's outputs
  (dev records, dev refs) at `work/` so v1.0's `results/` and `harness/registry` are never written.
- `bank/` bank builder (harness/build_bank.py driven per named bank; `run_banks.sh` = 8 workers + finalize).
- `trace/` GPU training of trace ensembles (`train_ens.py`, v1.0 nets and recipe imported unchanged; `dyncache.py`).
- `table/` MLP control on several banks (`train_mlp3.py`, v1.0's train_mlp.py imported).
- `arms/` round-2 arm folders (`common/enslib.py`: numpy ensemble inference).
- `eval.py` harness evaluate.py on round-2 banks (`--val val_v4` for the P0 physics's own val).
- `work/` (git-ignored): banks, dyn caches, checkpoints, eval outputs, dev records, logs.

## If the session restarts
- Bank build: `bash bank/run_banks.sh <banks...>` is resumable (finished chunks are skipped).
- Training: `train_ens.py <config> <member>` resumes from work/trace/<name>/m<k>/phase<j>.resume.pt.

## Log
- 24 Nov 22:40 EST. Worktree fast-forwarded to 5a4b6da (bake-off import). `shared/vehicle/data` is a junction to the
  main checkout's data (the tracked files there equal this commit's except CRLF line endings).
- 22:44 bank build started: val, train101, train102, train103 (8 workers, ~3 min per 600-trip chunk).
- 23:00 PREREG.md written and committed (d1a98a4) before any training.
- EnergyNet v2.1's own dev terms (training/ratchet2.json, r17), read for the P0 evidence rule: VED Leaf bias +4.6 %
  (1 dev car; exam -20 %: not visible on dev), VED plug-ins -7.3 %, Mersin CNG -17.0 %, Mersin diesel -3.1 %
  (dev is February only, no A/C trips), ICCT +19.0 % (3 cycles of one truck).
