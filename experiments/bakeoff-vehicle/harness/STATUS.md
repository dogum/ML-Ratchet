# Vehicle Bake-off harness: STATUS

**READY** (2026-11-23 14:05 EST). Phase 2 arm agents can start. Read `HARNESS.md` (section 0 first).

The exam has NOT been run. Do not run `exam.py --run` before every arm is registered (Phase 3). `real/exam/` does not
exist; exam.py builds it under the lock.

## Done
- Vendored the repo's shared/vehicle (roadsim, realdata, checks/hybrid_specs.py) and models/energynet (v2.1 code +
  weights): `vendor/VENDOR.json`, 140 files byte-identical to the repo. `vendor/vehicle/data/proc` -> the local
  /home/claude/road/data/proc (read-only); `vendor/vehicle/data/raw/ved` -> `data/raw/ved` (the two VED static xlsx
  tables, fetched from raw.githubusercontent.com/gsoh/VED, the URL in the repo's data/build.py; the local tree has no raw/).
- Data reconciliation (HARNESS.md section 2): every repo-tracked proc table equals the local one (sha256), incl.
  splits.json d917df78; the 13 Nov EPA-label and VED-mass fixes are in the local build; missing locally: proc/d3
  (not used by EnergyNet) and the full ZTBus traces (scoped out: the exam keeps the 78 v1 ZTBus legs / 8 missions).
- EnergyNet v2.1 reproduces EXACTLY on dev: 21 published dev terms, max |diff| 5.6e-16 (`results/check_enet_dev.json`).
  Bake-off dev score D16 (D2 minus the 2 full-ZTBus terms): 0.1133.
- Bank (EnergyNet v2.1's recipe, RoadSim physics v2 pinned): val 4,000 trips (= val2[h]: v2.1 MdAPE 0.0746, skill
  0.6797, all terms equal to 1e-9) + train 28,800 trips (= train2di, seed 101); 681 MB; built in ~45 min wall on
  2 workers (~1.3 CPU-h). `bank/*/MANIFEST.json`.
- API: bake.py (table + trace tracks, tiers, sim val, dev gate, guarded exam, block_means), score.py, refs.py,
  evaluate.py, armlib.py, register.py, exam.py, showcase.py (`showcase.json`: 8 seeded exam trips, frozen).
- Reference arms registered: `arms_ref/enet_v21` (table; sim val skill 0.680, D16 0.1133) and `arms_ref/climatology`
  (both tracks; skill -0.002, D16 0.309). Dev reference rows (E_phys, label, MOVES-like, class median, EPA regression,
  dev medians): `results/dev/`.
- Tests: `tests/test_guard.py` 30/30; `exam.py --rehearsal` (dev, both tracks) OK; `tests/test_exam_mock.sh` (exam-mode
  pipeline with dev standing in for the exam) OK; `exam.py --check` passes.

## If the container restarted
- Nothing is running. All outputs are on disk.

## Notes for Phase 2-3
- Before the exam: every arm registered; `python3 exam.py --check` must pass; then `python3 exam.py --run`
  (resumable with `--resume`). It builds the exam records first (~15 min), then runs each arm per track.
- If harness .py files change after an arm registered, `--check` only notes it; re-register to record the new hash.
- Disk: harness 705 MB now; the exam adds ~0.1 GB (records + predictions).
