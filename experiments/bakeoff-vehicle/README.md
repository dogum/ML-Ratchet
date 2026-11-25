# Vehicle Bake-off (experiment, 23-24 Nov 2026)

One physics problem, every learner family, equal terms, one real exam. The problem is EnergyNet's trip energy, in two
input forms: EnergyNet's 151 physics features (table track) and the 1 Hz trip trace (trace track). Write-up:
[`docs/experiments/bakeoff-vehicle.md`](../../docs/experiments/bakeoff-vehicle.md).

```
experiments/bakeoff-vehicle/
  harness/          the shared harness: bank recipe (build_bank.py, run_bank.sh) + bank/{train,val}/MANIFEST.json
                    (sha256 of every array), data API (bake.py), scorer (score.py), references (refs.py),
                    registration (register.py), the once-only exam (exam.py), showcase draw (showcase.json),
                    HARNESS.md, registry/ (21 examined arm hashes, 2 excluded TabPFN records + excluded.json),
                    arms_ref/ (EnergyNet v2.1 as released, climatology), vendor/VENDOR.json (vendored-code hashes)
  arms_table/       MLP (control), gradient-boosted trees, sparse GP, TabPFN (excluded; weights not included)
  arms_trace/       1-D CNN, bidirectional GRU (control), S4D, transformer: shared pipeline common/tracelib.py, train/
  arms_phys/        trip-energy law (formula), physics anchor, conformal/ (Mondrian wrappers for every arm),
                    lawfinder_road/ (LawFinder's fifth law)
  results/          exam_20261123T204251.json (boards, paired bootstrap, conditions, EnergyNet reproduction, compute,
                    hashes), exam_20261123T204251/{showcase.json, showcase.npz} (the 194 MB per-unit
                    scores.parquet is in the release zip's harness/results, not here),
                    analysis.json + leaderboards.md (phase3/analyze.py), extra_comparisons.json (phase3/extra.py),
                    excluded.json, the Phase 1 checks
  phase3/           analyze.py, extra.py, the page builder + template + text + checker, STATUS.md (exam log),
                    register_exclusion_fix.diff
  page/             vehicle-bakeoff.html (the self-contained page)
```

**Not included:** the 681 MB simulated bank (rebuild with `harness/run_bank.sh`, ~1.3 CPU-h; check against the
MANIFESTs), the vendored `shared/vehicle` and `models/energynet` code (byte copies; hashes in `vendor/VENDOR.json`;
copy them back to run), the rebuilt real records (`harness/build_real.py` from RoadData), training checkpoints and
work/ folders (in the release zip), and TabPFN's third-party weights.

**Rules kept.** Zero real labels in training; dev used for rung gates and conformal calibration only; each arm at most
about 4 single-thread CPU-hours and 3 rungs; one exam (`exam.py --run`, run id 20261123T204251) for all 21 registered
arms at once, which re-hashed every arm before and after scoring it. The exam chose nothing. TabPFN v2 and its wrapper
were excluded before the exam for compute (reason in `results/excluded.json`).

**Reproduce the analysis** from the exam outputs alone (copy `scores.parquet` from the release zip into
`results/exam_20261123T204251/` first): `python3 phase3/analyze.py` and `python3 phase3/extra.py` (they read `harness/results/`). `exam.py --check` refuses to run the exam again (`results/EXAM_DONE.json`).

**In this repository (imported 24 Nov 2026).** Every file is as shipped (SHA256SUMS.json of the release zip checked on
import), except the paths: `harness/config.py` and `phase3/{analyze,extra,build_page,check_page}.py` now resolve the
repo root, RoadData's processed data (`$ML_RATCHET_VEHICLE/data`, default `shared/vehicle/data`) and the code
(`shared/vehicle` and `models/energynet` themselves: all 140 vendored files equal the committed blobs on import)
instead of `/home/claude/...`, and `analyze.py` maps the registry's cloud arm folders onto this folder. Results stay in
`results/`; re-runs write to `work/` (git-ignored). `scores.parquet` is git-ignored: copy it from the release zip
(`vehicle-bakeoff-v1.0-exam-raw.zip`) into `results/exam_20261123T204251/`. The arms' own training scripts still name
their cloud paths: they are the record of what ran. LawFinder's packaging of the formula arm's law is in
`arms_phys/lawfinder_road/` (with its exam conversion in `exam_results/`); it was not merged into LawFinder, which
already released its own road law (models/lawfinder/NOTEBOOK.md, 24 Nov 2026).
