# TreeTwin + ConformalRatchet: experiment scripts

Method experiment M2. Write-up: [`docs/experiments/treetwin.md`](../../docs/experiments/treetwin.md).
Pre-registration & freeze log: [`PREREG.md`](PREREG.md), [`prereg_log.json`](prereg_log.json).
Published results: [`results.json`](results.json), [`RESULTS.md`](RESULTS.md), and per-platform
exam records in `scores/<platform>.json`.

## What it tests

1. **Monotone physics-trained GBDTs (`mlratchet.gbm.TreeEnsemble`):** Pure-numpy gradient-boosted
   trees (exported from LightGBM, zero library dependency at inference, mirrored in
   `site/src/lib/nn/trees.ts`), trained strictly on physics simulation banks and compared against
   each platform's released neural incumbent on the exact same held-out real exam units.
2. **Exact TreeSHAP twin-test diagnostics (`mlratchet.twintest`):** Exact path-dependent TreeSHAP
   (Lundberg et al. 2020 Algorithm 2) turning the twin test into a ranked physical feature list
   of simulator-vs-reality gaps — replacing the placeholder chart on `site/src/pages/model-families.astro`
   with real measured VED-vs-RoadSim attributions.
3. **Mondrian conformal calibration (`mlratchet.calibrate.MondrianConformal`):** Per-stratum
   finite-sample split conformal calibration with empirical-Bayes shrinkage (`shrink_k=10`,
   `widen_only=True`) vs ML Ratchet's single pooled widening factor (`widen_factor`).
4. **Cross-family disagreement (`mlratchet.twintest.disagreement_detector`):** Testing whether
   `|GBDT-Mono - Incumbent MLP|` flags high-error real exam units without labels at inference.

## Rebuilding from scratch

All intermediate downloads, simulated banks, models and dev logs write to `WORK`
(`$ML_RATCHET_WORK/treetwin/`, else `experiments/treetwin/work/`, git-ignored).

```bash
export PYTHONPATH=experiments/treetwin
python3 experiments/treetwin/fetch_data.py              # UIUC LSAT, EIA RECS 2020 + NASA POWER, VED + 3DEP + ASOS
python3 experiments/treetwin/harness.py all train_tune  # physics banks + pure-numpy GBDT-Plain / GBDT-Mono
python3 experiments/treetwin/harness.py all dev_twin    # real dev TreeSHAP twin test + conformal fits
python3 experiments/treetwin/freeze.py                  # freeze PREREG.md SHA-256
python3 experiments/treetwin/harness.py all exam        # one-shot real exam across all 3 platforms
python3 experiments/treetwin/make_results.py            # write results.json and RESULTS.md
```
