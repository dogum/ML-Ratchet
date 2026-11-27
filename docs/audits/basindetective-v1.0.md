# BasinDetective v1.0 · Reversing the gauged period

**Cross-checked input guardrails and directional response.** All four packaged basins give a negative predicted change in mean flow when both forcing periods are identical. In one basin with observations for both periods, the model also predicts a decline in both directions when the periods are swapped. The trainer documents an asymmetric precipitation prior. Quantifying its contribution separately from the learned relationships requires a matched retrain. Model snapshot `61c70bf`, second review at `ea2bfde`. [First pass](reviews/reviewer-a/basindetective-v1.0.md) · [Model card](../../models/basindetective/MODEL_CARD.md#L73)

The packaged older decade (1989–99) contains forcing and six observed signature changes. The daily older flow needed to reconstruct the full calibration vector is absent. The recorded upstream sources were inaccessible from this workspace. Reviewer B therefore used two disjoint four-water-year windows with **measured daily flow on both sides**, October 1999–September 2003 and October 2004–September 2008. Each direction uses only its own calibration period’s discharge. The observed target uses the release’s log transform with a 0.01 mm/day offset. Inverting the periods negates that observed target exactly. [Input inventory](../../models/basindetective/examples/README.md) · [Label definition](../../models/basindetective/basindetective/features.py#L285)

| CAMELS gauge | Observed forward ΔlogQ | Predicted forward | Predicted reverse | Sum of predictions |
|---|---:|---:|---:|---:|
| 04185000 | +.2693 | +.0500 | −.2371 | −.1871 |
| 08165300 | −.5319 | −.3057 | +.3599 | +.0541 |
| 12388400 | +.1793 | −.0052 | −.1186 | −.1238 |
| 14020000 | +.0227 | +.0263 | −.0648 | −.0385 |

Changing the gauged period changes the hydrological evidence available to the network. The sum measures the disagreement between the two estimates. These shorter windows also differ from the training duration. The original 1989–99 → 1999–2008 replay is still open. The probe accepts `--old-flow-dir` containing one date-indexed `camels_<gauge>_flow.csv` with daily `Q_mm` per basin.

**Numerical correction.** Reviewer A reported float32 raw-output parity below `1e-5`. Its unmodified probe exits at that assertion here with Python 3.12.14, NumPy 2.5.5, SciPy 1.18.1 and PyTorch 2.8.0+cpu (one thread). Using the identical mixed batch, maximum error was `2.94e-6` on the four real rows and `1.53e-5` on the 16 synthetic rows. Float64 error was `7.11e-15`. All four shipped example predictions matched `expected.json` exactly. In a diagnostic continuation past that single assertion, the input failures (all-NaN and short flow raise `KeyError('bfi_lh')`, zero flow returns a finite estimate) and all seven historical rung decisions reproduced. [Original assertion](reviews/reviewer-a/basindetective_v1_0_probe.py#L107)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/peer_ea2bfde_probe.py basindetective
```
