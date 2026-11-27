# TripDetective v2.1 | Missing bus power shifts an entire mission and car reader lacks empty-input guards

**Cross-checked material public-inference defect under missing telemetry.** Reviewer A independently reran reviewer B's `759c345` probes against model base `61c70bf`, measured partial power-dropout propagation across the 14-leg ZTBus mission, verified that `training/busbank.py` applies the same zero-fill, and tested car-reader empty/all-NaN guardrails. [Original report and probe](reviews/reviewer-b/tripdetective-v2.1.md) · [Public bus reader](../../models/tripdetective/tripdetective.py#L376) · [Bus exam loader](../../models/tripdetective/training/busbank.py#L169)

**What failed and reproduced.**
- **Whole-mission propagation from missing bus power:** [`tripdetective.py:376`](../../models/tripdetective/tripdetective.py#L376) converts `NaN` power to `0.0` before checking sample validity, declares battery power present, and includes the leg in the window-weighted day mean. On the packaged 14-leg ZTBus mission, setting the longest leg's (index `8`, `8,695` samples) power to `NaN` lowered its estimate by **`−1,326.85 kg`** (`20,430.50 → 19,103.65 kg`) and raised every one of the 13 untouched legs by **`+238.83 kg`** (`+199.74 kg` vs. omitting leg `8`). Measured zero power gave bitwise-identical results, and explicitly marking every sample invalid (`valid = 0`) still kept the leg in the day mean (`18,868.59 kg`, shifting all 13 untouched legs by **`+281.14 kg`**).
- **Car grade & shared RoadSim isolation:** The offline car reader replaces missing grade with `0.0` without backward interpolation; all-NaN and flat grade matched identically on all 4 VED examples (`−25.47` to `+30.30 kg` vs. measured grade).
- **Weights, parity, and ratchet:** `0/1,343` near-zero rows in both weight archives; folded Torch↔NumPy max diffs **`<= 3.35e-7`** across all heads; `7/7`, `16/16`, `18/18` car decisions, `2/2` bus decisions, and `6/6` mission-calibration challengers replayed cleanly (with the v2.1 off-stop rule change explicitly visible in `ratchet.yaml`).

**Second-pass partial-dropout propagation, exam-loader check & car guardrails.**
- **Partial power dropout on ZTBus leg 8 (`10%–100%` `NaN`):** Even partial telemetry loss on a single leg monotonically corrupts all 13 untouched legs in the same mission:

| Fraction of leg 8 `power_W` set `NaN` | `Δ` leg 8 mass (`kg`) | `Δ` each of 13 untouched legs (`kg`) |
|---:|---:|---:|
| `10%` | `−88.16` | `+15.87` |
| `25%` | `−470.51` | `+84.69` |
| `50%` | `−789.56` | `+142.12` |
| `75%` | `−1,005.37` | `+180.97` |
| `100%` | `−1,326.85` | `+238.83` |

- **ZTBus exam loader uses the same zero-fill:** [`training/busbank.py:169`](../../models/tripdetective/training/busbank.py#L169) executes `P = np.nan_to_num(np.asarray(a['elec_power'], float))` before calling `bus_raw(...)`, matching the public reader's missing-to-zero behavior.
- **Car reader (`TripDetective.read_car`) empty/all-NaN guardrails:** On `VED_436` (base `1,621.73 kg`, `hev_parallel_p2`), setting speed `v` to all-`NaN` or all-`0` both silently return **`1,181.74 kg` (`conv`)**, and setting all float channels to `NaN` silently returns **`1,944.28 kg` (`hev_parallel_p2`)** without warning or error.

**What remains open.** Because the external ZTBus archive is not packaged in the repo, the prevalence of partial or complete `elec_power` dropouts across the 1,115 exam missions and its effect on published passenger-mass accuracy remain unmeasured.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/tripdetective_v2_1_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/wave5_export_probe.py tripdetective
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_759c345_probe.py tripdetective
```

Uses NumPy, SciPy, and CPU PyTorch. [Second-pass probe](reviews/reviewer-a/peer_759c345_probe.py)
