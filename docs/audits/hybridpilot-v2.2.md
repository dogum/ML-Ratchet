# HybridPilot v2.2 | Missing road grade changes earlier no-preview control

**Cross-checked shared-simulator look-ahead and optional preview boundary defect.** Reviewer A independently reran reviewer B's `db285c4` probe against model base `61c70bf`, mapped all RoadSim consumers across `models/`, and measured full-trip closed-loop fuel and state-of-charge divergence between RoadSim's bidirectional grade interpolation and causal forward-fill on the same 10-second gap with `+4%` post-gap grade step (`preview_s=0`, `nav=False`, `init_soc=0.55`). [Original report and probe](reviews/reviewer-b/hybridpilot-v2.2.md) · [RoadSim normalization](../../shared/vehicle/roadsim/roadsim/sim.py#L53) · [Policy features](../../models/hybridpilot/hybridpilot.py#L139)

**What failed and reproduced.**
- **RoadSim bidirectional grade interpolation (no-preview look-ahead):** [`roadsim/sim.py:53`](../../shared/vehicle/roadsim/roadsim/sim.py#L53) linearly interpolates missing road grade across surrounding finite samples over the entire route before stepping closed-loop simulation. With preview and navigation disabled, withholding 10 consecutive grade samples and raising grade by `+4%` starting strictly at the first sample after the gap changed earlier engine power by up to **`8.64 kW`** (and flipped an earlier engine on/off decision) on `VED_125_1275`, **`11.51 kW`** on `VED_428_465`, and battery current by **`35.98 A`** on `VED_492_2826`. Complete-grade controls produced **`0.0`** earlier difference on all three trips. Because RoadSim feeds the interpolated grade to both the drivetrain and controller, earlier power deltas couple physical demand and policy response.
- **Optional preview one-step endpoint overrun:** For a preview ending at second `k + L`, [`hybridpilot.py:85`](../../models/hybridpilot/hybridpilot.py#L85) reads speed and grade at `k + L + 1` to compute the final road-power interval. Bounding that endpoint left no-preview runs and two trips identical, while on `VED_428_465` with 30 s preview it shifted fuel by **`-0.273 g`** (out of `398.405 g`) and final SOC by **`-0.000735`** with zero engine on/off flips.
- **Weights, parity, seeding, and ratchet:** All 66 inputs have no EnergyNet work-fraction denominator (`49` varying features had normalized `std >= 0.01` on the 3 trips), all weights are finite with `0` dead columns, Torch↔NumPy raw-head parity is **`9.85e-6`** for shipped v2 (`1.19e-5` for historical v1), training seeds Torch and splits by case, and all **`23/23`** entries on the v2.2 release ladder replay cleanly.

**Second-pass closed-loop fuel/SOC comparison & shared-simulator scope.**
- **Shared RoadSim consumers:** Four models in `models/` import `roadsim` (`energynet`, `hybridpilot`, `lawfinder`, `tripdetective`; **49 Python files**).
- **Full-trip closed-loop divergence (`RoadSim interp` vs. `causal ffill` across the exact same 10 s missing-grade window + `+4%` post-gap step, `preview_s=0`, `nav=False`, `init_soc=0.55`):**

| VED trip | Gap window (s) | True-step fuel (g) | RoadSim interp fuel (g) | Causal `ffill` fuel (g) | `ΔFuel` (`interp − ffill`) | `ΔSOC` (end) | Max earlier `|ΔP_eng|` | Earlier engine flips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `VED_125_1275` | `41..50` | `352.477` | `359.159` | `350.155` | **`+9.004 g`** | `+0.00169` | `10.23 kW` | **`1`** |
| `VED_492_2826` | `33..42` | `46.577` | `46.577` | `46.577` | `0.000 g` | `-0.00200` | `0.00 kW` | `0` |
| `VED_428_465` | `28..37` | `806.077` | `809.456` | `804.378` | **`+5.078 g`** | `+0.00083` | **`15.35 kW`** | `0` |

**What remains open.** All three packaged VED examples have complete grade (`pipeline.py:49` zero-fills missing grade on examples, while `training/hp_env.py:244` forwards raw VED grade on exam routes). How often raw grade gaps occurred across the 432 exam cases—and whether causal grade gap handling alters cross-policy fuel rankings or affects `tripdetective` / `energynet`—remains unmeasured without the external VED archive.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/hybridpilot_v2_2_probe.py
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_db285c4_probe.py hybridpilot
```

Uses NumPy, SciPy, CPU PyTorch, and bundled RoadSim. [Second-pass probe](reviews/reviewer-a/peer_db285c4_probe.py)

## Resolution (27 Nov 2026)

| Finding | Real? | Action | Evidence |
|---|---|---|---|
| RoadSim interpolates a grade gap from both sides, so a no-preview controller "sees" grade from after the gap | Not changed | RoadSim's `normalize_trip` is the plant's estimate of the road that exists at each second; a real car's inclinometer reads that current grade. A causal forward-fill would make the simulated road less accurate, not the controller more honest. More to the point, HybridPilot's sets (`training/sets.py`) apply `np.nan_to_num` to grade before RoadSim sees it, so the interpolation path never runs on its dev or exam cases. | Measured on the local VED archive, same selection rule as `build_exam` (2 trips per exam hybrid): **0 of 119,352 grade samples missing on 216 exam trips**; 0 of 10,401 on 14 dev trips. The open question in the report ("how often raw grade gaps occurred across the exam cases") is answered: never. |
| Preview reads one second past its window (`hybridpilot.py:85`) | Yes | **Fixed**: road power at second t uses speed and grade at t + 1, so the power bins now sum t = k+a−1 … k+b−1 and read nothing past k + L. | New test changes speed and grade after k + L and checks the preview is identical (fails on the old code). The released R22 runs with `preview_s = 0`; preview appears only in rejected rungs (R4, R7, R14), whose margins (+0.007, score fell, −0.011) a −0.07 % fuel effect cannot flip. Not re-scored. |

The golden demo output is unchanged.
