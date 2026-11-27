# BatteryPilot v1.3 | Pre-hour wear leakage, first-hour padding, and runtime parity

**Cross-checked critical pre-hour information-boundary defect.** Reviewer A independently reran reviewer B's `d92a17b` probe against model base `61c70bf`, verified weight health and Torch↔NumPy parity across all three shipped node-years, and inspected real-input standardization and training seeding. [Original report and probe](reviews/reviewer-b/batterypilot-v1.3.md) · [Released policy](../../models/batterypilot/batterypilot.py#L428) · [History padding](../../models/batterypilot/batterypilot.py#L271)

**What reproduced.** Both information-boundary findings reproduced identically:

- **Future temperature through planned wear:** On ERCOT West at 14:00 local (January 2, 2024), raising only the next 12 hours' temperatures by `+5 °C` moved the submitted offer curve by up to **$1.1877/MWh** (through the day-ahead wear planner), whereas raising the same future real-time prices by `+$100/MWh` moved the offer by `$0.0000/MWh` and raising next-day published day-ahead prices first moved the offer at 14:00 (`$18.0569/MWh`). Replacing realized temperatures in the wear planner with last-observation persistence preserved every strategy ranking across the three shipped node-years (`CAISO_TH_SP15_2025`, `ERCOT_HB_HOUSTON_2023`, `ERCOT_HB_WEST_2024`), shifting BatteryPilot annualized net revenue by at most `$0.11/kW-year`.
- **First-hour real-time price padding:** Increasing only hour 0's clearing real-time price by `+$100/MWh` moved the hour-0 offer curve by **$3.0164/MWh** because missing history indices clip to row 0.
- **Rung arithmetic:** All 9 candidate decisions matched the logged scores under the strict `0.01` margin and development gate (`R8` final incumbent).

**Independent implementation checks.** Shipped weights (`batterypilot_weights.npz`) have zero NaN/Inf, zero dead rows, and bounded feature scales (`x_sd ∈ [0.0500, 0.9514]`). Exact PyTorch ([`training/train.py:46`](../../models/batterypilot/training/train.py#L46) `Net`) ↔ NumPy (`PolicyNet.z_knots`) forward parity held within `3.05e-7` on random batches and within `2.92e-7`, `1.17e-6`, and `6.42e-7` across the full 8,760-hour feature matrices of `CAISO_TH_SP15_2025`, `ERCOT_HB_HOUSTON_2023`, and `ERCOT_HB_WEST_2024`. [`training/train.py:192`](../../models/batterypilot/training/train.py#L192) seeds PyTorch (`torch.manual_seed`). During scarcity spikes, far-horizon day-ahead feature `col 35` (`sd = 0.050` floor) reaches `max |z| = 23.53–34.06` unclipped before layer 0, while output knots are clamped to `[−12, 12]` in [`PolicyNet.z_knots`](../../models/batterypilot/batterypilot.py#L417).

**What remains open.** The revenue effect of causal wear forecasts and non-leaking first-hour history padding across the other 45 exam node-years, alternative battery durations, and retraining/rung selection remains unmeasured.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/batterypilot_v1_3_probe.py
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_dc30c02_probe.py batterypilot
```

The second-pass probe uses NumPy, SciPy, pandas, and CPU PyTorch. [Second-pass probe](reviews/reviewer-a/peer_dc30c02_probe.py)

## Resolution (27 Nov 2026)

| Finding | Real? | Action | Evidence |
|---|---|---|---|
| Day-ahead wear planner reads realized future temperatures | Yes (a perfect temperature forecast) | **Opt-in fix, re-score queued.** `causal_wear(bat, T)` gives the planner temperatures from 24 h earlier; `run_policy(..., c_lin_plan=...)` uses it. Default stays as released, so the card's exam numbers remain what v1.3 does. | Three packaged node-years: −$0.062, +$0.011, +$0.012/kW-year (the audit measured ≤ $0.11). Test checks a temperature change after hour h does not reach the plan before h + 24. |
| Hour 0 reads its own real-time price as history | Yes | **Fixed** in `build()` and `training/features.py`: at t = 0 the day-ahead price stands in; from t = 1 nothing changes. | +$100 on hour 0's RT no longer changes hour 0's features (test). Packaged node-years move by at most +$0.0016/kW-year. |
| Day-ahead column 35 reaches |z| 24–34 in scarcity | Observation | Not changed: output knots are clamped and the exam ranks held; noted for the next retrain. | — |
