# ThermoPilot v1.1 | Evaluation-weather card look-ahead and BuildingSim weather-cache collision

**Cross-checked evaluation look-ahead and shared-simulator cache defect.** Reviewer A independently reran reviewer B's `d903cb6` probe against model base `61c70bf`, audited the optional v1.0 MLP weights and shared BuildingSim consumers, and extended the wind-card experiment to four 30-day closed-loop BuildingSim episodes comparing evaluation-month cards against causal prior-month cards. [Original report and probe](reviews/reviewer-b/thermopilot-v1.1.md) · [Card fit](../../models/thermopilot/training/tp/features.py#L50) · [Weather cache](../../shared/building/buildingsim/buildingsim/weather.py#L219)

**What failed and reproduced.**
- **Evaluation-weather card look-ahead:** In the exam setup, `full_card` / `card_from_model` fits the 12-parameter house map and summary fields (`UA`, `P_fan`, `Qheat`) on the nominal twin's hourly coefficients over the evaluation episode, which include realized wind via infiltration. Changing only wind from hour 48 onward (`2 -> 8 m/s`, beyond the 24-hour forecast horizon) moved `At` from **`0.89788` to `0.87349`** and changed issued heating commands in **19 of 72** matched fresh starts (up to `2 °C`, `22 -> 20 °C` at hour 0).
- **Shared BuildingSim weather-cache collision:** [`buildingsim/weather.py:219`](../../shared/building/buildingsim/buildingsim/weather.py#L219) hashes outdoor temperature, timestamps, and preparation settings but omits `wind_speed`, so warm-cache runs silently reused the `2 m/s` wind term (`0.000924`) instead of `8 m/s` (`0.014784`, `16×`). Five models in `models/` import `buildingsim` (`buildingdetective`, `faultfinder`, `lawfinder`, `loadcast`, `thermopilot`; 34 files).
- **Fixed-card & tariff controls:** With a card fixed, future tariffs/base loads beyond 24 h and cold-start observer history had `0.0` effect on all 36 first-action costs. Repricing the three packaged trajectories with Houston 2023 RT vs DA future prices changed commands in **12, 32, and 45 of 744 hours**.

**Second-pass closed-loop replay & weight checks.**
- **30-day closed-loop BuildingSim episodes (full-episode card vs. causal prior-month card, fresh cache, TOU `$0.48` weekday 15–20 vs. `$0.10`):** Across all four wind-mismatch scenarios (three winter heating, one summer cooling), swapping the evaluation-month card for a prior-month card (`2 m/s` vs `8 m/s`) shifted `v1.1`'s closed-loop monthly bill/objective `J` by **`-$1.02` to `+$0.23` (`<= 0.21%`)**, kept comfort violations at **`0.00 K·h`**, and preserved every strategy ranking:

| 30-day BuildingSim scenario | `At` full → prior | `UA` full → prior (W/K) | `v1.1` diff hrs | `fixed` / `setback` `J` | `v1.1` full → prior `J` (`ΔJ`) | `v1.0` full → prior `J` |
|---|---:|---:|---:|---:|---:|---:|
| `winter_8vs2` | 0.86881 → 0.89822 | 321.12 → 267.56 | 148 / 720 | \$500.75 / \$484.84 | \$491.78 → \$490.76 (−\$1.02) | \$480.18 → \$481.25 |
| `winter_2vs8` | 0.89822 → 0.86881 | 267.56 → 321.12 | 141 / 720 | \$428.03 / \$415.07 | \$417.34 → \$416.71 (−\$0.63) | \$409.50 → \$408.36 |
| `winter_step2to8` | 0.87074 → 0.89822 | 317.69 → 267.56 | 141 / 720 | \$496.35 / \$480.71 | \$486.84 → \$486.03 (−\$0.80) | \$475.83 → \$476.91 |
| `summer_8vs2` | 0.88600 → 0.91993 | 307.70 → 261.32 | 21 / 720 | \$428.86 / \$423.01 | \$383.04 → \$383.27 (+\$0.23) | \$405.18 → \$403.15 |

- **v1.0 non-map card path & weights:** `card_from_model` also carries wind into `UA`, `P_fan`, and `Qheat`, causing `v1.0` to differ in `46–202 / 720` hours when the full prior card is substituted (`|ΔJ| <= $2.03`). All arrays in `thermopilot_v1.npz` are finite with **0 dead rows or columns** across `W0/W1/W2`; Torch↔NumPy logit parity reproduced within `1.28e-5`–`3.25e-5` with **0/2,232** argmax disagreements; both 7-row ratchet ladders reproduced.

**What remains open.** Re-running the full archived multi-building RECS/CBECS exam (external survey & weather archives) with strictly pre-episode cards—and auditing historical runs across the other four BuildingSim models for warm-cache collisions—remains unmeasured.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/thermopilot_v1_1_probe.py
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_d903cb6_probe.py thermopilot
```

Uses NumPy, pandas, SciPy, pvlib, Numba, and CPU PyTorch. [Second-pass probe](reviews/reviewer-a/peer_d903cb6_probe.py)
