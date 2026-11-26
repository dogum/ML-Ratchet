# FaultFinder v1.1 | Date-selection defect, shared weather-cache collision, and API guardrails

**Cross-checked date-selection defect, shared BuildingSim cache collision, and missing public API guardrails.** Reviewer A independently reran reviewer B's `db285c4` probe against model base `61c70bf` and extended the public API audit to self-reference leakage (`ref_days=[day]`) and truncated partial-day prefixes (`6 h` and `12 h`). [Original report and probe](reviews/reviewer-b/faultfinder-v1.1.md) · [Flatline check and date selector](../../models/faultfinder/faultfinder.py#L83) · [Weather cache key](../../shared/building/buildingsim/buildingsim/weather.py#L219)

**What failed and reproduced.**
- **Date-selection flatline look-ahead:** In [`faultfinder.py:83`](../../models/faultfinder/faultfinder.py#L83), `raw_day` computes temperature-channel flatline checks over the entire supplied DataFrame *before* filtering to the requested date. On `flexlab` with target-day supply temperature fixed flat at `18 °C`, scoring the day alone or with an appended `18 °C` day gave **`P(fault) = 0.51336`**; changing only the appended future day's supply temperature to `19 °C` moved the target day to **`0.49950`**, crossing the `0.50` fault alarm threshold. All four packaged target days exhibited this sensitivity under the flatline stress, whereas unmodified single-day controls and four genuine multi-day same-unit pairs were unchanged. The Nehasil exam runs daily quality control ([`exam2_common.py:75`](../../models/faultfinder/training/exam2_common.py#L75)).
- **Shared BuildingSim wind-cache collision:** [`buildingsim/weather.py:219`](../../shared/building/buildingsim/buildingsim/weather.py#L219) omits `wind_speed` from its cache key. Reusing a warm `2 m/s` cache during an `8 m/s` synthetic winter VAV office run hid a **`5.19 °C`** supply-temperature change (`P(fault) = 0.46271 -> 0.46677` once cleared). No mislabeled bank or ThermoPilot building-card leakage was found.
- **Weights, parity, seeding, and ratchet:** All folded weight arrays are finite with **`0`** dead output filters/rows; all three heads match PyTorch within **`7.16e-7`** on the four examples; `train8.py` seeds Torch; all **`6/6`** v4 decisions and the scored v5 reference candidates replay cleanly.

**Second-pass public API guardrail probes (`self-reference` and `partial-day prefix`).**
- **Reference chronology & self-reference leakage:** Reviewer B showed re-dating references two weeks later had zero effect; reviewer A verified that passing the target day itself as its own reference (`read_day(df_day, ref_days=[df_day])`) is silently accepted and shifts `P(fault)` across all four shipped examples—most sharply on `ornl_rtu` (**`-0.08888`**, `0.63244 -> 0.54357`).
- **Truncated partial-day prefixes & empty inputs:** Reviewer B found an all-empty day silently returns `P(fault) = 0.45255`. Reviewer A passed true partial-day prefixes (`6 h = 24 rows` and `12 h = 48 rows`) to `read_day`: all three examples that triggered a full-day alarm fell below 0.50 on both prefixes. Two have documented faults. Iowa ERS is labelled healthy, so its full-day alarm was a false positive. [Example labels](../../models/faultfinder/examples/labels.json)

| Packaged example | Full-day `P(fault)` (with shipped refs) | Self-reference `ref=[day]` (`Δ`) | `6 h` prefix (`24` rows) | `12 h` prefix (`48` rows) | Alarm flipped at `6 h`? |
|---|---:|---:|---:|---:|---|
| `flexlab` | `0.62082` | `0.62557` (`+0.00475`) | `0.47895` | `0.48324` | **Yes (`>= 0.50 -> < 0.50`)** |
| `iowa_ers` | `0.58881` | `0.59954` (`+0.01073`) | `0.46462` | `0.46630` | **Yes (`>= 0.50 -> < 0.50`)** |
| `nesbitt` | `0.45723` | `0.45855` (`+0.00132`) | `0.43518` | `0.44089` | No (already `< 0.50`) |
| `ornl_rtu` | `0.63244` | `0.54357` (**`-0.08888`**) | `0.49225` | `0.49288` | **Yes (`>= 0.50 -> < 0.50`)** |

**What remains open.** How often the `raw_day` multi-day flatline bug changes alarms on real multi-day BMS export files—and whether any historical simulation bank run experienced a warm wind-cache collision—remains unmeasured.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/faultfinder_v1_1_probe.py
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_db285c4_probe.py faultfinder
```

Uses NumPy, pandas, CPU PyTorch, and BuildingSim dependencies. [Second-pass probe](reviews/reviewer-a/peer_db285c4_probe.py)
