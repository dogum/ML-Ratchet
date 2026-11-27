# SunDoctor v2.1 | CSV row-order collapse, incomplete-day totals, and phantom fault attribution

**Cross-checked material public-API reporting and fault-attribution defects.** Reviewer A independently reran reviewer B's `759c345` probes against model base `61c70bf` and extended the tests to partial out-of-order blocks (one adjacent-day swap and one late-arriving day at EOF), fault-attribution shifts under 5-hour midday gaps in both irradiance and power-only modes, and all-NaN power guardrails. [Original report and probe](reviews/reviewer-b/sundoctor-v2.1.md) · [CSV reader](../../models/sundoctor/sundoctor.py#L250) · [Energy calculation](../../models/sundoctor/sundoctor.py#L191)

**What failed and reproduced.**
- **Descending CSV collapse:** Reversing row order (`newest-first`) on the first 30 days of both packaged PVDAQ sites (`CA_2107` fixed inverter, `CO_9068` tracker) caused `diagnose_csv` to return **`0` valid days and `0.00 kWh`** (vs. `30` days / `7,248.35 kWh` and `54,311.03 kWh`). Re-sorting ascending restored exact totals.
- **Midday gap under-reporting:** Masking `10:00–15:00` local time left all `30` days counted while cutting `kwh_made` by **`60.8%`** (`7,248.35 → 2,840.16 kWh`) on `CA_2107` and **`67.1%`** (`54,311.03 → 17,855.38 kWh`) on `CO_9068`, because `sundoctor.py` counts any day with `> 50%` coverage and substitutes `0` for missing intervals. Equal-duration overnight gaps (`00:00–05:00`) left totals unchanged.
- **Model controls:** `0/2,047` near-zero rows; folded PyTorch↔NumPy max diffs `8.11e-6 / 9.54e-7` (real fortnight) and `1.14e-5 / 4.77e-7` (random; slightly above `1e-5` on random daily head); `5/5` v1 and `4/4` v2 ratchet decisions matched (`v2.1` raw log/frozen tensors unavailable).

**Second-pass partial out-of-order & fault-attribution extensions (`home=False` irradiance vs `home=True` power-only).**
- **Single out-of-order block in an otherwise sorted 30-day month:**
  - Moving **only day 0 to the end** (`day0_at_end`, 29 of 30 days ascending) collapsed **both** sites to **`0` valid days and `0.00 kWh`** in both modes.
  - Swapping **only two adjacent days** (`swap_d14_d15`) silently dropped 1 day (`30 → 29`) at both sites and fabricated phantom power-only faults: on `CA_2107` (`home=True`) `outage` jumped **`7.4×`** (`0.53% → 3.94%`); on `CO_9068` (`home=True`) `outage` jumped **`9.7×`** (`0.19% → 1.84%`) and `snow` jumped from **`0.00% → 1.71%`** (`0.26%` even with irradiance).
- **Midday `10:00–15:00` gap fabricates severe fault attributions in both modes** (while `all_nan_P` cleanly returns `0` days / `0.00 kWh`):

| Site & mode | Input variant | Valid days / `kwh_made` | Key reported losses (`%` of possible) |
|---|---|---:|---|
| `CA_2107` (`home=False`) | Original 30 d | `30` / `7,248.35` | `outage 0.48%`, `clipping 1.24%`, `soiling 0.41%` |
| `CA_2107` (`home=False`) | `10:00–15:00` missing | `30` / `2,840.16` | **`outage 29.57%` (`61.6×`)**, **`shading 1.12%`** |
| `CA_2107` (`home=True`) | Original 30 d | `30` / `7,248.35` | `outage 0.53%`, `clipping 1.50%`, `soiling 0.99%` |
| `CA_2107` (`home=True`) | `10:00–15:00` missing | `30` / `2,840.16` | **`outage 32.67%` (`61.6×`)**, **`shading 3.62%`** |
| `CO_9068` (`home=False`) | Original 30 d | `30` / `54,311.03` | `outage 0.23%`, `shading 2.36%`, `stuck_trackers 2.84%` |
| `CO_9068` (`home=False`) | `10:00–15:00` missing | `30` / `17,855.38` | **`shading 10.95%` (`4.6×`)**, **`stuck_trackers 5.65%`**, `outage 2.45%` |
| `CO_9068` (`home=True`) | Original 30 d | `30` / `54,311.03` | `outage 0.19%`, `shading 3.54%`, `stuck_trackers 2.19%` |
| `CO_9068` (`home=True`) | `10:00–15:00` missing | `30` / `17,855.38` | **`shading 15.29%` (`4.3×`)**, `outage 1.38%`, `soiling 1.42%` |

**What remains open.** Ordered archived exam records were not re-evaluated, and interval coverage under daytime dropouts plus raw `v2.1` frozen-tensor rescoring remain unavailable in the repo.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/sundoctor_v2_1_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/wave5_export_probe.py sundoctor
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_759c345_probe.py sundoctor
```

Uses NumPy, pandas, and CPU PyTorch. [Second-pass probe](reviews/reviewer-a/peer_759c345_probe.py)

## Resolution (27 Nov 2026)

| Finding | Real? | Action | Evidence |
|---|---|---|---|
| Newest-first or out-of-order CSV rows collapse to 0 days | Yes | **Fixed**: `diagnose_csv` sorts rows by time and drops duplicate timestamps. | An ascending file is unchanged; golden demo unchanged. |
| Midday gaps count as zero energy and read as outages | Yes | **Warned, opt-in fix, re-score queued**: counted days missing > 2 h of 08:00–16:00 solar-time readings trigger a warning; `diagnose(min_daylight=0.9)` drops them. Default keeps the exam's half-day rule. | — |
