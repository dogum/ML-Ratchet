# TsunamiNet v1.0 | Empty-input sentinel leak, target lead bounds, and API calibration defaults

**Cross-checked empty-input public API defect and forecast-lead boundary.** Reviewer A independently reran reviewer B's `d903cb6` probe against model base `61c70bf`, audited weight health across all 73 arrays (including verifying all 8 RealNVP coupling output masks), and checked held-out target amplitude dependence under zeroed waveforms. [Original report and probe](reviews/reviewer-b/tsunaminet-v1.0.md) · [Encoder & posterior](../../models/tsunaminet/tsunaminet.py#L56) · [Archived exam rows](../../models/tsunaminet/training/logs/exam_rows.json)

**What failed and reproduced.**
- **Zero-buoy sentinel leak in `posterior(ev)`:** Passing full-length (`1,441`-minute) arrays with all telemetry masked yields `0` selected input buoys, and [`tsunaminet.py:64`](../../models/tsunaminet/tsunaminet.py#L64) feeds the `-1e4` max-pooling sentinel directly into the context network without an empty-set guard. All four packaged events returned **`Mw −2053.32`** with nominal 68% interval **`[−2055.30, −2051.31]`** under the default seed. `demo.py` guards `if not ev['inputs']`, and all 28 archived exam events had at least one usable reporting buoy, so the defect affects public API callers rather than archived exam scores.
- **Input-window endpoint vs. target arrival:** Across all **119** archived held-out forecasts, comparing the latest scheduled input-window end (`arrival + 69 min`) against target predicted arrival gave median nominal lead **352 min** (range **`−66` to `+1,082` min**), with **5 of 119** target arrivals and **7 of 119** target-window starts at or before the nominal input endpoint. Perturbing processed samples strictly after the latest input endpoint changed zero input tensor entries on all four examples, while upstream detiding ([`training/dart.py`](../../models/tsunaminet/training/dart.py#L1)) uses up to 34 subsequent samples for glitch removal.
- **API default & docstring discrepancies:** Calling `GreensFunctions()` without attaching `net.cal` and passing `widen=net.cal['widen']['amp']` shifts British Columbia's first target forecast from **`1.76 cm [0.87, 3.54]`** (demo/calibrated) to **`1.12 cm [0.79, 1.57]`**. Additionally, [`tsunaminet.py:10`](../../models/tsunaminet/tsunaminet.py#L10) documents `gf.forecast(ev, post, site, eta, mask)`, whereas the actual signature takes window bounds `(ev, post, site, a, b, mask=None, ...)`.

**Second-pass checks (weights, amplitude ablation, parity, and ratchet).**
- **Weights:** All numeric arrays in `tsunaminet_weights.npz` (`73` keys) are finite with **0 dead active units**; the 4 zero output rows in each of `f0w2..f7w2` correspond 1-to-1 with the masked pass-through dimensions of the 8 RealNVP affine coupling layers.
- **Waveform dependence on both `Mw` and target amplitudes:** Zeroing input waveforms while preserving epicentres, station geometry, arrivals, and masks collapsed `Mw` to `7.40–7.45` across all four events and dropped held-out target median amplitudes sharply on all three events with targets (`5467_20121028` / `46402`: **`1.63 -> 0.26 cm`**; `5557_20140401` / `21414`: **`0.70 -> 0.14 cm`**; `5658_20170908` / `32412`: **`0.65 -> 0.41 cm`**).
- **Scoring, parity, and ratchet:** Outcome SNR filtering (`peak > 5 × pre-arrival noise`) reproduced all 18 target IDs on the four examples (adding the 9 excluded stations moved mean log-amplitude error from `0.22560` to `0.22562` dex). Torch↔NumPy parity reproduced within `1.61e-6` (`6.5e-7` on source samples), dev (5) and exam (28) IDs are disjoint, and all 6 logged rungs replayed cleanly.

**What remains open.** Computing the real-time issuance-lead distribution from raw DART timestamps and telemetry/detiding latency across all 28 exam events—and testing a separately trained metadata-only baseline across the full population—requires raw DART archives outside the repository.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/tsunaminet_v1_0_probe.py
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_d903cb6_probe.py tsunaminet
```

Uses NumPy, SciPy, and CPU PyTorch. [Second-pass probe](reviews/reviewer-a/peer_d903cb6_probe.py)

## Resolution (27 Nov 2026)

| Finding | Real? | Action | Evidence |
|---|---|---|---|
| Zero buoys → sentinel reaches the network (Mw −2053) | Yes | **Fixed**: `posterior` raises `ValueError` when no buoy is usable. | All 28 exam events had one: exam unchanged. |
| `GreensFunctions()` without calibration; docstring signature wrong | Yes | **Fixed**: the calibration loads from the released weights by default (`cal=` overrides); the module docstring shows `forecast(ev, post, site, a, b, mask)`. | The demo sets `gf.cal` itself and is unchanged. |
| 5 of 119 target arrivals at or before the input endpoint | Yes (limit) | Documented in the card. | — |
