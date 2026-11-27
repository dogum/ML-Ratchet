# QuakeNet v2 | Catalogue arrival shortcut vs. waveform reliance

**Cross-checked critical exam-contract defect (for an independent-location claim).** Reviewer A independently reran reviewer B's `d92a17b` probe against model base `61c70bf`, verified GNN weight health and Torch↔NumPy parity, and tested the shipped network's own dependence on waveforms versus the catalogue `p_guess` input across all four packaged 10-station exam earthquakes. [Original report and probe](reviews/reviewer-b/quakenet-v2.md) · [Arrival fetch](../../models/quakenet-v2/training/fetch2.py#L58) · [Release locate](../../models/quakenet-v2/quakenet2.py#L137)

**What reproduced.** The split manifest has 40 unique dev IDs and 1,734 unique exam IDs (400 multi-station, zero dev/exam overlap), and all 15 logged promotion decisions matched the strict `+0.01` / `1.05` dev-gate rule. Fitting only station coordinates and relative catalogue arrivals (`t_start + p_guess`) with the shipped ak135 table recovered all four 10-station exam epicentres within **4.24 m** (`1.12–4.24 m`) without opening any waveform.

**Does the shipped network actually rely on waveforms?** To test the open question left by the first pass, the second-pass probe holds station coordinates and `t_start + p_guess` fixed while replacing all waveforms with zeros or white noise, and separately holds real waveforms fixed while jittering `p_guess` by uniform `±15 s` or `±60 s` (seed 123):

| Exam event (10 stations) | Real waveforms + catalogue `p_guess` | Zero waveforms (same `p_guess`) | White-noise waveforms (same `p_guess`) | Real waveforms + `±15 s` `p_guess` jitter | Real waveforms + `±60 s` `p_guess` jitter |
|---|---:|---:|---:|---:|---:|
| `us6000ngn7` | 103.0 km (`Mw 5.70`) | 16,307.4 km (`Mw −10.91`) | 840.5 km (`Mw 8.66`) | 460.1 km | 955.1 km |
| `us7000kulj` | 24.7 km (`Mw 6.09`) | 2,156.3 km (`Mw −11.18`) | 8,279.6 km (`Mw 8.12`) | 24.7 km | 391.6 km |
| `us7000nx17` | 71.0 km (`Mw 5.96`) | 7,055.9 km (`Mw −11.10`) | 519.3 km (`Mw 8.62`) | 112.5 km | 241.0 km |
| `usb000scur` | 128.6 km (`Mw 5.54`) | 7,066.2 km (`Mw −11.06`) | 6,308.0 km (`Mw 8.64`) | 157.7 km | 325.4 km |

Replacing waveforms with zeros or white noise destroys location accuracy (`519–16,307 km`) and magnitude (`−11.2` / `+8.7`), confirming the released network genuinely reads the waveforms rather than solving the catalogue shortcut directly. At the same time, jittering `p_guess` beyond the STA/LTA pick window (`±60 s`) degrades error to `241–955 km` (and `±15 s` already moves `us6000ngn7` from `103` to `460 km`), confirming that the catalogue-assisted pick window remains load-bearing for the exam's reported location accuracy.

**Independent implementation checks.** Both shipped archives (`quakenet_weights.npz`, `quakenet2_weights.npz`) have zero NaN/Inf and zero dead units, Torch↔NumPy parity for the v2 GNN (`training/gnn.py` vs `quakenet2.py`) held within `1.33e-6` across all four exam events, `sigma_factor` is strictly widen-only (`epi 1.305, z 1.0, mw 1.016, t0 1.0`), and [`training/train2.py:65`](../../models/quakenet-v2/training/train2.py#L65) seeds both PyTorch and NumPy.

**What remains open.** End-to-end performance when stations and P-wave arrivals are detected independently from continuous recordings without catalogue assistance—and full-exam error across all 400 multi-station events under independent picks—remain unmeasured.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/quakenet_v2_probe.py
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_dc30c02_probe.py quakenet-v2
```

The second-pass probe uses NumPy, SciPy, and CPU PyTorch. [Second-pass probe](reviews/reviewer-a/peer_dc30c02_probe.py)
