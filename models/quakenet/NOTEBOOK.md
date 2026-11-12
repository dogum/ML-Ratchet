> **Lab notebook.** The build notes as recorded during the original work in a cloud sandbox (Oct–Nov 2026), kept
> as written. Paths like `/home/claude/...` and `/mnt/user-data/outputs` refer to that sandbox: in this repository
> the code is in `training/` and everything it writes goes to `work/` (see `training/_paths.py`). Page links to
> claude.ai are the original private copies; the lab now lives at `lab/index.html`.

# QuakeNet — status (v1.2, 5 Nov 2026)

One seismometer, one earthquake: from 41 minutes of three-component ground velocity at a single GSN station, it estimates Mw, distance, depth and back-azimuth, each with an error bar. It was trained only on simulations.

- Page (runs the network live in the browser, with noise and record-cutting controls): https://claude.ai/artifact/DNinPsWLG3EEv2ezCygYJn
- Release: `quakenet-v1.2.zip`, containing:
  - numpy/scipy inference
  - pipeline
  - weights
  - 4 real example records
  - model card
  - all training and diagnostic scripts

## Recipe as built

- **Physics:**
  - Syngine ak135f_2s Green's functions on 27 depths × 71 distances (25–95°), combined locally into any moment tensor.
  - Verified against Syngine full synthetics: correlation 0.9999 at the equator.
  - Convention found by search: all signs +, az' = 180° − az.
- **Simulator:**
  - random mechanism plus CLVD; source duration scaled to size
  - travel-time stretch, amplitude scatter, misorientation
  - injected into 1,284 real quiet-time GSN noise windows
- **Real data:**
  - USGS catalog M ≥ 5.5, 2012–2025
  - IU/II LH channels; response removed; ZNE; 0.008–0.35 Hz; 1 sps
  - 3,179 records: 74 dev (40 events), 3,105 exam (1,734 events)
- **Ratchet:** a rung is kept if it scores more than +0.01 on the frozen simulated set and the dev records get no more than 5% worse.

## Rungs (final re-scoring on val8)

| Rung | Change | Sim (val8) | Exam: Mw | distance | depth (median) | back-azimuth |
|---|---|---|---|---|---|---|
| 1 | Textbook ak135 | 0.556 | 0.21 | 11.0° | 110 km | 8.7° |
| 2 | Slower surface waves: warp after S, median 1.065, measured on dev | 0.690 | 0.15 | 7.4° | 72 km | 8.0° |
| 3 | Scattering coda: above 0.04 Hz, fitted on dev coda/P energy ratios | 0.808 | 0.14 | 5.7° | 52 km | 6.6° |
| 4 | World mix: 70% of depth/Mw drawn from the 1990–2011 catalog | 0.827 | 0.13 | 5.3° | 44 km | 6.4° |
| 5 | Layered Earth: water-column reverberation for offshore sources (82% of shallow quakes, from the 1990–2011 catalog and coastlines), Moho Ps and sediment reverberation at the station | 0.838 | 0.13 | 5.2° | 42 km | 6.2° |
| 6 | Weaker short-period surface waves: after S, 9–12 s ×0.2 and 12–18 s ×0.5 on all components (`calib_swq.py`). **Not kept on val8** (0.839 vs 0.838) | 0.839 | 0.13 | 5.2° | 43 km | 6.1° |
| 7 | Surface waves per component (`calib_love.py`, `crust.surface_zrt`): Z as rung 6, R nearly unchanged, T ×0.5 at 12–25 s, plus short-period Rayleigh delay 17 s/1000 km at 9–25 s. Fine-tuned from rung 6. **Released** | 0.869 | 0.13 | 5.1° | 39 km | 6.0° |

- **Rung 5** fell 0.002 short on val5 (0.869 vs 0.861) and was rejected at first. It passed on val7 when every decision was re-taken.
- **Rung 6** was trained from rung 4 on v4 + damping physics; layering and rupture-duration scatter are not in its training data. A rupture-duration-scatter run (layers + STF ×/÷1.6) was stopped in favor of the measured fix. Rung 6 passed on val7, but when rung 7 rebuilt the set (val8) it fell 0.001 short of rung 5.
- **Rung 7** is compared with rung 5 and kept (0.869 vs 0.838).
- **Frozen-set versions:** val, val2, val3, val4, val5, val7, val8. Every rung was re-scored on each.
- **Error bars:** one factor per output was fitted on dev, with a widen-only rule (Mw 1.00, dist 1.64, logz 2.42, baz 1.14). Exam coverage of the 68% bars:

  | Output | Coverage |
  |---|---|
  | Mw | 63% |
  | Distance | 66% |
  | Depth | 51% |
  | Back-azimuth | 68% |

## Against classical methods (exam, v1.2)

| Output | QuakeNet | Classical |
|---|---|---|
| Magnitude | 0.13 (bias +0.05) | Ms, dev-calibrated: 0.57 |
| Distance | 5.1° | S−P: 13.0° |
| Back-azimuth | 6.0° | P polarization: 43.7° |
| Depth | 46% within a factor of 2 | Guessing 31 km: 22% |

- Deep quakes (> 70 km): median error 52 km, against 136 km for the guess; 82% within a factor of 2.
- Shallow quakes (≤ 70 km): median error 33 km, against 21 km for guessing 31 km. They are still placed too deep: exam median ratio 3.7 (was 4.2 at rung 4), or 2.9 excluding USGS 10-km placeholders. Dev ratio 1.8 (was 2.3).
- One-station epicentre: median miss 764 km.

## Shallow-depth investigation (5 Nov, dev only)

- **Twin test** (`twin.py`): each dev quake was re-simulated from its USGS moment tensor at the same station, through the full simulator in real noise.
  - The network reads twins correctly (shallow median ratio 0.97) and real records at 2.2× the true depth.
  - The USGS centroid ("derived") depth doesn't explain the gap: real/centroid is 2.1.
- **Hybrid test** (`hybrid.py`): parts of the real record and its twin were swapped.

  | Combination | Depth read (× true) |
  |---|---|
  | Twin body + real surface waves | 2.0 |
  | Real body + twin surface waves | 0.98 |
  | Real short periods only | 1.44 |
  | Real horizontals only | 1.40 |
  | Real long periods only | 1.09 |
  | Real Z only | 1.13 |

  So the cue is in the surface waves after S, spread over short periods and horizontals.
- **Twin knobs:**

  | Knob | Effect on twin depth ratio |
  |---|---|
  | Damping short-period surface waves (became rung 6) | 1.00 → 1.38 |
  | Noise level up to ×2.5 | none |
  | STF ×0.5 | 0.93 |
  | STF ×2 | 1.30 |
  | Smearing short-period surface waves | none |
  | Realigning surface-wave timing | 2.23 → 1.93 |

- **Not the cause:**
  - Surface/P amplitude ratio (15–60 s): matches dev (−0.06 log).
  - Long-period amplitude: matches (high-SNR median +0.01).
  - Noise: the pre-P noise on real dev records is about 1.4× above the quiet pool (`QNOISE` flag exists; the twin test says it doesn't drive depth).

## Love waves and dispersion (5 Nov, dev only, `calib_love.py`)

- Real vs USGS-MT synthetic after S, shallow dev quakes, log10 RMS ratio:

  | Band | Z | R | T (Love) |
  |---|---|---|---|
  | 9–12 s | −0.70 | −0.19 | −0.22 |
  | 12–18 s | −0.31 | −0.06 | −0.31 |
  | 18–25 s | +0.21 | +0.16 | −0.32 |

  Rung 6 had applied the Z depletion to all components, which over-damped R. The per-component gains in rung 7 bring every band to within about ±0.1 (18–25 s Z and R stay around +0.2).
- Dispersion: real 12–25 s Rayleigh arrives later than ak135, relative to its own 25–80 s energy, by about 17 s per 1000 km (IQR −1 to 56). Love timing shows no clear trend.
- Twin test (rung 6 network, shallow depth ratio; real is 2.24):

  | Variant | Twin ratio |
  |---|---|
  | Baseline | 0.97 |
  | Love damping | 1.03 |
  | Dispersion | 0.97 |
  | Both | 1.01 |
  | Per-component gains | 0.89 |
  | Per-component gains + dispersion | 0.85 |

  None of these moves the twin toward the real reading. Trained anyway as measured physics: the dev shallow ratio improved 2.24 → 1.76, the exam barely (3.76 → 3.72).
- Conclusion: the remaining shallow-depth cue is not in the average surface-wave spectrum, Love energy or dispersion.

## Next rungs worth trying

- **Shallow depth, continued.** Candidates not yet tested:
  - Source-region structure (oceanic crust plus water at the source changes the excitation of surface waves against period and depth).
  - Overtone (higher-mode) surface waves.
  - Per-record surface-wave shape, not average spectra.

  A saliency map of the network on real shallow records against their twins would show where it looks.
- **Magnitude bias** rose to +0.05 with rung 7 (shallow +0.06). Watch it.
- **Very deep quakes (500–700 km)** come out too shallow (exam ratio 0.83). Check the deep-event STF and ak135f Q.
- **Wider coverage:** BH channels for M4.5+, distances under 25° (needs a regional GF grid).
- **Multi-station:** combine 2+ stations, which turns the one-station locator into a real one.

## Sandbox gotchas

- Syngine needs `model=ak135f_2s` (prem_i_2s returned 204) and `sourcewidth` ≥ 2.1. Strip '+' from exponents in moment-tensor strings, or the request returns 422.
- The container can restart mid-training. train.py writes a `.resume` checkpoint every 500 steps; launch jobs with `setsid nohup`.
- Bank generation alongside training starves the trainer even with `nice`. Generate banks first.
- simbank env flags:
  - QPRIOR=1 (world mix)
  - QSW (physics version: 6 = damping, 7 = per-component + dispersion)
  - QLAYERS
  - QSTF
  - QNOISE
