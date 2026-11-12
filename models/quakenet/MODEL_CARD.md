# QuakeNet v1.2 — model card

**What it does.** It reads one seismometer: 41 minutes of three-component ground velocity at a single station, from 60 s before the P wave to 2,400 s after it. It returns the earthquake's moment magnitude, epicentral distance, depth and back-azimuth (the compass direction from the station toward the quake), each with an error bar. Distance plus back-azimuth places the epicentre from that one station.

**How it learned.** Only from simulations. There is no real earthquake in its training data.

## Inputs

- `zne`: (3, n) vertical, north and east ground velocity in m/s, instrument response removed, 1 sample/s, band-passed 0.008–0.35 Hz (`pipeline.SOS_BAND`). Use long-period (LH) or decimated broadband data.
- `p_guess`: the sample where the P wave is expected, to within about 30 s. A recursive STA/LTA picker refines it.
- Valid range: distance 25–95°, magnitude about 5.5–8, any depth from 0 to 700 km.

## Outputs

| key | meaning |
|---|---|
| `mw`, `mw_sig` | moment magnitude and 68% error bar |
| `dist`, `dist_sig` | epicentral distance, degrees |
| `depth`, `logz`, `logz_sig` | depth in km; the 68% error bar is on log10(depth) |
| `baz`, `kappa` | back-azimuth in degrees, with von Mises concentration (68% half-width ≈ 57°/√kappa) |
| `p_pick`, `p_strength` | the P pick used and its STA/LTA peak |

Error bars are widened by one factor per output, fitted on the 74 development records only: magnitude ×1.00, distance ×1.64, log depth ×2.42, back-azimuth ×1.14. The rule only ever widens. Pass `calibrated=False` to `predict_windows` to get the raw simulation-trained bars.

## Physics and simulation

- **Green's functions:** ak135f (2 s) from EarthScope Syngine, on 27 depths (2–700 km) × 71 distances (25–95°). The ten basis seismograms per grid point are combined locally into any moment tensor.
- **Physics check:** local synthesis matches Syngine's full synthetics with correlation ≥ 0.9999 on equatorial receivers, where both use the same distance. Off the equator it is 0.86–0.9, because the service uses ellipsoidal distances.
- **Source:**
  - random double couple plus a small CLVD part (σ = 10% of M0)
  - triangle source-time function with half-duration 1.05×10⁻⁸ · M0^(1/3) (M0 in dyne·cm)
- **Augmentations for what a 1-D Earth leaves out:**
  - travel-time stretch ±1.5%
  - amplitude scatter ×/÷1.25
  - sensor misorientation σ = 3°
- **Corrections measured on development events:**
  - **Surface waves:** everything after S is time-stretched by a factor drawn around 1.065 (log-σ 0.15), the median delay of real Rayleigh waves against ak135.
  - **Scattering coda:** above 0.04 Hz, each component gets the three input components convolved with exponentially decaying random noise. Amplitude is 0.2 × 10^N(0, 0.3), decay 60–250 s. The values were fitted to the real ratio of coda to P energy.
  - **Surface waves, component by component (v1.2):** after S, each of vertical, radial and transverse gets its own gain, measured on shallow development quakes against synthetics of the same quakes (`training/calib_love.py`, `crust.surface_zrt`). All three gains are scaled per record by U(0.5, 1.5).
    - Vertical: ×0.2 at 9–12 s, ×0.5 at 12–18 s.
    - Radial: ×0.65 at 9–12 s, nearly unchanged above.
    - Transverse (Love): ×0.5 at 12–25 s.
    - Short-period Rayleigh waves (vertical and radial) also get an extra group delay against their long periods: 17 s per 1,000 km at 9–25 s and 4 s at 25–35 s. It is scaled per record by a lognormal (σ 0.8), because paths differ.
  - **Earlier version, replaced:** v1.1 applied the vertical's depletion to all three components (`crust.damp_short_sw`), which over-damped the horizontals.
  - **Tested, not in the released model's training data:** water-column reverberation for offshore sources and crust/sediment conversions under the station (rung 5, `crust.apply`), and rupture-duration scatter (stopped).
- **Noise:** every simulated record is added into a real quiet-time window, 1,284 windows from the same 127 GSN stations.
- **Source mix:** 70% of training quakes take their depth and magnitude from the 1990–2011 USGS catalogue (before every exam event). The other 30% are uniform: 60% shallow, Mw 5.4–8.0.

## Network

- **Architecture:** 1-D residual CNN, 7 stride-2 blocks (widths 32–192), 1.44 M parameters.
- **Input:** the normalized traces, their log envelopes (9 s smoothing), the log10 amplitude scale and the pick strength.
- **Loss:** β-NLL (β = 0.5) on Mw, distance and log10 depth; von Mises NLL on back-azimuth.
- **Training:** EMA weights. Each rung fine-tunes for 5,000 steps on 80,000 fresh simulated records. The released network (rung 7) was fine-tuned from rung 6, which descends from rungs 1–4.

## Ratchet

A rung is kept only if both of these hold:

1. It beats the incumbent by more than 0.01 on a frozen simulated test set. The score is the mean skill over Mw, distance, log depth and back-azimuth.
2. It does not get more than 5% worse on 40 real development earthquakes (74 records).

When the simulator's physics changed, the frozen set was rebuilt and every rung was re-scored and re-decided. The final set, `val8`, uses the world-mix source prior and the per-component surface waves. Rung 5 fell 0.002 short of the margin on `val5`, then passed when every decision was re-taken. Rung 6 passed on `val7` but falls 0.001 short on `val8`, so it is not kept. Rung 7 is compared with rung 5.

| rung | change | sim score | exam Mw MAE | exam distance MAE | exam depth median | exam back-azimuth median |
|---|---|---|---|---|---|---|
| 1 | ak135 in real noise | 0.556 | 0.21 | 11.0° | 110 km | 8.7° |
| 2 | slower surface waves | 0.690 | 0.15 | 7.4° | 72 km | 8.0° |
| 3 | scattering coda | 0.808 | 0.14 | 5.7° | 52 km | 6.6° |
| 4 | world's depth/magnitude mix | 0.827 | 0.13 | 5.3° | 44 km | 6.4° |
| 5 | ocean and station layering | 0.838 | 0.13 | 5.2° | 42 km | 6.2° |
| 6 | weaker short-period surface waves, all components (not kept) | 0.839 | 0.13 | 5.2° | 43 km | 6.1° |
| 7 | surface waves component by component (released) | 0.869 | 0.13 | 5.1° | 39 km | 6.0° |

## Exam

The exam set is 3,105 records of 1,734 earthquakes (M ≥ 5.5, 2012–2025) at 127 IU/II stations. These records were never used to train, tune or choose anything.

| | QuakeNet | classical, same windows |
|---|---|---|
| Magnitude, mean abs error | **0.13** (bias +0.05) | 0.57 (Ms, dev-calibrated offset) |
| Distance, mean abs error | **5.1°** | 13.0° (S−P with ak135, found on 79%) |
| Back-azimuth, median error | **6.0°** (85% within 30°) | 43.7° (P polarization) |
| Depth within a factor of 2 | **46%** | 22% (always guess 31 km) |
| Depth, quakes > 70 km, median error | **52 km** | 136 km (guess 31 km) |
| Depth, quakes ≤ 70 km, median error | 33 km | **21 km** (guess 31 km) |
| One-station epicentre, median miss | 764 km (31% within 500 km) | — |

Coverage of the calibrated 68% bars on the exam:

| output | coverage |
|---|---|
| Mw | 63% |
| distance | 66% |
| log depth | 51% |
| back-azimuth | 68% |

## Limits and known failure modes

- **Shallow depth.** QuakeNet tends to place shallow quakes too deep. For quakes in the top 70 km, always guessing 31 km beats it on median error. 22% of exam quakes carry the USGS default depth of 10 km, which is a placeholder.
  - Twin test (`training/twin.py`): each development quake was re-simulated from its USGS moment tensor at the same station. QuakeNet reads the twins at the right depth (median ratio 0.97) and the real records about twice too deep (2.2).
  - Hybrid test (`training/hybrid.py`): the cue sits in the surface waves after S, mostly their short periods and horizontal components.
  - Things that moved the twin little or not at all: noise level up to ×2.5, rupture duration ×0.5–2, smeared short-period surface waves, realigned arrival times, Love-wave energy and short-period Rayleigh dispersion.
  - Rung 7 made the per-component surface-wave spectra match the real ones and cut the development shallow-depth ratio from 2.2 to 1.8. The exam shallow ratio barely moved (3.76 → 3.72; 2.9 excluding the 10-km placeholders). What remains is not in the average surface-wave spectrum.
  - Magnitude bias grew from +0.03 to +0.05 (shallow +0.06).
- **Very deep quakes.** Quakes at 500–700 km are often placed too shallow (median ratio 0.87).
- **Depth error bars** are too narrow even after calibration.
- **Range.** Outside 25–95°, below M5.5 or with short-period data, the input is out of distribution.
- **One station.** Location is coarse. This is a one-station tool, not a network locator.
- **P time.** It needs a P time within about 30 s.

## Data credits

- **USGS:** ComCat earthquake catalogue and moment tensors.
- **EarthScope Consortium / IRIS DMC:** waveforms and the Syngine Green's functions.
- **Networks:** IU (GSN, IRIS/USGS) and II (GSN, IRIS/IDA).
- **Natural Earth:** coastlines on the page.
