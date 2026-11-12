> **Lab notebook.** The build notes as recorded during the original work in a cloud sandbox (Oct–Nov 2026), kept
> as written. Paths like `/home/claude/...` and `/mnt/user-data/outputs` refer to that sandbox: in this repository
> the code is in `training/` and everything it writes goes to `work/` (see `training/_paths.py`). Page links to
> claude.ai are the original private copies; the lab now lives at `lab/index.html`.

# SwellCast — status (v1.0, 6 Nov 2026)

One ocean buoy, one distant storm: from ten days of NOAA NDBC wave spectra (energy plus first directional moments), a small CNN says whether storm swell is arriving, how far away the storm was, when it peaked and which direction it lies in. This is Munk's 1963 idea: swell frequency at the buoy rises linearly in time, df/dt = g/(4πD). The network was trained only on simulations laid on real buoy records.

- Page (network runs live in the browser; real-exam explorer with spectrogram and buoy-centred globe; storm lab): https://claude.ai/artifact/PiqMh45i3co2RguewgUg9k
- Release: `/mnt/user-data/outputs/swellcast-v1.0.zip` (3.2 MB). It holds:
  - numpy-only network (BatchNorm folded; numpy vs torch 1.7e-5)
  - NDBC pipeline (raw files give the same window as training: 0.0 difference)
  - weights, model card, README
  - demo: raw 51004 files for Hurricane Olivia 2018, plus 5 exam windows drawn at random; runs in 0.4 s
  - all training scripts and result JSON
- Code: `/home/claude/swell/` (src, bank, ckpt, STATUS.md).

## Recipe as built
- **Physics** (`src/swellsim.py`):
  - c_g = g/(4πf)
  - moving, growing and decaying JONSWAP storm patch (7 sub-areas), partly beamed along the storm heading
  - spherical spreading on the great circle
  - Ardhuin et al. 2009 swell dissipation
  - directional moments
  - χ²(24) estimation noise and NDBC 0.01 m²/Hz quantisation
- **Physics check** (`physics_check.json`):
  - group speed exact
  - ridge slope vs g/(4πD): within 1.4% at 1,500 km and 0.1% at 9,000 km; t0 from the intercept within 0.55 h
  - energy received/emitted 1.0003
  - dissipation e-folding scales 3,800 km to over 100,000 km, the Ardhuin range
- **Real case** (`physics_real.json`, Hurricane Marie 2014, not in dev or exam): the real ridge at 4 SoCal buoys runs parallel to the IBTrACS-predicted ridge. 0.0625 Hz arrives about 19 h late because the storm stayed strong while moving closer.
- **Real noise:** 240-h windows from 27 NDBC buoys (2008–2025). Each is ≥70% complete, has no candidate hurricane window, and is below the buoy's 60th percentile of long-period energy. 25% of each batch is a real null window (detection negatives).
- **Truth:** IBTrACS v04r01, storms with peak wind ≥64 kt. The truth point is the time and place of peak wind. Pairs are 1,000–8,000 km apart over open water, 2008–2025, with crowded pairs removed.
  - "Visible": energy on the true-storm ridge is ≥2× what it was 48 h before.
  - "Confirmed" (headline): visible, and the buoy's measured ridge direction is within 45° of the storm.
  - Dev = 18 storms by seed 2026 (29 confirmed arrivals). Exam = every other storm: 288 confirmed, 632 visible, 2,081 pairs in all.
- **Ratchet:** a rung is kept if it scores > +0.01 on the frozen simulated set (mean skill over log D, t0, direction, log Hs and detection) and the dev errors get no more than 5% worse. Frozen sets val1 → val2 → val3; every rung was re-scored in order on each.

## Rungs (final scoring on val3)
| Rung | Change | val3 | Dev D/t0/dir (median) | Decision |
|---|---|---|---|---|
| r1 | 296k CNN, 30k v1 sims (v1 = textbook priors) | 0.386 | 40% / 35 h / 16° | kept |
| r2 | v2 sims, calibrated on dev: weaker, longer-lived storms; shorter periods; r1 0.35–0.8; crossing swells; NDBC low-f cutoff | 0.442 | 58% / 50 h / 31° | rejected (dev +30%) |
| r3 | v3: 65% of storms are real IBTrACS 1985–2007 hurricane tracks (Hs = a·Vmax) at their real place relative to the buoy | 0.582 | 46% / 56 h / 24° | rejected (dev +21%) |
| r4 | Mix of v1 + v2 + v3 banks (54k), fine-tuned from r1. **Released** | 0.536 | 37% / 37 h / 22° | kept (dev +2%) |
| r5 | r4 + 3,000 steps at half the learning rate | 0.545 | 43% / 36 h / 19° | rejected (+0.009 < 0.01) |

- **Dev calibration** (`calib_v2_record.json`): the v1 sims were too strong (ridge anomaly 1.49 vs 0.87 real), too long-period (0.0575 vs 0.0725 Hz) and too directional (r1 0.79 vs 0.47).
- **Network twin test** (`src/twin_net.py`): each dev storm is re-simulated from its real track into quiet records of the same buoy. r1's errors on the twins correlate with its real errors (log D r = 0.60, t0 r = 0.65), so storm behaviour explains much of the error. That motivated v3. Twin MAE is still smaller than real (r4: log D 0.124 vs 0.207; t0 30 h vs 45 h).

## Exam (288 confirmed arrivals, never used for any choice)
| Method | D err median | log D MAE | t0 err median | dir err median | detected | false alarms (quiet 2025) |
|---|---|---|---|---|---|---|
| SwellCast r4 | 38% | 0.200 | 33 h | 19° | 63% | 11% |
| Classical ridge fit f = a(t−t0) | 59% | 0.326 | 48 h | 20° | 43% | 10% |
| Dumb (dev median: 2,364 km, −48 h, 171°) | 28% | 0.175 | 24 h | 56° | always | 100% |

- **Coverage of the 68% bars** (widen-only factors from dev: D ×1.19, t0 ×1.45, dir ×1.33): D 67%, t0 74%, dir 70%.
- **By basin:**

  | Basin | n | D | t0 | dir |
  |---|---|---|---|---|
  | EP | 138 | 32% | 30 h | 18° |
  | NA | 119 | 36% | 30 h | 19° |
  | WP | 29 | 52% | 73 h | 45° |
- **By distance:** 5,000–8,000 km is worst (D 52%, t0 59 h).
- **Looser sets:**

  | Set | Network D / t0 / dir | Dumb D / t0 |
  |---|---|---|
  | Visible (632) | 42% / 39 h / 36° | 38% / 30 h |
  | All pairs (2,081) | 44% / 41 h / 43° | 40% / 33 h |

## Limits
- **Doesn't beat the dumb median guess on distance or time.** The exam geography is narrow: Mexican hurricanes to California, Atlantic hurricanes to the US east coast. On top of that, the window is centred within ±36 h of the true arrival, a hint every method gets. The network beats the classical fit on distance, time and detection, ties it on direction, and has calibrated bars.
- **Truth is tropical cyclones only.** Southern Ocean and North Pacific winter storms are not graded.
- **One-point truth** (peak wind) for a storm that makes swell for days along a track. The twin test says this is a big part of the error.
- **The confirmed set favours direction-readable events.** Selection uses the buoy's measured direction, which helps the classical direction reading.
- **v2/v3 sims match dev statistics better, but alone they make real dev readings worse.** The cause is not understood.

## Next rungs worth trying
1. **Sliding-window search:** scan the record without the ±36 h centring hint, and score detection plus localisation. This is the honest monitoring task and removes the dumb baseline's edge.
2. **Track-level truth and loss:** predict the effective source (energy-weighted track point) or the whole track segment. Alternatively, weight v3 track sims more once the v2/v3 dev regression is understood: hybrid-swap test of real vs twin parts, as QuakeNet did.
3. **Two-buoy consistency:** the same storm at two buoys at different distances; joint triangulation. Also extratropical truth from reanalysis if a documented endpoint is reachable (Open-Meteo hit its daily limit from this IP).

## Sandbox gotchas
- **Open-Meteo archive and marine API:** "Daily API request limit exceeded" (shared IP). Not used.
- **Container restarts:** three during the build. `train.py` resumes from `ckpt/<name>.resume`, and `simbank.py` resumes per 1,000-sample chunk. `setsid` forks, so read real PIDs from `ps`.
- **Bank speed:** about 70 s per 1,000 v1 samples alone, 3–4 min per 1,000 under 4-agent load. v3 track sims take about 0.26 s each.
- **Training speed:** about 0.3 s/step when the machine is quiet, 1 s/step under load.
