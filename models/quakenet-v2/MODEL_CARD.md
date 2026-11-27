# QuakeNet v2 — model card (v1.0, 22 Nov 2026)

**What it does.** It reads 2 to 10 seismometers that recorded the same earthquake and returns where the earthquake was
(epicentre, with a 68% error ellipse), how deep, how big (Mw) and when it started (origin time), each with an error bar.

**How it works.** Each station is read by QuakeNet v1.2, unchanged: distance, back-azimuth, Mw, depth and a P pick.
Those readings are combined classically into an anchor. A graph network, with stations as nodes and station pairs as
edges, predicts the correction to that anchor from the readings, the station geometry and the P arrival times.

**How it learned.** Only from simulations. No real earthquake is a training example.

## Inputs

For each of 2-10 stations (`QuakeNet2.locate`):

- `zne`: (3, n) vertical, north and east ground velocity in m/s, response removed, 1 sample/s, band-passed
  0.008-0.35 Hz (`pipeline.SOS_BAND`), as QuakeNet v1.2 wants it;
- `lat`, `lon`: the station's coordinates;
- `t_start`: the time of the first sample on any common clock (e.g. a UNIX timestamp);
- `p_guess`: the sample where P is expected, within about 30 s. v1.2's STA/LTA picker refines it.

Valid range: 25-95° from the quake, M about 5.5-8, depth 0-700 km. Long-period (LH) data.

## Outputs

| key | meaning |
|---|---|
| `lat`, `lon`, `cov_km2`, `err_ellipse`, `epi_sig_km` | epicentre; 2×2 covariance (north, east) in km²; 68% ellipse semi-axes (km) and orientation |
| `depth`, `logz`, `logz_sig` | depth (km); the 68% bar is on log10(depth) |
| `mw`, `mw_sig` | moment magnitude and 68% bar |
| `t0`, `t0_sig` | origin time on the input clock, and 68% bar (s) |
| `anchor` | the classical combination of v1.2's readings |
| `stations` | v1.2's per-station readings and picks |

Error bars are widened by one factor per output, fitted on the 40 dev quakes only (widen-only): epicentre ×1.31,
depth ×1.00, Mw ×1.02, origin time ×1.00. `calibrated=False` returns the raw bars.

## Physics and simulation

- **Per station, QuakeNet v1.2's simulator (physics v7):** Syngine ak135f Green's functions (27 depths × 71 distances),
  random double couple + CLVD, GCMT source duration ×/÷1.6, misorientation, travel-time stretch, slower surface waves,
  scattering coda, ocean and station layering, per-component surface-wave gains, site amplitude scatter.
- **New for many stations:**
  - one source per event, shared by every station (epicentre, depth, Mw, mechanism, source duration, ocean layer);
  - 2-10 real GSN stations at 25-95°, with real noise from the same station where the pool has it (80%);
  - exact distances: the nearest 1° Green's function is stretched so that S−P matches ak135 at the true distance;
  - absolute timing: P arrives at the ak135 time plus a station residual N(0, 1.5 s); records are cut 300 s before the
    ak135 prediction, as the real ones were; picks carry the simulated picker's own errors;
  - epicentres from the 1990-2011 USGS catalogue (80%, jittered 1.5°) or uniform on the sphere (20%).
- **Physics checks:** the ak135 table (bilinear) matches direct TauP within 0.078 s and iasp91 within 0.24 s; distances
  match ObsPy to 6e-14°. Inherited: local synthesis vs Syngine, correlation 0.9999.
- **Dev twins and physics v2/v3.** Each dev quake was re-simulated from its USGS moment tensor at the same stations.
  Paired with the real records (399), v1.2 reads real records +1.5° farther (median), with more scatter in distance and
  back-azimuth (robust sd 8.6° real against 5.5° twin), reads shallow quakes 2.4× too deep (log10 +0.37), and real P picks
  are tighter than simulated ones (robust sd 8.9 s against 11.8 s). Physics v2 (distance, back-azimuth, picks) and v3
  (depth and Mw) add these measured differences to the simulated readings.
- **Bank:** 10,000 events (60,093 station records) → 26,683 training graphs (all stations, 2 stations, a random
  subset in between). Frozen set: 2,000 events → 3,785 graphs.

## Network

- Graph network, 115k weights. Node features (16): v1.2's readings relative to the anchor and their log error bars,
  amplitude scale and STA/LTA strength, distance and direction from the anchor, P pick residual at the anchor and the
  ak135 slowness. Edge features (6): residual difference, station separation, direction and distance differences.
- Node encoder, 3 rounds of message passing (width 64; message MLP over sender, receiver and edge; mean over
  neighbours; residual update), mean + max readout, output MLP.
- Outputs: epicentre offset (north, east) from the anchor with a bivariate Gaussian (Cholesky), and offsets of log10
  depth, Mw and origin time with Gaussian bars. Loss: bivariate NLL + β-NLL (β 0.5). P residuals redrawn every batch.
- numpy inference matches torch to 6e-6; the lab page's JavaScript matches Python within 0.14 km, 0.05 km depth, 2e-5 Mw.

## Ratchet

Keep a rung if it beats the incumbent by more than 0.01 on the frozen simulated set (mean over epicentre, log depth, Mw
and origin time of 1 − error / rung-0 error) AND the 40 dev quakes (first 2 and all stations) get no more than 5% worse.
When the physics changed, every rung was re-scored (val1 → val2 → val3) and every decision re-taken; none flipped.

| rung | change | sim (val3) | dev epicentre (mean, km) | exam epicentre 2 / all (median km) | exam depth all | exam Mw all | exam t0 all | decision |
|---|---|---|---|---|---|---|---|---|
| r0 | v1.2 per station, combined classically | 0.000 | 533 | 521 / 240 | 29.1 km | 0.089 | 5.8 s | baseline |
| r1 | graph network on v1.2 readings + geometry | 0.025 | 487 | 512 / 205 | 27.8 | 0.111 | 5.9 | kept |
| r2 | + P pick times | 0.070 | 375 | 452 / 114 | 25.3 | 0.104 | 5.7 | kept |
| r3 | + v1.2's 256-number record summary | 0.071 | 373 | 451 / 122 | 26.9 | 0.102 | 5.8 | rejected |
| r4 | + P-wave cross-correlation lags | 0.075 | 371 | 446 / 117 | 26.7 | 0.098 | 6.1 | rejected |
| r5 | physics v2 (distance, back-azimuth, picks from dev twins) | 0.108 | 331 | 417 / 121 | 28.4 | 0.110 | 5.9 | kept |
| r6 | physics v3 (shallow depth and Mw from dev twins) | 0.219 | 322 | 427 / 127 | 17.6 | 0.097 | 4.3 | **released** |

Exam columns were computed after the fact and chose nothing. r2 reads the 400 all-station quakes better (114 km) than
the released r6 (127 km); the ratchet could not know that, and the exam does not change it.

## Exam

QuakeNet v1.2's exam earthquakes (2012-2025, M ≥ 5.5), graded against USGS. "2 stations": every exam quake with 2+
records, using v1.2's original station pairs (1,456). "3 / 5 / all": a 400-quake subset (seed 7, depth strata) with
extra stations fetched, using the first stations in a fixed order (up to 10).

| stations | method | epicentre median | mean | depth median | shallow depth | deep depth | Mw mean |error| | origin time median |
|---|---|---|---|---|---|---|---|---|
| 2 | **QuakeNet v2** | **427 km** | 918 | **23.5 km** | **12.8** | 59.8 | 0.114 | **13.5 s** |
| 2 | v1.2 combined | 521 | 1,129 | 34.4 | 32.1 | **42.0** | **0.109** | 15.0 |
| 2 | travel-time locator | 4,139 | 5,271 | 190 | 216 | 189 | – | 56.5 |
| 2 | station centroid | 6,190 | 5,986 | 67 | 66 | 90 | 0.309 | 189 |
| 3 | **QuakeNet v2** | **279** | 587 | **21.5** | **12.1** | 58.2 | 0.102 | **9.8** |
| 3 | v1.2 combined | 410 | 817 | 31.6 | 30.9 | **33.7** | **0.095** | 11.9 |
| 3 | travel-time locator | 669 | 1,739 | 196 | 240 | 183 | – | 31.0 |
| 5 | **QuakeNet v2** | **188** | 322 | **17.3** | **11.4** | 49.4 | 0.097 | **7.0** |
| 5 | v1.2 combined | 320 | 542 | 30.3 | 29.7 | **32.4** | **0.091** | 7.8 |
| 5 | travel-time locator | 276 | 379 | 190 | 196 | 172 | – | 21.3 |
| all (≤10) | **QuakeNet v2** | **127** | **179** | **17.6** | **11.7** | 38.3 | 0.097 | **4.3** |
| all | v1.2 combined | 240 | 371 | 29.1 | 29.3 | **27.7** | **0.089** | 5.8 |
| all | travel-time locator | 173 | 194 | 153 | 178 | 141 | – | 14.4 |
| all | station centroid | 4,223 | 4,193 | 67 | 67 | 119 | 0.315 | 72.3 |

Dumb baseline: station centroid, dev-median depth 77 km and Mw 5.85. The travel-time locator uses STA/LTA P and AIC S
picks with an ak135 grid search; it gives no Mw.

**Coverage of the 68% bars (after the dev widening):**

| stations | epicentre ellipse | depth | Mw | origin time |
|---|---|---|---|---|
| 2 | 65% | 60% | 70% | 65% |
| 3 | 64% | 61% | 68% | 66% |
| 5 | 68% | 61% | 68% | 69% |
| all | 62% | 60% | 64% | 74% |

v1.2 combined, also widened on dev: epicentre 65-74%, depth 43-61%, Mw 41-70%, origin time 79-90%.

## Supplementary exam: independent picks (27 Nov 2026)

The exam above gave every station the catalogue's predicted P time as its pick guess (audit). This dated supplementary
exam ran the released model once (27 Nov 2026, 19:18 UTC) on the same 400 multi-station quakes, from continuous LH
records fetched again from EarthScope (`training/fetch_cont.py`). Each record runs from 10 minutes before the UTC hour
of the origin to 2 hours after, so where P lies is not given. 3,986 records (same station order, first 10 usable).
Two pickers, both fixed on the 40 dev quakes before the exam (`training/indep_exam.py`):

- **per record:** the strongest STA/LTA trigger in the first 75 minutes after the hour;
- **associated:** a grid search over 12,000 epicentres and origin times for the source on which most stations have one
  of their 5 strongest triggers within ±6 s of the ak135 P time; each station's guess is that trigger.

| stations | P guess | epicentre median | mean | depth median | Mw mean \|error\| | origin time median | ellipse coverage |
|---|---|---|---|---|---|---|---|
| all (≤10) | catalogue (control, same records) | 123 km | 179 | 17.6 km | 0.096 | 4.5 s | 63% |
| all | associated | 497 | 1,346 | 26.6 | 0.122 | 40.4 | 26% |
| all | per record | 1,284 | 1,603 | 47.0 | 0.159 | 475 | 4% |
| 5 | catalogue | 191 | 324 | 17.0 | 0.098 | 7.2 | 67% |
| 5 | associated | 548 | 1,597 | 28.8 | 0.131 | 35.8 | 33% |
| 3 | catalogue | 279 | 583 | 21.2 | 0.102 | 9.7 | 64% |
| 3 | associated | 694 | 1,927 | 29.6 | 0.131 | 41.4 | 35% |

- The control reproduces the published exam (123 vs 127 km with all stations) from freshly fetched records.
- Picks against the catalogue P (grading only): associated 40% within 10 s, 51% within 30 s, 30% off by more than
  5 minutes; per record 28% within 10 s. At 1 sample/s and 0.05–0.3 Hz the P of an M 5.5–6 quake is often not the
  strongest trigger in two hours of record.
- So the headline location skill depends on knowing roughly when P arrives. Without that, QuakeNet v2 misses by about
  500 km (median) with all stations. The network was not changed.

## Where it fails

- **Magnitude:** it loses to the plain inverse-variance average of v1.2's Mw at every station count (0.097 vs 0.089 with
  all stations). The network's Mw correction helps in simulation (+7% skill) and hurts on real records.
- **Deep quakes** are read too shallow (all stations: 38 km median error vs 28 km). The dev twins showed the
  over-correction (deep depth ratio 0.74) before the exam.
- **Two stations** remain weak: median miss 427 km, mean 918 km. The P-time information needs more stations.
- **Independent picking:** without the catalogue's P guess the median miss is 497 km with all stations, not 127 km
  (supplementary exam above).
- **Inputs inherited from v1.2:** LH only, 25-95°, a ±30 s P search around a guess (from the catalogue in the exam), and
  record start times rounded to the second when the archive was cut (±0.5 s on every pick).

## Data and credits

USGS ComCat (public domain); EarthScope / IRIS DMC waveforms of the IU (doi:10.7914/SN/IU) and II (doi:10.7914/SN/II)
networks; EarthScope Syngine Green's functions (ak135f); ObsPy TauP (ak135). See `DATA_LICENSES.md`. How it sits beside
QuakeNet v1.2: `BESIDE_V1.2.md`.
