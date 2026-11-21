# FireCast v1.0: model card

Part of ML Ratchet ("Trained on physics. Graded on reality."), fire platform (Wildfire Phase 2, project 1).
Imported into the repository on 21 Nov 2026 (see `NOTEBOOK.md`); weights and every number unchanged. Paths below are
repository paths; the card's text is otherwise as shipped, with sections added at the end (inputs, verdicts, rule
notes, live instrument).

## What it does
Predicts where a wildfire will have burned by the next afternoon satellite overpass. Output: per 90 m cell, a
calibrated probability; plus a yes/no map at a dev-fitted threshold. Input: today's observed fire (FEDS-style burned
area and active fire line, VIIRS S-NPP 375 m hot spots at the issue overpass and the night before) and gridMET daily
weather for today and tomorrow. Optional LANDFIRE fuels, slope and canopy (mode L). The 12 h product was not built.

## Physics and simulator
FireSim (ML Ratchet Wildfire Phase 1; verified against BehavePlus/Rothermel tables, pyretechnics and analytic
ellipses, see shared/fire/checks/): Rothermel 1972 surface spread with Albini 1976, Scott & Burgan 40 fuel models, Anderson
1983 ellipses, Van Wagner 1977 crown initiation, Nelson 2000 dead-fuel moisture, minimum-travel-time spread, spotting,
containment lines. FireCast wraps it (src/simbank.py) without editing it:
- Each training fire burns 2-8 days on a real LANDFIRE landscape (dev fires and guard-passed non-exam windows; 30 %
  parametric) with real HRRR hourly weather, in 12 h segments.
- Randomised per fire: a ROS factor (lognormal around the dev-calibrated replay level, median 0.35, sd 0.7), per-fuel
  group ROS scatter, night slow-down (x0.15-1.0), growing containment (70 % of fires; final fraction 0.2-0.95, time
  scale 1-4 days), spotting (30 %), wind speed/direction and RH error, live fuel moisture.
- Physics version p1 adds a fire-age ROS factor fitted to the p0 dev twin test (k0 x2.6 ... k>10 x0.55).
- Observed through `firesim.observe` (S-NPP VIIRS at real overpass times, detection vs FRP, scan-angle footprints,
  clouds 30 % of fires) and a FEDS emulator fitted on dev (training/results/calib_feds.json: dev Dice 0.95, fire-line F1 0.62).
- Weather features: in simulation the HRRR series is turned into gridMET-like daily values through a dev-fitted
  operator with its residual noise (training/results/calib_wx.json), so the network sees the same kind of weather it gets on exam fires.

## Network
U-Net on 90 m patches (stem at 90 m, encoder to 1.44 km), 374k weights. 22 map channels (burned now, yesterday's
growth, fire line, two VIIRS FRP rasters, distances to burn / line / pixels, offset from the nearest fire-line cell
along and across tomorrow's wind, scoring zone, 11 landscape channels) + 22 scalars (16 weather, 6 fire) through an MLP
that adds per-channel biases at every level. Loss: BCE on unburned cells in the 10 km zone. 128-cell random crops,
dihedral augmentation, landscape dropout 0.5. CPU only, one thread.

## Ratchet
Rule: keep a rung if sim AP on the current frozen set beats the incumbent by > 0.01 AND dev AP (40 dev fires, 535
growth fire-days) >= 0.95 x incumbent. Frozen set p1: 149 simulated fires, 562 fire-days.

| Rung | Change | Sim AP (p1) | Dev AP | Dev F1 @0.5 | Decision |
|---|---|---|---|---|---|
| r0 | p0 bank (668 fires), 1,800 steps | 0.573 (p0 set: 0.599) | 0.303 | 0.158 | base |
| r1 | p1 physics (fire-age ROS), init r0, 1,200 steps, p1 bank (415 fires) | 0.582 | 0.325 | 0.176 | rejected (+0.009) |
| **r2** | init r0, 1,200 steps on p0+p1 banks (1,083 fires, 4,402 fire-days) | **0.587** | 0.316 | 0.181 | **kept, released** |
| r3 | init r2, +1,000 steps, lr 5e-4 | 0.588 | 0.327 | 0.188 | rejected (+0.001) |

Twin tests (dev): p0 prior-predictive twins (470 days x 3) gave median log10(sim/real new area) +0.16, with day 0
at -0.44 and days > 5 at +0.33..+0.35. p1 on the same 126 days (partial run): day 0 -0.82 -> +0.20, k > 10 +0.44 -> +0.01,
days 1-10 about -0.34.

## Calibration (dev only)
p = sigmoid(logit / T + b), T >= 1 (widen-only), fitted on the 535 dev fire-days: T = 1.44, b = -1.23. Threshold for
the yes/no map = the one maximising mean dev F1: 0.10 (edge of the searched grid). Dev F1 0.227 (leave-one-fire-out
0.228) vs 0.181 raw at 0.5. Mode L: T 1.57, b -1.20, thr 0.10.

## Exam (run once; 3,030 MTBS exam fires, 25,105 growth fire-days; means with 95 % fire-bootstrap CIs)

| Method | All AP | All F1 | Early AP / F1 | Main AP / F1 | Recent AP / F1 | Total-area Dice | log10 new area pred/obs |
|---|---|---|---|---|---|---|---|
| **FireCast** | **0.323** [0.318, 0.328] | **0.239** [0.234, 0.244] | 0.318 / 0.232 | 0.327 / 0.249 | 0.318 / 0.222 | 0.922 | +0.28 |
| Persistence | 0.013 | 0.000 | 0.013 / 0 | 0.013 / 0 | 0.012 / 0 | 0.964 | -2.20 |
| Climatological median growth | 0.204 | 0.004 | 0.206 / 0.003 | 0.201 / 0.006 | 0.209 / 0.003 | 0.964 | -2.14 |
| Yesterday's growth repeated | 0.191 | 0.140 | 0.192 / 0.138 | 0.189 / 0.144 | 0.192 / 0.131 | 0.924 | -0.30 |
| Yesterday's growth from the fire line (best) | 0.252 [0.247, 0.257] | 0.184 [0.180, 0.188] | 0.249 / 0.181 | 0.257 / 0.190 | 0.242 / 0.175 | 0.947 | -0.38 |

- Per fire-day win rate vs the best baseline: AP 71 %, F1 50 %. FireCast leads in all 9 regions and all 5 size bins.
- Replay subset (48 exam fires with LANDFIRE + HRRR; 627 replayable growth days; 14 skipped for crops > 500k cells,
  72 without HRRR coverage): FireCast F AP 0.351 / F1 0.265, FireCast L 0.355 / 0.271, calibrated physics replay
  (replay_calibration.json as stored) deterministic 0.150 / 0.217 and 12-member ensemble 0.165 / 0.216,
  fire-line baseline 0.273 / 0.210. The replay over-predicts new area x3.5 (log10 +0.54).
- Reliability (pooled exam cells): close to the diagonal up to p = 0.35; over-confident above (forecast 0.9 -> 61 %).
- Context only (different target/resolution): published WildfireSpreadTS AP 0.19-0.37.

## Limits
- Probabilities above ~0.35 are over-confident; widen-only calibration could not fix the shape.
- Predicts the FEDS/VIIRS view of the fire (375 m detections, 12-hourly perimeters), not ground truth.
- Persistence still wins total-area Dice; at the threshold FireCast over-predicts new area ~1.9x.
- LANDFIRE adds little (+0.002-0.004 AP) and exists only for 48 exam fires; fuels are LANDFIRE 2016-2023 releases.
- Simulator gaps: no suppression tactics beyond containment growth, no plume-dominated runs, no fuel change since the
  LANDFIRE year. Sim AP 0.59 vs real dev 0.32.
- Only three trained rungs (2 shared CPU cores); decisions within +-0.01 are noise-level.

## Data credits and licences
FEDS-MTBS (Zenodo 20187963; Chen et al. 2022 FEDS method; CC BY 4.0); MTBS perimeters (USGS/USFS, public domain);
VIIRS S-NPP active fire (NASA LANCE/FIRMS, via FEDS); gridMET (Abatzoglou 2013, public domain); LANDFIRE (USGS/USFS,
public domain); HRRR (NOAA, AWS Open Data / Univ. of Utah hrrrzarr); FireData splits and baselines, FireSim (ML Ratchet
Wildfire Phase 1).

## Inputs and scoring (from the release README)
Grid: 90 m cells, north up, rows north to south (the release used EPSG:5070 CONUS Albers windows from FireData). Use a
crop that covers the burned area plus at least 10 km on each side. One fire-day `day` is a dict:
- `now`: bool H×W, burned area at the issue overpass (afternoon, about 13:30 local).
- `prev`: bool H×W or None, burned area 24 h earlier; `has_prev` True if given.
- `fl`: bool H×W or None, FEDS active fire line at issue time (cells on the burning edge).
- `pix_pm`, `pix_am`: VIIRS detections at the issue overpass and the night overpass 12 h before, as dicts of arrays
  `row`, `col` (fractional grid coordinates of the pixel centre), `ps`, `pt` (pixel size along scan and track, km),
  `frp` (MW).
- `wx`: 16 floats = `fc.feature_vector(d0, d1, gamma_deg)`, where `d0`, `d1` are the gridMET daily values of the issue
  day and the next day with keys `vs` (m/s), `th` (deg, wind from), `rmin` (%), `tmmx` (°C), `vpd` (kPa), `pr` (mm),
  `fm100` (%), and `gamma_deg` is the grid convergence (grid north = true north + gamma).
- `first`: True on the fire's first detected day.
- `land` (optional, LANDFIRE): dict of uint8 rasters from `fc.land_from_landscape({'fuel', 'slope', 'aspect', 'cc'})`.
  The default exam mode is without it (`land=False`); the calibration in the weight file is for that mode.

```python
import firecast as fc
net = fc.FireCastNP()                          # firecast_weights.npz next to this file
prob = fc.predict(net, day, land=False)        # calibrated probability, same shape as day['now']
win = fc.score_window(day['now'])              # cells within 10 km of the burned area
fc.nextday(day['now'], day['nxt'], prob, fc.CELL_KM2, thr=net.thr, window=win)   # AP, F1, Dice, area error
```

The weight file holds 373,122 numbers (BatchNorm folded into the convolutions: 592 conv biases instead of
1,184 BatchNorm scales and shifts), plus `cal_T`, `cal_b` and `thr`; the torch model it came from has 373,714
trainable weights, the card's "374k".

## Verdicts (model.yaml)
A row is a win only if FireCast beats every baseline on it. Intervals are the shipped 95 % fire-bootstrap intervals
of each method separately; the per-fire-day rows were not shipped, so there is no paired interval on a difference.

| Row | FireCast | Best other | Verdict |
|---|---|---|---|
| All, mean AP | 0.323 [0.318, 0.328] | fl_prev 0.252 [0.247, 0.257] | win |
| All, mean F1 | 0.239 [0.234, 0.244] | fl_prev 0.184 [0.180, 0.188] | win |
| Early 2012-16, AP | 0.318 [0.310, 0.327] | fl_prev 0.249 [0.241, 0.258] | win |
| Main 2017-21, AP | 0.327 [0.320, 0.334] | fl_prev 0.257 [0.250, 0.263] | win |
| Recent 2022-24, AP | 0.318 [0.304, 0.332] | fl_prev 0.242 [0.229, 0.256] | win |
| Replay subset, AP (627 days) | L 0.355 [0.332, 0.378] (F 0.351) | fl_prev 0.273; replay ensemble 0.165 | win |
| Replay subset, F1 | L 0.271 [0.247, 0.291] (F 0.265) | replay deterministic 0.217 | win |
| All, median total-area Dice | 0.922 | persistence 0.964 | loss |
| Reliability, cells forecast 0.90-0.95 | 61 % burned | nominal 92 % | info |

## Rule notes (from the build handoff, section 6, as they concern FireCast)
- **One exam fire's landscape was opened outside the exam script:** its array shape was read to size the memory guard
  of the physics replay (crops over 500k cells are skipped).
- **The first physics-replay baseline attempt was killed** after 2 fires (1.7 GB) and re-run from scratch; the replay is
  deterministic. The killed attempt is kept.
- **The exam process peaked at about 1.35 GB** on large fires, over the sandbox's comfortable limit, and was allowed to
  finish rather than abort a once-only run.
- **Dev data in the gate:** dev fires fed the physics calibration (FEDS emulator, weather operator, the p1 age factor
  through the twin test) and also gate every rung and fit the calibration, so the gate is partly in-sample. The exam
  was untouched.
- **Interruption:** the build was interrupted once (an interface stall) and resumed from its notes after checking that
  no exam had started.

## Live instrument and lab
The site instrument (`site/src/instruments/firecast/`) runs the released network (float16 weights) in the browser on
the three exam fire-days of `examples/`: it rebuilds the 22 input maps from the raw inputs (an exact Euclidean
distance transform for the distances to the burn and the hot spots; the nearest-fire-line offsets come precomputed
from scipy) and runs the U-Net on the map window (up to 240 × 240 cells), with tomorrow's wind speed and direction,
minimum humidity and 100-h fuel moisture as sliders, and the VIIRS hot spots and yesterday's perimeter as switches.
It matches `firecast.py` to 3.9e-6 on 35 reference cases. The offline lab (`lab/index.html`) is the page published
with the release: twelve seeded exam fire-days, a weather lab, the exam, the ratchet and the twin test.
