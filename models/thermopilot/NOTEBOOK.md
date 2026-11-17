# ThermoPilot status (BuildingSim phase 2)

A learned, causal thermostat / heat-pump controller for homes and small commercial buildings. Every hour it picks a heating and a cooling setpoint (36-pair menu: off, band edge, pre-heat / pre-cool up to 3 K) to minimise bill + λ × kelvin-hours outside the occupant's comfort band. Trained only in BuildingSim by imitating a perfect-foresight DP teacher; graded on simulated twins of real exam units under real weather and real tariffs.

- Page: `lab/index.html`, the original page (week explorer over BuildingSim runs, live JS network lab, exam, WPuQ reality check, ratchet).
- Release: `thermopilot-v1.0.zip` (numpy `thermopilot_core.py`, weights, `demo.py` replaying 3 exam episodes in ~1 s, `pipeline.py`, MODEL_CARD, training code).
- Code: the sandbox's `bldg/thermopilot/` (tp/, scripts/, STATUS.md, ratchet.json); in this repository `training/`.

## Recipe as built
- **Physics:** BuildingSim 5R1C + equipment curves at 1 h through the step API. The teacher and the classical MPC use BuildingSim's own one-step zone maps. Numba DP == numpy DP exactly. JS port == Python on 2,232 / 2,232 demo hours. numpy == torch to 0.
- **Teacher:** a DP over the mass temperature (0.25 K grid, 24 h horizon, hourly replanning) with the true weather, the true house and the true state. On demand tariffs it does a two-pass monthly peak target using exact re-simulation. "Perfect foresight" = best of that DP and the rules.
- **Training data (simulation only):**
  - Buildings: generated homes and small commercial buildings, plus RECS *prior* twins (nearest ResStock match). The "true" house is a perturbed copy of the controller's nominal one.
  - Weather: 2016 BDG2 site weather and 17 TMYx cities.
  - Tariffs: 18 URDB tariffs, synthetic TOU/demand tariffs and ERCOT 2018 RTP.
  - Forecast error grows with lead time.
- **Student:** MLP 298→384→384→36 (277k params). Loss = soft-target CE + expected regret. BC, then 3 DAgger rounds (533 episode-months, 391k hours), plus a 6 h comfort shield.
- **Dev gate:** 30 episodes (11 RECS dev respondents + 4 BDG2 dev buildings × Jan/Jul 2017). Frozen sim val: 60 episodes.

## Rungs (skill: 0 = fixed thermostat, 1 = perfect foresight; MPC with forecast = 0.83 val / 0.83 dev)
| rung | change | val | dev | decision |
|---|---|---|---|---|
| R1 | behaviour cloning, 173 episodes | 0.08 | −3.02 | rejected (dev comfort) |
| R2 | + 4 h shield | 0.38 | −0.81 | rejected |
| R3 | + 6 h shield | 0.42 | −0.19 | rejected |
| R4 | + DAgger 1 | 0.52 | 0.04 | kept |
| R5 | + DAgger 2 | 0.50 | −0.07 | rejected |
| R6 | 384×384 | 0.55 | 0.20 | kept |
| R7 | + DAgger 3 | 0.56 | 0.36 | **kept = v1.0** |

## Exam (run once, R7)
Bill saving vs fixed thermostat; K h = kelvin-hours outside the band per episode; 90 % bootstrap over units.

| set | ThermoPilot | setback (own schedule) | MPC with forecast | perfect foresight | TP skill |
|---|---|---|---|---|---|
| RECS exam twins, 96 home-months, 12 tariffs | 2.2 % [1.6, 2.8], 1.4 K h | 2.2 %, 2.5 K h | 3.5 %, 0.9 K h | 3.6 % | 0.33 |
| BDG2 small commercial, 32 | 2.8 % [1.4, 4.2], 0.05 K h | 1.2 % | 3.1 % | 7.2 % | 0.45 |
| HUE BC homes, 21 | 4.3 %, 0.4 K h | 5.9 %, 3.1 K h | 5.7 % | 5.8 % | 0.54 |
| WPuQ heat-pump homes, 38 | 3.1 % | 5.6 % | 7.2 % | 7.5 % | 0.35 |

- ThermoPilot always beats fixed. Its comfort is better than setback.
- On homes it is roughly equal to setback or worse.
- It is close to MPC only on commercial demand tariffs. The classical MPC beats it everywhere.
- Standard setback (−4/+4 K) saves 8–11 % but leaves the band for ~700–870 K h per month.
- Setback raised the bill on 2 of 12 RECS tariffs here (phase 1's 1R1C house: 6 of 11).

**WPuQ reality check (report only):**
- Generic twins overstate the heating slope 2.7× (median). The twins also put heat-pump load at night, while the real homes run flat (hourly-profile correlation r = 0.21).
- On generic vs envelope-scaled twins, the controller order holds with Kendall tau 0.70. MPC > ThermoPilot > fixed holds.
- ThermoPilot vs setback flips in 18 of 37 homes, so that comparison cannot be trusted from the simulator.

## Limits
- The exam is simulated: there are no real thermostat logs (ecobee gated).
- Time-shifting savings are the least certain part.
- The model uses an hourly ideal thermostat: no short-cycling, no humidity control.
- Twins are survey characteristics plus a ResStock match. HUE and WPuQ use stand-in US tariffs.
- Heat pumps are under-represented in training.
- The student trails the MPC by 0.1–0.5 skill.

## Next ideas
1. More DAgger rounds, plus a heat-pump-heavy bank (val still rose with every round).
2. Make the student a residual on a cheap MPC, or distil the classical MPC's value function.
3. Calibrate residential twins on RECS dev annual energy, then build a new frozen val.
4. Get real thermostat logs (ecobee DYD application) for a real exam.

## Sandbox gotchas
- The shared 2-core machine ran at load 8–10 for the first 2 h, so episodes took 4–5× longer.
- Recovery was resumable banks and baseline files, skipping work already done.
- An oracle that only amortised demand charges was worse than a fixed thermostat on commercial demand tariffs. The fix was a two-pass peak-target oracle.
- Ratio-of-sums skill was dominated by large commercial bills; it is now relative per episode.

## 17 Nov 2026: imported into ML Ratchet

What changed in the import (code and weights as released; only paths and file places moved):

- `thermopilot_core.py`, `thermopilot_v1.npz`, `demo.py`, `pipeline.py` and `MODEL_CARD.md` sit at the model root as
  in the zip. `demo_episodes.npz` moved to `examples/bdg2/` because it is BDG2-derived (2017 site weather of Cockatoo,
  Hog and Gator; the third episode is a twin of the BDG2 building Gator_public_Geraldine); `demo.py` reads it from
  there. `examples/bdg2/LICENSE-DATA.md` and `lab/LICENSE-DATA.md` carry the CC BY-SA 4.0 notice.
- `training/_paths.py` is new: `SRC` (training/), `WORK` (`models/thermopilot/work/` or `$ML_RATCHET_WORK/thermopilot`),
  `BLDG` (`shared/building/` or `$ML_RATCHET_BUILDING`, holding `buildingsim` and `buildingdata`) and `PROC`
  (`BLDG/data/proc`). `tp/__init__.py` and `tp/env.py` import it in place of the sandbox's hard-coded package roots;
  the HUE and WPuQ parquet reads in `tp/env.py`, `tp/cases.py` and `scripts/wpuq_check.py` use `PROC`.
- In the sandbox the scripts' `ROOT` was the `thermopilot/` folder (tp/, scripts/, data/, results/, ratchet.json).
  In the repository `tp/` and `scripts/` are in `training/`, and the scripts' `ROOT` is now `WORK`: `data/cases`,
  `data/raw/nasa_yvr`, `results/<set>/`, `ratchet.json` and `exam_*.json` from a re-run land there. The shipped
  `training/ratchet.json`, `training/results/exam_r7.json` and `training/results/wpuq/` are the v1.0 logs, kept as
  shipped. `build_page.py` reads those logs from `training/` (as shipped it pointed one folder too high for the
  release layout) and writes the page to `WORK`. `scripts/build_release.sh` assembles `WORK/release/` and zips it to
  `WORK/thermopilot-v1.0.zip` instead of the sandbox's output folder.
- `lab/index.html` is the original page, copied byte for byte (5.4 MB). Its "Student" paragraph still describes the
  R1-era 298→256→256→36 network (152k parameters); the released network is 298→384→384→36 (276,516 parameters).
- `web/export_instrument.py` (new) writes the site's live instrument from the lab page's data: the three RECS exam
  twins in five 2017 climates, two weeks and five tariffs, with BuildingSim's hourly house maps and the full
  BuildingSim runs as the truth. The browser runs the network closed-loop on the linear house; on the 150 lab weeks
  ThermoPilot's bill there is a median 0.953× the full-simulator bill (IQR 0.896–1.003). Parity with Python: 20
  cases, worst relative difference 8.5e-15.

Re-run here: `python demo.py` (numpy only, about 1 s) replays the three exam episodes and chooses the recorded
setpoints in 744 / 744 hours each; its bills, comfort and J for every controller are the closed-loop BuildingSim
results stored in the file. Recorded as `tests/golden/thermopilot.txt`. Nothing in `training/` was re-run: it needs
BuildingSim, BuildingData and their processed data (`shared/building/`), which arrive with the platform zip
(buildingsim-buildingdata-v1.0), plus numba (DP) and torch (training).

Rule notes from the build (no exam result was used to choose anything): the RECS exam uses 4 of the 6 respondents
drawn per tariff slot, a cut made for compute and decided before the exam was run. During the build the agent once
ran `pkill -f` with its own script name to stop a job; the pattern could have matched other processes, but it matched
only its own.

Waits on the platform zip and numba: `pipeline.py`, every training script, rebuilding the lab page and the demo
file, and re-running the WPuQ check. `shared/building/` arrived in the worktree during the import; with it,
`pipeline.py` finds BuildingSim and BuildingData through `_paths.py` and stops at `import numba` (`tp/dp_fast.py`),
which is in neither of the repository's virtual environments. Not installed here.
