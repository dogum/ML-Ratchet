# LawFinder notebook

## 2026-11-22 (B batch, v1.0)

- Folder `/home/claude/bb/lawfinder` on a shared 2-core machine with three other B-batch builds (load 5-7 most of the
  day). One thread, `nice`, every job resumable (`STATUS.md` kept current).
- **Engine.** PySR would need Julia (~1 GB with packages) and a long compile on the shared cores; gplearn has no
  constant fitting. Wrote `lawsearch.py` (numpy/scipy): island populations, regularised evolution, Levenberg-Marquardt
  constants, hall of fame per complexity. Test: planted Budyko-Fu with 1 % noise reached the noise floor in 90 s.
- **Water** (WaterSim): banks w1 (as shipped) and w2 (+ FlowCast's dev-fitted ERA5 water-yield multiplier),
  8,000 + 3,000 basins each, ~0.1-0.2 s per basin-decade. The forcing-error model on the inputs makes the simulated
  runoff ratio noisy (Fu reaches only R² 0.46 on the bank). w1 released c18 on val_w1; after w2 every rung was
  re-scored on val_w2 and w2-c13 released. Twin test: law vs ungauged twins 0.091 MAE, vs real 0.130.
- **Fire** (FireSim): 600 + 240 fires per version, 256² cells at 90 m, 3-6 days; ~2-3 s per fire. Simulated fires
  grow on 98 % of days (real dev 68 %). Every law where growth keeps rising with perimeter failed the dev gate against
  the constant; f2-c3 (saturating) passed. Dev residuals have a spike at the zero-growth days (32 %), so the widening
  factor is large (×3.75).
- **Grid**: GridSim's raw GB `generate_states` gives inertia ≈ 4.5 s × demand, far below NESO (~7 s): the inertia
  law has the right shape and the wrong level. Nadir via GridPulse's `gpsim` (copied unchanged; `_paths.py` patched to
  import the vendored GridSim). g2 (GridPulse's release physics, Dynamic Containment in 60 % of states) makes every g1
  rung's R² negative on val_g2; g2 rungs failed the dev gate (2019 events, before DC). Released g1-c12.
- **Homes** (BuildingSim): 1,000 + 300 single-family homes per version, ~1.7 s each under load. b1 homes 1.4× too
  energy-intensive vs RECS dev; b2 (physics 1.4) fixed the median; b2-c13 released.
- **Exams** run once each (logs/exam_*.started), after a `--dry-dev` run of the same script on dev units.
- Not done: RoadSim trip-energy law; water recession law.

## 2026-11-22 — imported into ML Ratchet

- Copied from `for_repo/models/lawfinder/` of `lawfinder-v1.0.zip` (22 Nov 2026) with `tests/golden/lawfinder.txt`. Every `.npz` loads with
  `allow_pickle=False` and holds no object arrays; inference imports only numpy and scipy.
- `demo.py` run from this folder reproduces the shipped golden output line for line (timing aside), so
  `tests/golden/lawfinder.txt` is the zip's own file, unchanged.
- No training script was re-run, and nothing was fetched.
- `order: 33`; `accent` changed from `#1d6fb8` to `#6741d9` (the zip's value was a placeholder, almost AirfoilNet's
  `#1d4fb8`); `exam.short` changed from "Five platforms' real exams" to "Four platforms' real exams, five laws" (the
  RoadSim law was not built).
- Two findings recorded as platform issues, not fixed: FireSim's implied next-day growth is a median 36× the real one
  (`shared/fire/STATUS.md`, backlog `firesim-growth`), and GridSim's GB fleets commit about 4.5 s of inertia per MW
  against NESO's ~7 s, about 35 % low (`shared/grid/STATUS.md`, backlog `gridsim-gb-inertia`). The BuildingSim 1.0
  finding (homes 1.4× RECS dev energy per m²) was already fixed in physics 1.4.
- Backlog: `lawfinder` replaced by `lawfinder-road` (the unbuilt fifth law). Model families: `symbolic` is now a trait
  in use.
- Training scripts ported to `training/_paths.py` (the shipped ones wrote next to the model folder and read sandbox
  paths). `ROOT` is now the work folder (`models/lawfinder/work/` or `$ML_RATCHET_WORK/lawfinder`), seeded on first
  use with the shipped `training/results/` and page inputs, so a re-run never overwrites a shipped record and the
  exam scripts still refuse to run twice. Platform code comes from `shared/`; data roots from `$ML_RATCHET_WATER`,
  `_FIRE`, `_GRID`, `_BUILDING`, `_VEHICLE`. The network rows' inputs come from FlowCast's, FireCast's and GridPulse's
  work folders. GridPulse's `gpsim` is imported through `_paths.import_gpsim()` (both modules are named `_paths`).
  The shell wrappers `cd` to `training/` and source a portable `env_sandbox.sh`. Only `fit_law.py`-level imports were
  smoke-tested; no shipped bank or search was re-run.

## 2026-11-22 — the fifth law: RoadSim trip energy per km (built in the repository)

Asked for by the owner after the import ("Follow the existing pipelines exactly"). Details and the exam table are in
`MODEL_CARD.md` ("The road law"); the steps, as run:

- **Data.** Nothing downloaded. RoadData's processed VED trips, EPA tables and cycles were read from an existing data root
  (`ML_RATCHET_VEHICLE` pointed at another checkout's git-ignored `shared/vehicle/data`). EnergyNet's exam frames
  (`exam2_ved_v2.parquet`, `exam2_epa_v2.parquet`) and its corrected EPA labels (`epa_fixed.parquet`) came from that
  checkout's `models/energynet/work/` (`ML_RATCHET_PEER_REPO`, a new switch in `training/_paths.py`). RoadData's splits
  were not changed.
- **Bank** (`road_bank.py`, `run_road_banks.sh`): 2 physics versions × (6,000 train + 2,000 frozen) trips, 6 workers,
  about 4 minutes. 0 trips failed. Simulated energy per km sits above the dev cars' (petrol 3.4-3.5 against 2.6-2.7
  MJ/km; electric 0.6-0.75 against 0.44-0.47): the generator's cars are heavier (median 2.2 t, electric 2.65 t, against 1.3-1.5 t for the VED dev cars
  and 1.9 t for the EPA dev tests).
- **Search + ratchet** (`fit_law.py road`, 600 s per version): released p1-c23, R² 0.767 on val_p2, dev MdAPE 16.8 %.
  No p2 rung cleared the 0.01 margin. First run on a dev set missing the 35 trips of mass-less VED cars released
  p1-c18; the missing masses were set to class defaults (EnergyNet's rule) and the ratchet re-scored before the exam.
- **Dry run** on dev units (`exam_road.py --dry-dev`), then **the exam, once** (logs/exam_road.started). The law loses
  every stratum: VED trips 18.6 % (MOVES-like 11.8 %, road-load 18.3 %), EPA tests 17.3 % (EnergyNet 7.5 %, road-load
  23.8 %). Electric cars are its worst case (±42 % bias).
- **What failed.** Exam run 1 mis-scaled EnergyNet's VED rows (its frame is already in MJ; the script multiplied by the
  unit factor again). A corrected re-score was not run: the exam runs once, and the re-run was stopped for the owner to
  decide. `exam_road_errata.py` reports EnergyNet's published values on the same trips instead and marks its pooled VED
  row not scored. The law's own numbers do not depend on the bug.
- **Release.** `add_road_release.py`: the road entry merged into `lawfinder_laws.json`, `examples/road.csv` (fixed
  draw by terciles of v̄, seed 20261122), the road block added to the lab page (JS == numpy to 6e-16 on 1,500 points,
  `training/results/js_parity_road.json`). The other four laws were not rebuilt (their banks and point files are not
  shipped); their entries are unchanged. `lawfinder.py` gained the `road_load` textbook and the missing
  `degree_day_signature` form (`lf['bldg'].textbook()` raised KeyError before). The ratchet ladder was appended to
  `ratchet.yaml` by `build_card.py`'s own ladder code; `model.yaml` was edited by hand (as the shipped one had been).
- Windows note: the scripts need Python's UTF-8 mode (`PYTHONUTF8=1`) for the JSON and HTML they read.
- Backlog: `lawfinder-road` removed (built).

## 2026-11-22 — road exam: EnergyNet's VED rows re-computed (owner's OK)

- **What was wrong.** `exam_road.py` (exam run 1) multiplied EnergyNet's VED frame (`exam2_ved_v2.parquet`) by its
  `conv` column. That frame already stores `y` and `pred` in MJ (EnergyNet's `realfeat.py` multiplies before saving), so
  EnergyNet's VED predictions came out ×0.042 for petrol and hybrids and ×3.6 for electric cars. Only those comparison
  rows were wrong; the law's predictions, every rung and every other row never touched the factor.
- **What was re-computed.** EnergyNet's VED rows only, in `exam_road_errata.py`: its stored per-trip predictions / trip
  km (MJ/km), against run 1's stored truth for the same 22,877 trips (`exam_road_points.parquet`). EnergyNet's own
  truth agrees with it to 6e-8 (max relative difference); run 1's EnergyNet column is exactly 0.042× the corrected one
  at the median. The law's exam was not re-run, and `exam_road.json` is left as run 1 wrote it.
- **Before → after (MdAPE, bias):** petrol 95.8 % (−95.8 %) → 8.6 % (+0.3 %); hybrids 95.6 % (−95.6 %) → 13.4 %
  (+3.8 %); electric 187.0 % (+187.0 %) → 23.2 % (−20.3 %); all VED trips 95.8 % → 10.1 % (+1.1 %). The three
  per-powertrain values equal EnergyNet's published exam values. The interim report (published values, pooled row "not
  scored") is replaced.
- **What changed downstream.** The pooled VED headline now shows EnergyNet at 10.1 %; the page verdict reads "loses
  to EnergyNet" (was "loses to MOVES-like", 11.8 %). Every road verdict is still a loss. `exam_road_reported.json`,
  `lawfinder_laws.json`, the lab page, `model.yaml` (rows, limits, help_wanted, next) and `MODEL_CARD.md` updated.

## 2026-11-22 — rebuild sweep: every training path checked

No change needed: `_paths.py` creates `banks/`, `results/`, `logs/`, `web/`, `release/` and `tmp/` and seeds the shipped
results and page inputs; the bank scripts and `build_release.py` create their own subfolders. Inputs from outside:
each platform's data root (`$ML_RATCHET_WATER`, `_FIRE`, `_GRID`, `_BUILDING`, `_VEHICLE`) and the exam outputs of
FlowCast, FireCast, GridPulse and EnergyNet in their work folders (`$ML_RATCHET_PEER_REPO`). Not reproducible from this
folder: the other four laws' banks and exam point files (not shipped), which is why `add_road_release.py` exists.

## 2026-11-24 — a second road law from the Vehicle Bake-off (recorded, not adopted)

- **What arrived.** The Vehicle Bake-off (cloud, 23-24 Nov 2026; `experiments/bakeoff-vehicle`) built its own road law as
  its formula arm, on its own RoadSim physics v2 bank (EnergyNet v2.1's recipe) with a log-target search, and packaged it
  for LawFinder (`experiments/bakeoff-vehicle/arms_phys/lawfinder_road/`, MERGE_NOTES.md). The handoff called it ready
  to merge as the fifth law.
- **Why it is not merged as the fifth law.** LawFinder already released a road law here on 22 Nov 2026 (p1-c23), and its
  exam ran once. The bake-off law was graded on the bake-off's exam, and it scores better: on the same 14,469 VED petrol
  exam trips, MdAPE 14.0 % (s1-c21) against 19.3 % for p1-c23. Swapping the release after seeing both exam scores would
  let the exam choose the law. So p1-c23 stays released, and s1-c21 stays where it was built, as the bake-off's formula arm.
- **For the record (exam, VED petrol trips, n 14,469):** bake-off s1-c21 14.0 % (bias −12.5 %); LawFinder p1-c23 19.3 %
  (−18.9 %); RoadSim's own law p2-c10 14.0 %; road-load E_phys 33.0 %; MOVES-like 10.3 %; EnergyNet v2.1 8.6 %. The
  bake-off law misses dynamometer cycles (EPA conv HWY 30.5 %).
- **What it suggests for a next rung.** The two searches differ in the bank (EnergyNet's recipe, heavier coverage of
  real routes) and in the loss (squared log error). Both are changes LawFinder can try as its own rung on its own frozen
  val set and dev gate, with no exam involved. LawFinder's own rule, applied to the bake-off's fronts, would have kept the
  simpler s1-c16 (`road_ratchet.json`).

## 2026-11-27 — audit resolution: EV grade sign documented, input-range warnings

The two-reviewer audit (`docs/audits/lawfinder-v1.0.md`) found that the road law's battery energy falls as grade rises,
for every battery-electric car (0.408 → 0.324 MJ/km from flat to +2 %, where RoadSim gives 0.332 → 0.659). The law is
not refitted here: that needs a new search and a new exam. The card now says so, and `next` queues a search that rejects
laws whose energy falls with net grade. `Law.predict` now warns when an input is outside the range its bank covered
(`strict=True` raises); the homes law's heating slope turns negative past 2020 and the grid inertia law can fall below the
nadir law's floor. One of the demo's real grid events (a loss above 2 GW) is outside the nadir law's range. `LawFinder()`
also read its JSON with the platform's default encoding; it now reads UTF-8.

## 2026-11-27 — road: sign-checked search (the queued rung): rejected; p1-c23 stays

Run on the owner's approval (`training/road_sign.py`, records `training/results/road_sign.json` and
`road_search_p2s.json`).

- **Banks rebuilt from their seeds.** This machine had no road banks, so `road_bank.py` rebuilt p2 val (2,000 trips,
  425 BEV) and train (5,994 usable). Every logged rung re-scored on the rebuilt val_p2 gives exactly its logged R²
  (worst |ΔR²| 0), and the released law's dev error is unchanged (0.1680). So this is the same frozen set.
- **Sign check:** E(G + 0.25) ≥ E(G − 0.25) at every val_p2 row, with the other inputs held. The released p1-c23 fails
  it on 21 % of rows and on every BEV row. The six p2 candidates that rise with grade for BEVs (c18, c20, c21, c22,
  c24, c25) fail it too: they fall with grade on 4-14 % of the non-BEV rows. So none of the 40 logged laws both reads
  grade and passes.
- **The rung:** a new p2 search on the train bank, where a law that fails the sign check on its fitting rows gets an
  infinite loss. Same engine and seed as the logged p2 search, 1,200 s instead of 600 s (the check triples the cost of
  each evaluation), 143,084 evaluations. It found 14 laws, c1-c19. The search found no way to use grade with the right
  sign. Only p2s-c15 reads G, and it just adds G (MJ/km per %), giving −1.41 MJ/km for a BEV at −2 %. The rest ignore
  grade.
- **Decision under the unchanged rule: all 14 rejected.** The best is p2s-c19, `E = (M + 0.27·R/v)·exp(−2·K^h·(e − 1.142·K))`:
  frozen R² 0.757 against the incumbent's 0.767 (it needs more than 0.777). Its dev error is better (0.137 against
  0.168), but the rule promotes on the frozen set. The exam was not opened. p1-c23 stays released.
- **For the owner, not adopted:** if the sign check rejected a rung (a rule change), the same walk would release p1-c17
  (R² 0.754, dev 0.175), which ignores grade. Dev MdAPE on electric-car rows (the Leaf's 258 VED trips plus 29 EPA BEV
  tests in the dev draw): p1-c23 0.271, p1-c17 0.236, p2s-c19 0.221, and the best of the six grade-rising p2 laws
  (p2-c20) 0.209. Choosing among these on EV dev is a different rule, and the owner's call. What a correct grade law
  needs, judging by these fronts, is a grade term that switches with the powertrain: energy rises with grade for every
  car, and BEVs recover part of the descent. The search's one-line, pooled form has not found one in 1,800 s.

## 2026-11-27 — road law v1.1, part 1: the sign gate adopted as a release rule; p1-c17 chosen (before the exam)

**Rule change (owner decision, 27 Nov 2026).** From now on a road-law candidate may not be kept, and so not released,
if its energy falls as net grade rises at any val_p2 row (E(G + 0.25) ≥ E(G − 0.25) at every row, other inputs held;
`road_sign.py`'s check). The reason is physics: tank and battery energy rise with uphill grade for every powertrain.
The released v1.0 law p1-c23 failed the check on 21 % of val_p2 rows and on every battery-electric row. The audit
(25 Nov 2026) prompted the rule. No exam result did. The rest of the rule is unchanged: frozen R² beats the incumbent
by more than 0.01, dev error no more than 5 % worse.

**The walk, re-run with no new search** (`training/road_gate.py`). It covers the 40 v1.0 rungs plus the 13 sign-checked
p2s rungs, all on val_p2. The kept rungs are r0, p1-c5, c7, c10, c11, c14 and **p1-c17**. p1-c23 and p1-c25 are now
rejected by the gate, and nothing after p1-c17 beats its R² by 0.01 while passing the gate. **Released: p1-c17**,
`E = M·exp(2·(v^-0.6288 - e)·(sqrt(0.1124·R) - h))`, which does not read grade. It has frozen R² 0.754 (v1.0: 0.767)
and dev MdAPE 0.175 (v1.0: 0.168). Its error bar is σ_sim 0.804 MJ/km, widen factor 1.0, and dev coverage 83 %.
Records: `results/road_ratchet.json` (v1.1), `results/road_ratchet_v10.json` (v1.0 as released).

**What is already known about the exam.** The v1.0 exam scored every rung after the fact (`exam_road.json`
`strata_rungs`), so p1-c17's exam numbers were published with v1.0. On VED, all trips, its MdAPE was 18.67 % against
p1-c23's 18.60 %. The switch therefore does not favour the exam: p1-c17 was slightly worse on it. The exam is run once
more for v1.1 (part 2) so the release has its own dated record and bars.
