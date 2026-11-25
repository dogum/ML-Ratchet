# LawFinder law 5: RoadSim trip energy (from the Vehicle Bake-off, agent C, 23 Nov 2026)

The law LawFinder skipped ("road: not started"). Built as the bake-off's `formula` arm, on the bake-off's frozen RoadSim
bank (physics roadsim-physics-v2, EnergyNet v2.1's recipe) and dev gate. It is graded ONCE in the bake-off exam
(`harness/exam.py`); LawFinder converts that result and does not run a separate exam.

## The law (conventional vehicles; e in MJ/km)

    e = 3.406 p^0.835 + 0.385 x 1.184^(exp c) x m t exp(-0.058 m)        (s1-c21, 21 nodes)

p = positive wheel work per km (MJ/km, at the known road load), t = min/km, m = mass (t), c = start state (-1 warm,
0 unknown, +1 cold). Textbook rival: road-load, e = a p + k (m / 1.5)^0.7 t (EnergyNet's E_phys): EnergyNet's simulation
constants a = 3.515, k = 0.279; fitted on the VED ICE dev trips a = 3.616, k = 0.793. Companion laws (hybrid,
battery-electric) are in `road_law_entry.json` -> "companions".

Numbers (bake-off harness): conventional sim val R^2 of log e 0.912 (road-load 0.757; a constant -0.15); VED ICE dev MdAPE
0.098 (road-load 0.299, EnergyNet v2.1 0.066, MOVES-like dev-fitted 0.065). 68 % bar: x exp(+-0.194) (log half-width on
the frozen set; dev coverage 0.80, so no widening). Whole formula arm (all powertrains): sim val skill 0.323, dev D16
0.1752 (E_phys 0.2278, EnergyNet 0.1133).

Two rules, two answers, both recorded in `road_ratchet.json`: LawFinder's rule (R^2 of log e + 0.01, dev MdAPE within
5 %) walking the conventional fronts keeps **s1-c16**, e = (4.334 p)^0.8335 + m t exp(-(14.79 + m) / 18.36)
(sim R^2 0.906, VED ICE dev 0.083: c21's +0.006 R^2 is under LawFinder's margin). The bake-off's rule (sim val skill over all powertrains + D16) released the c21 law, and only
that law is graded on the exam, so the entry uses c21 and history[-1] marks it kept with the bake-off's reason.

## Files here

| File | Goes to (LawFinder) | What |
|---|---|---|
| `law_road.py` | `src/law_road.py` | law config (VARS, VAR_INFO, TARGET, load_bank / load_dev from the bake-off harness, road-load textbook + fit) |
| `lawvars.py` | `src/lawvars.py` | the trip summaries from EnergyNet's 151 inputs or a 1 Hz trace (law_road imports it) |
| `lawsearch_logtarget.patch` | `src/lawsearch.py` | `patch -p1`: adds `Search(log_target=True)` (squared log error); default behaviour unchanged |
| `road_ratchet.json` | `results/road_ratchet.json` | every conventional front member (searches s1, s2) with sim / dev scores, decisions, bars, history |
| `road_law_entry.json` | merge into `release/lawfinder_laws.json["laws"]` | the `road` entry (+ companions, bake-off numbers); `exam` filled by exam_road.py |
| `road_points.json` | for `src/webdata.py` | 1,400 simulated conventional val trips + 645 VED ICE dev trips (vars + e) |
| `exam_road.py` | `src/exam_road.py` | run ONCE after the bake-off exam: bake-off scores -> `results/exam_road.json` + `exam_road_points.parquet` |

The search fronts themselves are in the bake-off: `arms_phys/work/search/s{1,2}_ice.json` (copy to
`results/road_search_s1.json` etc. if you want them in `training/results/`).

## Code changes in LawFinder (small)

1. `release/lawfinder.py`: the road law's bar is multiplicative. In `Law.__init__` read `self.kind = d.get('sigma_kind',
   'abs')`; in `predict` return `(y, y * (np.exp(sigma) - 1))` when kind == 'log' (or document that sigma is a log
   half-width). Add the textbook form:
   `def _road_load(c, p, m, t, **_): return c[0] * np.asarray(p, float) + c[1] * (np.asarray(m, float) / 1.5) ** 0.7 * np.asarray(t, float)`
   and `_TEXTBOOK['road_load'] = _road_load`.
2. `src/webdata.py`: `LAWS += ['road']`; `QUESTION['road'] = 'How much fuel energy does a car, van, bus or truck use per km
   on a real trip?'`; `PLATFORM['road'] = 'RoadSim'`; `XDEF['road'] = 't'`; `XLOG['road'] = ['m']`. `law_block` works with
   law_road.py as is (it loads the bake-off bank through the harness; set BAKEOFF_HARNESS if the path differs), or
   read `road_points.json` instead of re-sampling.
3. `src/build_page.py` META['road']: exam_key 'mdape', exam_cols [['mdape', 'MdAPE', '%'], ['bias_median_pct', 'bias', '%'],
   ['rho', 'ρ'], ['cov90', 'cov90']], exam_order ['law', 'law_conformal', 'textbook', 'network', 'moves_like', 'label',
   'dumb'], exam_desc 'VED ICE trips (Ann Arbor, 2017-18), exam cars', exam_unit 'trips', textbook_rows ('textbook',),
   network 'network' (EnergyNet v2.1), target_ml '<mi>e</mi>', and a `textbook()` branch with
   js `(TB.a * p + TB.k * Math.pow(m / 1.5, 0.7) * t)`. The exam's other conventional sets (EPA tests, Mersin buses,
   ICCT) are in `exam_road.json["by_set"]`.
4. `src/build_release.py`: `PLATFORM['road'] = 'RoadSim'`; `TB['road'] = ('road_load', 'coef_dev', 'textbook')`; the
   exam points file is `exam_road_points.parquet` (columns t, T, d, e and the rows' predictions; p, m, n are not
   kept by the bake-off's scores, so the examples table shows t, T, d only).
5. `model.yaml` / `ratchet.yaml`: add the road law to `returns`, `reads` ("trip summaries from EnergyNet's inputs or a
   1 Hz trace") and the ladder.

## Run order (after Phase 3 of the bake-off)
```
python3 src/exam_road.py                      # once; refuses if results/exam_road.json exists
python3 src/webdata.py && python3 src/build_page.py && python3 src/build_release.py
```
Dry-run of the converter on the harness's dev rehearsal: `python3 exam_road.py --rehearsal --out /tmp/x.json`
(the formula row only appears in the real exam).
