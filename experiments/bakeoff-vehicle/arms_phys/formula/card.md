# formula: a readable trip-energy law (LawFinder-style), one formula per powertrain

**What it is.** Three short formulas, found by symbolic regression on the simulated bank only, that give a trip's
energy per km from seven trip summaries. Same idea as LawFinder (`/home/claude/bb/lawfinder`): what does RoadSim
believe, in one line, and is it true? It is also LawFinder's missing fifth law (export in `../lawfinder_road/`).
The textbook rival is road-load (EnergyNet's E_phys, the `phys_anchor` arm). No network at inference; numpy only.

**The released laws (rung r1).** e in MJ/km, E = e x d.

| Powertrain | Law (constants to 4 digits) | Size |
|---|---|---|
| Conventional (petrol, diesel, CNG) | e = 3.406 p^0.835 + 0.385 x 1.184^(exp c) x m t exp(-0.058 m) | 21 |
| Hybrid (P2, series, power-split) | e = 2.832 p + 0.583 x 0.0178^n x t + 0.489 c n + ln m - 0.473 | 22 |
| Battery-electric | e = 1.101 p - 1.443 n + n exp(-0.01715 T t) + 0.0406 m t | 20 |
| Plug-in | f E_bev + (1 - f) E_hev (EnergyNet's charge-depleting share f, as road-load does) | |

Variables (`lawvars.py`, the same numbers on both tracks): p = positive wheel work per km (aero, rolling, climbing,
accelerating, at the tier's road load; MJ/km); n = negative wheel work per km (braking and descending); t = minutes
per km; m = mass (t; exact, reported or class mass by tier); T = mean ambient temperature (C); c = start state (-1 warm,
0 unknown, +1 cold soak); d = km. Exact trees: `laws.json`.

How to read them:
- Conventional: wheel work at an efficiency that improves with load (p^0.835: light-load driving is inefficient, which
  road-load's fixed 28 % misses), plus a time term (idle, accessories, warm-up) that grows with mass and is 49 % larger
  after a cold start than when warm (1.184^e vs 1.184^(1/e)). Road-load: e = 3.515 p + 0.279 (m / 1.5)^0.7 t.
- Battery-electric: regen gives back 1.443 - exp(-0.017 T t) of the braking work: about 0.84 n at 20 C and 1.5 min/km,
  0.15 n at -10 C (a cold battery takes little regen), plus 0.041 MJ per tonne per minute of driving (~0.7 kW per
  tonne: heating, cooling, electronics).
- Hybrid: road-load-like 2.83 p, an idle/engine-on term that shrinks when there is a lot of braking energy to recover
  (0.018^n), and a mass offset.

**How it was found** (`../src/search_law.py`, `lawsearch.py` vendored from LawFinder with one option added:
`log_target`, squared log error). Island evolution of expression trees (+ - * / ^ exp log sqrt inv sq), constants by
Levenberg-Marquardt, a hall of fame per complexity (<= 25 nodes), road-load's own forms planted as seeds. Data: the
28,800-trip train bank seen through two independent tier draws, per group (ice 26,660 rows, hev 12,274, bev 14,088),
target y / d per km (rows with y > 0.02 MJ, d > 0.3 km). 20 min wall per group on a shared core (80-120 k trees).

**Ratchet** (rule: sim val skill > incumbent + 0.01 AND dev D16 not worse by > 0.01; laws inside a rung are picked
on simulations only; all numbers from `harness/evaluate.py`, both tracks equal to 1e-4):

| Rung | Change | Sim val skill | MdAPE | Dev D16 | Decision |
|---|---|---|---|---|---|
| r0 | s1 front knees (LawFinder's rule): ice 3.37 p + t, hev 3.32 p + 0.25, bev c16 | 0.231 | 0.182 | 0.2060 | base |
| **r1** | **s1: the most accurate law of each front (ice c21, hev c22, bev c20)** | **0.323** | **0.160** | **0.1752** | **released** |
| r2 | s2: + dynamometer flag as a variable; best of s1 + s2 per group (ice s2 c18, bev s2 c24) | 0.322 | 0.160 | 0.1518 | rejected (skill +0.000) |
| (s3) | a dedicated plug-in law (+ f): no front member beats r1's mixing rule on simulated plug-ins | | | | no rung |

r2 is the honest miss of the rule: dynamometer trips are 9 % of the sim val set, so a median-based skill cannot see the
dyn term, while on dev it fixes EPA highway (0.262 -> 0.096) and FTP (0.194 -> 0.108). It is reported, not released.
Every front member's sim and dev scores: `work/ladder_<search>_<group>.json` (the full Pareto ladder).

**References** (same harness): road-load E_phys skill 0.048, MdAPE 0.210, D16 0.2278; EnergyNet v2.1 skill 0.680,
MdAPE 0.075, D16 0.1133; climatology D16 0.309.

**Released r1 on dev** (MdAPE / bias / cov50 / cov90):

| Set | n | MdAPE | bias | cov50 / cov90 | E_phys | EnergyNet v2.1 |
|---|---|---|---|---|---|---|
| VED ICE trips | 645 | 0.098 | -0.087 | 0.60 / 0.99 | 0.299 | 0.066 |
| VED HEV | 275 | 0.162 | -0.129 | 0.45 / 0.87 | 0.202 | 0.126 |
| VED EV | 258 | 0.142 | +0.013 | 0.55 / 0.91 | 0.153 | 0.140 |
| VED PHEV | 318 | 0.195 | -0.141 | 0.62 / 0.92 | 0.225 | 0.163 |
| EPA conv FTP / HWY | 2,877 / 2,835 | 0.194 / 0.262 | +0.19 / +0.26 | 0.42 / 0.84, 0.24 / 0.78 | 0.190 / 0.093 | 0.093 / 0.070 |
| EPA conv US06 / cold FTP / SC03 | 371 / 280 / 302 | 0.167 / 0.133 / 0.140 | | | 0.078 / 0.254 / 0.320 | 0.053 / 0.109 / 0.096 |
| EPA hybrid FTP / HWY | 70 / 72 | 0.095 / 0.228 | | | 0.152 / 0.054 | 0.157 / 0.100 |
| EPA BEV city / HWY | 98 / 97 | 0.151 / 0.109 | | | 0.201 / 0.084 | 0.089 / 0.059 |
| Mersin diesel / CNG | 56 / 56 | 0.106 / 0.223 | | | 0.171 / 0.304 | 0.130 / 0.170 |
| ICCT truck | 3 | 0.397 | | | 0.865 | 0.190 |
| ZTBus legs | 5 | 0.170 | | | 0.378 | |

Paired bootstrap on dev (VED ICE): beats E_phys (-0.20) and the class median; ties the window label x distance; loses to
MOVES-like (dev-fitted, in-sample) and EnergyNet v2.1 (+0.03). Mean over the 16 D16 sets: cov50 0.50, cov90 0.85,
CRPS (log) 0.128 (E_phys anchor 0.153, EnergyNet 0.093). Sim val: cov50 0.49, cov90 0.90, CRPS 0.137.

**Where it wins and loses.** It wins over road-load where road-load is worst: stop-and-go and cold trips (VED ICE
0.299 -> 0.098, cold FTP 0.254 -> 0.133, SC03 0.320 -> 0.140, heavy duty). It loses to road-load on steady
dynamometer highway cycles (EPA HWY 0.093 -> 0.262, US06 0.078 -> 0.167, hybrid HWY 0.054 -> 0.228): the time term
was fitted mostly on road trips with HVAC and accessory loads, and RoadSim agrees that dyno cycles use ~15-20 % less
(the r2 fix). Plug-ins stay hard (SOC and strategy are mostly unknown).

**Bands.** Simulation only: q_k = E x exp(r_k), r_k = the 5 / 25 / 50 / 75 / 95 % quantiles of log(y / E) on the
simulated train bank per architecture x light / heavy duty (`bands.json`). `formula_conformal` recalibrates them on dev.

**Compute.** 0.87 CPU-h of searching (7 searches; 3 used by the released rung), one thread; scoring ~0.5 CPU-h.
Inference: ~1 microsecond per row (table), ~0.2 ms per row (trace, physics integrals).

**Limits.** Time-boxed searches on a contended core: another seed gives another front (the ice c16-c25 members are
within 0.008 of each other in sim val median |log error|). The laws inherit RoadSim's beliefs and the tier's road load; no transmission,
engine size or window-sticker label enters them (EnergyNet uses all of these). The battery-electric law goes below
zero when braking / descending work exceeds the climb (n > ~p); a trip energy <= 0 falls back to E_phys: 1 of 3,985 sim
val trips and 1 dev record, both plug-ins on downhill trips.
