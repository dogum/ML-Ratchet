# BatteryPilot v1.3 — model card

**What it does.** A causal dispatch policy for a grid-scale battery doing energy arbitrage. Before each hour it
returns an offer curve: the marginal value of one stored MWh at 0, 25, 50, 75 and 100 % state of charge. The battery
sells while the real-time price (after losses and wear) beats that value and buys while the stored MWh is worth more
than its price. Works for 1, 2 and 4 h batteries; trained for ERCOT (primary) and CAISO (secondary).

**Versions.** v1.0 (released 14 Nov 2026) trained on GridSim physics 1.0, whose outage rates came from exam-period
reports. v1.1 (same day) is the leak fix: recalibrated and retrained on physics 1.1. v1.3 (15 Nov 2026, the shipped
default) trains on GridSim physics 1.2, which adds real-time scarcity the day-ahead market cannot see, and is a hybrid:
the rolling day-ahead optimiser's plan as a value curve plus a learned residual. (v1.2 was an attempt that released
nothing.) The sections below describe the recipe and v1.0's numbers; the v1.1 and v1.3 sections give what changed
and each exam look. v1.3 did better than v1.1 on dev and worse on the exam.

**Trained on physics, graded on reality.** No real price path was ever a training sample. Real dev-year prices
(ERCOT 2019 and 2021, CAISO Aug–Dec 2023) only calibrated the simulator and gated rungs. The exam (ERCOT 2023–2025,
CAISO 2024–2025) was run once, with the released network.

## Inputs (hourly, all causal)
- real-time prices of the last 24 h; day-ahead prices for the rest of today and, after 14:00 local, tomorrow
  (36 h read, masked where not yet published); RT−DA spread of the last 6 h
- the balancing authority's day-ahead demand forecast for the next 24 h (EIA-930), relative to its 30-day max
- last observed temperature (BA population-weighted, NASA POWER); hour, day of year, weekend
- battery: duration, wear cost per MWh / price scale, price scale (7-day median DA)
Prices are scaled by the 7-day median DA price and squashed with asinh. 135 features.

## Output
Value curve λ(SOC) at 5 knots ($/MWh of stored energy, strictly decreasing). Execution (`step_offer`): discharge down
to the SOC where λ = η_d (p − c_wear); charge up to the SOC where λ = p / η_c; respect power and energy limits.
The offer is fixed before the hour and cleared against that hour's RT price, as a battery's offer curve is in
ERCOT's real-time market.

## Battery physics (`Battery`, `settle`)
- 1 MW grid side; usable energy = P × duration (5–95 % of nameplate); round trip 85 % (√ on each way)
- wear: rainflow cycle counting (ASTM E1049; matches the `rainflow` package exactly on 200 random series) with a
  Wöhler depth curve (life used per cycle = depth^1.3 / 6,000; Xu et al. 2018 framework), LiFePO4 temperature and
  C-rate dependence (Wang et al. 2011: Ea 31.7 kJ/mol, 370.3 J/mol per C), cell temperature in an HVAC container
  (25 °C + 0.25 × (ambient − 25) + 3 °C × C-rate), replacement cost $130/kWh nameplate. Linear proxy used inside the
  optimisers: $25.4 / 22.5 / 21.1 per MWh sold (1 / 2 / 4 h, 25 °C). Calendar ageing is not charged.
- checks (`training/checks/battery_checks.json`): GridData's perfect-foresight, daily-spread and threshold baselines reproduced
  on 89 dev days at two hubs to 1e-13 $ (one spread day differs by $1.65 at HB_WEST: a price tie); energy balance of
  the executor 1e-16 MWh; rainflow cost of a summer PF schedule $2,414 vs linear proxy $2,416.

## Simulator (training data)
GridSim (Phase 1, unmodified) supplies the real ERCOT / CAISO fleets of 2018, 2020, 2022 (heat-rate curves, EIA-923
fuel prices + Henry Hub daily, random Markov forced and planned outages, the dev-2019 coal offer adder). BatteryPilot's
wrapper (`training/pricesim.py`) adds:
- merit-order price from pmin blocks and 3 incremental heat-rate segments per cluster; slow units needed for the
  day's peak stay at pmin (80 %), below that wind/solar set a negative price
- gas-turbine ambient derate (0.6 %/K CT, 0.4 %/K CC around the 35 °C summer rating)
- cold-weather outages f(T) = 0.55 Φ((−3 °C − T)/3 K), persisting with a 36 h time constant (fitted on Uri, dev 2021:
  simulated hours above $1,000 = 180 and 166 for two outage draws vs 153 real; 92–95 % of the real ones covered)
- a storage fleet that flattens the daily net load
- ERCOT-style ORDC: adder = (VOLL − λ) × LOLP(R − X), LOLP = 1 − Φ((R − X − μ)/σ), μ = −4,000 MW, σ = 5,000 MW
  (fitted on 2019 + 2021 without Uri: daily max/min quantiles, hours above $100/$300/$1,000, |RT−DA|, PF value)
- real-time noise (Student-t, AR(1), scaled by net-load ramps); day-ahead = forecast inputs (real EIA-930 demand
  forecast errors, AR(1) wind/solar errors, outages known at noon the day before) + the expected ORDC adder over the
  empirical reserve-error distribution
- scenario draws: demand ×0.97–1.22, wind ×0.8–1.6, solar up to 45 % of peak demand, storage 0–20 % of peak,
  gas ×0.6–1.8, retirements 0–8 %, X 2,000/3,000 MW, cap $5,000/$9,000, jittered μ, σ, noise, cold severity
- a CAISO-like family on the CISO fleet: solar 35–80 % of peak, fixed imports ×0.4–0.95, carbon $10–40/t, caps
  $1,000/$2,000 (centred on solar ×1.6, imports ×0.6, which reproduce the duck curve of Aug–Dec 2023: simulated hourly
  medians $18–25 midday and $59–61 evening vs real $21–25 and $66–72)
Banks: 120 ERCOT + 45 CAISO-like path-years (training), 24 + 9 frozen.

## Network and training
MLP 135 → 128 → 128 → 5 (SiLU), 34,565 weights. Trained by direct policy search: the battery physics (offer execution,
limits, losses, linear wear) runs inside the loss in torch; each step samples 32–64 windows of 14–21 days and a battery
(1/2/4 h, random start SOC) and maximises revenue + value of energy left − value of energy at the start, each window
divided by its own 2 h perfect-foresight value. numpy inference == torch; JS port == numpy to 2e-6 (`training/checks/jscheck.json`); the site's live instrument matches numpy on the same float16 weights to 1e-5 (`site/tests/parity.test.ts`).

## Ratchet, v1.0 ladder (frozen score = share of perfect-foresight net revenue on the frozen simulated set, mean over path-years; dev = same on 10 real dev node-years)
Rule: keep a rung if frozen > incumbent + 0.01 and dev ≥ incumbent − 0.01. When the simulator changed, a new frozen set
was built and every rung re-scored in order (v2: R0 0.489, R1 0.533, R2 0.767; v3: R0 0.447, R1 0.570, R2 0.747).

| Rung | Change | Frozen set | Frozen capture | Dev capture | Decision |
|---|---|---|---|---|---|
| R0 | classical rolling-horizon LP on day-ahead prices (incumbent) | v1_frozen | 0.102 | 0.593 | kept: starting incumbent |
| R1 | MLP value-curve policy trained by policy search on sim v1 (GridSim merit-order prices, no scarcity curve, no outage weather, no RT noise); 45 path-years | v1_frozen | 0.593 | 0.483 | rejected: frozen +0.491 but dev gate failed: dev capture 0.483 vs incumbent 0.593 (-0.110); the v1 prices have no scarcity, so the policy cycles on small spreads and is empty when spikes come |
| R2 | same network and policy search, trained on sim v2 (80 ERCOT path-years from 2018/2020/2022 inputs with dev-calibrated scarcity) | v2_frozen | 0.767 | 0.598 | kept: frozen 0.767 vs incumbent R0 0.489 (+0.278 > 0.01); dev 0.598 vs 0.593 (gate: not more than 0.01 below) -> kept, new incumbent |
| R3 | train on all 120 ERCOT path-years + 45 CAISO-like path-years (CISO fleet, duck-curve solar, carbon price, $1000-2000 caps), 2,500 steps | v3_frozen | 0.749 | 0.623 | rejected: frozen 0.7489 vs incumbent R2 0.7465 (+0.0024, below the 0.01 margin) -> rejected, although dev improved (0.623 vs 0.598) |
| R4 | fine-tune R3 on the same 165 path-years with longer windows (21 days), batch 64, lower learning rate (900 steps) | v3_frozen | 0.758 | 0.637 | kept: frozen 0.7580 vs incumbent R2 0.7465 (+0.0115 > 0.01); dev 0.637 vs 0.598 -> kept; released |

## Exam, v1.0 (real prices, never used for choosing) — $ per kW-year net of wear, mean over nodes
ERCOT: 4 hubs + 8 load zones, 2023–2025. CAISO: NP15, SP15, ZP26, DLAP PG&E/SCE/SDG&E, 2024–2025. 95 % intervals from a
weekly block bootstrap (2,000 draws, same weeks for all nodes and strategies).

| Market | Year | Battery | Perfect hindsight | BatteryPilot (95% CI) | Share of PF (95% CI) | DA optimiser | Spread rule | Threshold | BP − optimiser (95% CI) | BP cycles/yr | BP worst month, median node ($/kW) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAISO | 2024 | 1 h | 16.3 | 10.5 (6.6–16.0) | 64% (53–74%) | 12.0 | 10.4 | 1.6 | -1.5 (-2.2 to -0.8) | 269 | 0.10 |
| CAISO | 2024 | 2 h | 29.6 | 20.6 (14.8–28.8) | 70% (61–77%) | 22.3 | 19.4 | 5.0 | -1.7 (-3.1 to -0.4) | 268 | 0.28 |
| CAISO | 2024 | 4 h | 47.5 | 36.4 (27.9–46.5) | 77% (71–81%) | 37.5 | 32.3 | 9.9 | -1.1 (-3.2 to 0.7) | 246 | 0.64 |
| CAISO | 2025 | 1 h | 9.0 | 4.4 (3.3–5.6) | 49% (44–53%) | 5.0 | 4.3 | 1.1 | -0.6 (-1.2 to -0.0) | 239 | -0.06 |
| CAISO | 2025 | 2 h | 17.5 | 10.1 (7.8–12.6) | 58% (53–62%) | 10.7 | 9.5 | 3.0 | -0.6 (-1.6 to 0.5) | 242 | -0.12 |
| CAISO | 2025 | 4 h | 29.4 | 20.6 (15.9–25.4) | 70% (65–74%) | 19.7 | 17.3 | 6.0 | +0.9 (-0.5 to 2.4) | 222 | -0.17 |
| ERCOT | 2023 | 1 h | 87.8 | 47.1 (26.8–70.8) | 54% (50–60%) | 56.6 | 52.5 | 1.8 | -9.4 (-18.2 to -1.6) | 257 | 0.27 |
| ERCOT | 2023 | 2 h | 137.5 | 95.4 (52.8–153.7) | 69% (62–75%) | 93.7 | 84.5 | 6.9 | +1.7 (-6.7 to 9.3) | 244 | 0.55 |
| ERCOT | 2023 | 4 h | 196.3 | 162.5 (83.9–268.9) | 83% (76–88%) | 153.5 | 135.6 | 20.7 | +9.0 (-3.0 to 21.5) | 215 | 0.98 |
| ERCOT | 2024 | 1 h | 28.6 | 18.9 (11.5–29.3) | 66% (54–76%) | 18.4 | 16.2 | 0.8 | +0.5 (-1.3 to 2.3) | 251 | 0.10 |
| ERCOT | 2024 | 2 h | 47.7 | 35.3 (21.6–52.8) | 74% (64–82%) | 34.0 | 29.1 | 7.3 | +1.2 (-0.3 to 2.7) | 247 | 0.32 |
| ERCOT | 2024 | 4 h | 63.7 | 51.1 (32.7–73.9) | 80% (72–85%) | 47.7 | 35.3 | 19.9 | +3.3 (1.1 to 5.4) | 212 | 0.58 |
| ERCOT | 2025 | 1 h | 20.1 | 9.8 (7.2–13.1) | 48% (39–59%) | 12.1 | 8.7 | 0.3 | -2.4 (-5.6 to 0.0) | 265 | 0.27 |
| ERCOT | 2025 | 2 h | 34.4 | 20.0 (15.8–24.6) | 58% (50–66%) | 22.6 | 16.8 | 4.4 | -2.6 (-6.4 to 0.6) | 265 | 0.70 |
| ERCOT | 2025 | 4 h | 51.1 | 34.8 (28.1–42.2) | 68% (60–75%) | 35.6 | 26.0 | 11.9 | -0.9 (-5.9 to 2.9) | 241 | 1.23 |

Mean share of perfect hindsight over node-years × durations:

| | BatteryPilot | DA optimiser | spread rule | threshold | real-price variant (comparison only) |
|---|---|---|---|---|---|
| ERCOT (36 node-years) | 0.667 | 0.685 | 0.561 | 0.114 | 0.690 |
| CAISO (12 node-years) | 0.642 | 0.681 | 0.589 | 0.158 | 0.564 |

BatteryPilot beats the spread rule in 89 % (ERCOT) / 81 % (CAISO) of node-year-battery cases, the threshold rule in all,
and the day-ahead optimiser in 57 % / 19 %. By duration (ERCOT): 1 h 0.56 vs 0.63, 2 h 0.67 vs 0.68, 4 h 0.77 vs 0.74.
It cycles less (about 244 cycles a year vs 300 for the optimiser in ERCOT, 248 vs 282 in CAISO). Its worst month was never worse than −$0.21/kW (CAISO, December).
Baselines: perfect foresight = annual LP on RT (upper bound); DA optimiser = rolling LP that re-plans at 14:00 with DA
prices as the forecast and follows the schedule; spread rule = GridData's (d cheapest DA hours, d dearest later);
threshold = GridData's (dev-period RT quartiles of the node).
Error bars are on revenue (bootstrap), not on a predicted quantity, so the coverage/widening step of the recipe does not apply.

The **real-price variant** (same network trained on real ERCOT 2018/2020/2022 node prices) breaks the physics-only rule
and is not released; it is reported only to show what the simulator costs or gains.

## v1.1: the outage-rate leak fix (14 Nov 2026)

**What leaked.** GridSim physics 1.0 took the forced and planned outage shares of coal, combined-cycle, combustion-turbine
and gas-steam units from Texas RE's 2024 and 2025 reliability assessments. Those reports describe exam-period years.
GridSim physics 1.1 replaces those four rows with sources published before 2023: PJM's 2017-2021 class average EFORd
(May 2022) for forced outages and Texas RE's 2017 assessment (8.3 % scheduled outage factor) for planned ones. Its coal
offer adder was re-fitted on dev 2019 (−5.73 $/MWh, was −6.62).

| Forced / planned share | coal | CC | CT | gas steam |
|---|---|---|---|---|
| physics 1.0 (Texas RE 2024/2025) | 0.15 / 0.11 | 0.08 / 0.11 | 0.08 / 0.06 | 0.13 / 0.19 |
| physics 1.1 (PJM 2017-2021, Texas RE 2017) | 0.136 / 0.083 | 0.036 / 0.083 | 0.058 / 0.083 | 0.136 / 0.083 |

**How BatteryPilot touched it.** Every simulated price path: `pricesim.py` builds its thermal stack with GridSim's
`prepare_fleet` (which fills the outage shares) and `availability(mode='random')`, and used GridSim's coal must-run and
coal adder (hard-coded −6.62). The dev scarcity calibration (`calib.py`: ORDC μ, σ, RT noise, the Uri cold-outage
curve), all banks and frozen sets, every rung decision and the released network were built on those outages. Real
prices never touched them: the dev gate and the exam use real prices only.

**How much it mattered** (dev years only; `training/leakcheck.py`, `training/ratchet_v1_1.json`):

| Dev twins (ERCOT HB_HUBAVG statistics) | physics 1.0, v1.0 calibration | physics 1.1, v1.0 calibration | physics 1.1, v1.1 calibration | real |
|---|---|---|---|---|
| thermal MW out on random outage (2019 fleet of 74.1 GW) | 14.8 GW | 10.8 GW | 10.8 GW | — |
| 2019 hours above $1,000 / $300 / $100 | 22.5 / 120 / 277 | 0 / 20 / 91 | 32.5 / 140 / 279 | 26 / 75 / 213 |
| 2019 daily max, 90th percentile ($/MWh) | 212 | 67 | 264 | 232 |
| 2019 2 h perfect-foresight value ($/kW-yr) | 84.0 | 27.9 | 75.6 | 103.3 |
| 2021 (Uri masked) hours above $300 | 53.5 | 1 | 61 | 47 |
| 2021 2 h perfect-foresight value ($/kW-yr) | 61.1 | 32.9 | 47.3 | 54.4 |
| calibration loss (stage A) | 3.67 | 21.07 | 4.17 | — |
| Uri: hours above $1,000 (2 draws) | 180, 166 | 156, 149 | 158, 150 | 153 |

With the pre-2023 outage rates and the v1.0 calibration, scarcity almost disappears from the dev twins. The v1.0
calibration had absorbed the extra ~4 GW of leaked outages: refitted on physics 1.1, the ORDC μ moves from −4,000 MW to
0 MW (σ 5,000 MW, noise and DA scale unchanged; cold-outage f_max 0.55 → 0.40). On the release's own grid the best point
sat on the opposite edge (μ −2,000, σ 6,500, loss 4.25); the grid was widened once, as in v1.0, and μ 0 is again an edge.

On frozen sets with the same scenario seeds (24 ERCOT + 9 CAISO-like path-years), the v1.0 network captures 0.798 on
physics 1.0, 0.771 on physics 1.1 with the v1.0 calibration and 0.778 on the rebuilt v4 set (physics 1.1, v1.1
calibration). The rolling day-ahead LP scores 0.516, 0.421 and 0.504. Dev capture is on real prices and does not move
(v1.0 network 0.637, LP 0.593).

**What was done.** `pricesim.py` reads GridSim's coal must-run and adder from `gridsim.fleet.calibration()` for the
active physics version (`GRIDSIM_PHYSICS`; 1.0 gives −6.62 and reproduces the release's calibration numbers, twin test
and Uri hours exactly). `calib.py` was re-run on dev 2019, 2021 and Uri. The banks were rebuilt on physics 1.1: 120 ERCOT
+ 45 CAISO-like + 45 sim-v1 training path-years and frozen set v4 (24 + 9). Every rung was retrained and re-decided in
order with the rule unchanged:

| Rung | Change | v4 frozen | Dev capture | Decision |
|---|---|---|---|---|
| R0 | rolling day-ahead LP | 0.504 | 0.593 | kept: starting incumbent |
| R1 | policy search on sim v1, 45 path-years | 0.607 | 0.489 | rejected: dev gate (−0.104) |
| R2 | sim v2 with the v1.1 calibration, 80 ERCOT path-years | 0.779 | 0.638 | kept; **released as v1.1** |
| R3 | + all 120 ERCOT and 45 CAISO-like path-years, 2,500 steps | 0.788 | 0.644 | rejected: +0.0085 < 0.01 |
| R4 | fine-tune R3, 21-day windows, batch 64, lr 5e-4, 900 steps | 0.789 | 0.653 | rejected: +0.0098 < 0.01 |

R4 missed the margin by 0.0002 and is rejected as written; R2 is released. It was trained on ERCOT paths only.
`batterypilot_weights.npz` is now v1.1 (R2) and `PolicyNet()` loads it. The v1.0 network is kept as
`batterypilot_weights_v1_0.npz` (`bp.WEIGHTS_V1_0`), so v1.0's exam numbers stay reproducible.

**Exam, second look** (v1.1, run once on the same 48 node-years after the fix; labelled, next to the first; nothing was
chosen with it). Mean share of perfect hindsight:

| | v1.0 (first look) | v1.1 (second look) | DA optimiser | spread rule |
|---|---|---|---|---|
| ERCOT 2023-25, all batteries | 0.667 | 0.679 | 0.685 | 0.561 |
| ERCOT 1 h / 2 h / 4 h | 0.559 / 0.671 / 0.769 | 0.578 / 0.680 / 0.780 | 0.629 / 0.684 / 0.742 | |
| CAISO 2024-25, all batteries | 0.642 | 0.614 | 0.681 | 0.589 |
| CAISO 1 h / 2 h / 4 h | 0.563 / 0.634 / 0.730 | 0.521 / 0.604 / 0.719 | 0.641 / 0.680 / 0.723 | |
| beats the optimiser (ERCOT / CAISO cases) | 57 % / 19 % | 63 % / 11 % | | |
| beats the spread rule (ERCOT / CAISO cases) | 89 % / 81 % | 89 % / 58 % | | |
| cycles a year (ERCOT / CAISO) | 244 / 248 | 256 / 240 | 300 / 282 | |

The verdicts do not change in ERCOT: still a loss on the pooled mean, still a win with 4 h batteries (2024 +$4.2/kW-yr,
95 % CI [+2.1, +6.1]). CAISO gets worse, and its thin 4 h win becomes a loss (0.719 vs 0.723). v1.1 never trained on
the CAISO-like family, because the rung that adds it (R3) missed the margin. Full tables: `training/checks/exam_report_v1_1.json`.
Twin test, v1.1: 94 % on twins of dev 2019 vs 82 % on real 2019; 2021 twins 71-75 % vs 38 % real
(`training/checks/twin_r2_v4.json`). The sim-to-real gap is as before.

## v1.3: GridSim physics 1.2 and the day-ahead backbone (15 Nov 2026)

**Why.** v1.1 lost both pooled exam rows to the rolling day-ahead optimiser. The v1.2 attempt (not released) found
the cause on dev: a hybrid anchored on the optimiser's plan beat v1.1 on real dev (0.661 vs 0.638) but lost on the
simulated frozen set (0.702 vs 0.779), because simulated real-time spikes were too easy to see from the day-ahead
price. Measured on dev (`training/predictability.py`): the real DA price covers 31 % of the RT value of hours above
$300 in 2019 and 13 % in 2021 without Uri; physics-1.1 twins 69-72 % and 71-80 %.

**Simulator.** GridSim physics 1.2 (`gridsim.market`, opt-in) adds real-time scarcity the day-ahead market cannot see,
fitted on ERCOT 2018/2020/2022 prices only (shared/grid/gridsim/LIMITS.md I): unforecast events whose start rate
follows the DA price's rank within the day and its level against the past week, the month and the real demand-forecast
miss; geometric duration (mean 2 h); a generalised Pareto size that grows with the DA level. It is added to RT and its
expectation to DA, with the rate scaled ×0.6-1.5 per path, on ERCOT paths only (no CAISO fit). The dev calibration was
re-fitted with it, now also matching how much of the spike value the DA sees: ORDC μ −1,000 MW, σ 3,500 MW (a grid
edge), cold f_max 0.55 (a grid edge); Uri 166 and 156 hours above $1,000 vs 153 real. Twins of dev 2019 now have DA
coverage 60 % (real 31 %) and a 2 h perfect-foresight value of $77-82/kW-yr (real $103). Banks and frozen set v6 were
rebuilt with the v4 seeds.

GridSim's process took three fits. The first placed events by clock hour; on its twins the rolling optimiser earned
$9.6-11.0k/MW in hours with RT − DA > $100 against $47.8k in real 2019, because real spikes fall in the DA's dearest
hours. The second fit added the DA shape; the third added the size term. A first BatteryPilot build on the first fit
(frozen set v5) is kept as probes in `training/ratchet_v1_3.json`; the ladder is decided on v6.

**Ladder v1.3** (rule unchanged; every v1.1 and v1.2-attempt decision re-taken in order; `training/ratchet_v1_3.json`):

| Rung | Change | v6 frozen | Dev | Decision |
|---|---|---|---|---|
| R0 | Rolling day-ahead LP | 0.384 | 0.593 | incumbent |
| R1 | MLP on sim v1 (re-scored, bank unchanged) | 0.617 | 0.489 | rejected: dev gate |
| R2 | MLP, 80 ERCOT path-years | 0.755 | 0.628 | kept |
| R3 | MLP, 120 ERCOT + 45 CAISO-like | 0.769 | 0.596 | rejected: dev gate |
| R4 | R3 fine-tuned, 21-day windows | 0.776 | 0.599 | rejected: dev gate |
| R5 | Backbone alone (reference) | 0.673 | 0.630 | — |
| R6 | Hybrid, running minimum | 0.732 | 0.581 | rejected |
| R8 | Hybrid, knots ≥ 0.01 apart, 80 ERCOT | 0.772 | 0.660 | **kept, released** |
| R9 | R8 on 165 path-years | 0.778 | 0.685 | rejected: +0.006 |
| R10 | R3 recipe on the v5 banks (chosen after seeing its v5 dev score) | 0.769 | 0.658 | rejected |
| R11 | R9 fine-tuned, 21-day windows | 0.780 | 0.681 | rejected: +0.008 |

R8 cleared the margin by 0.007: seeds 0, 1, 2 give frozen 0.772, 0.772, 0.767 (sd 0.003) and dev 0.660, 0.667, 0.644
(sd 0.012). v1.1 re-scored on v6: 0.735 frozen, 0.638 dev. R8 beats it on both and is released as v1.3.

**The network.** The backbone (`value_curves`) is the rolling optimiser's plan as a value curve: at the first hour and
then at 14:00 local, a backward dynamic programme over the known DA prices (85 % round trip, linear wear proxy, no value
at the horizon end, 201 SOC points) gives the marginal value of stored energy, read at the 5 knots. The MLP reads the 135
features plus asinh(backbone / price scale) and outputs a residual; each knot is kept at least 0.01 below the one
before. 140 → 128 → 128 → 5, 35,205 weights; the last layer starts at zero, so training starts from the backbone.
Numpy in `batterypilot.py` matches the training code exactly; the browser port matches it to 3.3e-6 (float16 weights
move the knots by at most 5.8e-4).

**Did the simulator fix work?** In part. On twins of dev 2019 BatteryPilot now captures 76-85 % (real 82 %, 94 % on
1.1 twins) and the backbone 64-78 % (real 67 %), but the rolling optimiser only 55-56 % (real 79 %, 90 % on 1.1 twins):
the simulator now under-rates following the DA plan. Over each ladder's own rungs the v6 frozen set ranks like dev
(Spearman 0.77, the same top two: R9 and R11) where v4 did not (0.68; best frozen R4, best dev R6). Over 10 fixed
policies (the v1.1 and v1.2-attempt networks, the backbone and the LP) it agrees less than v4 (0.43 vs 0.67)
(`training/checks/agreement_v1_3*.json`).

**Exam, v1.3** (run once after the release choice was committed; v1.1 re-scored on the same run, identical to its
second look; `training/checks/exam_report_v1_3.json`). Share of perfect hindsight, net of wear:

| | v1.0 | v1.1 | **v1.3** | DA optimiser | Spread | Threshold |
|---|---|---|---|---|---|---|
| ERCOT, all | 0.667 | 0.679 | **0.671** | 0.685 | 0.561 | 0.114 |
| ERCOT 1 h | 0.559 | 0.578 | **0.548** | 0.629 | 0.531 | 0.018 |
| ERCOT 2 h | 0.671 | 0.680 | **0.681** | 0.684 | 0.570 | 0.109 |
| ERCOT 4 h | 0.769 | 0.780 | **0.785** | 0.742 | 0.583 | 0.216 |
| CAISO, all | 0.642 | 0.614 | **0.594** | 0.681 | 0.589 | 0.158 |
| CAISO 1 h | 0.563 | 0.521 | **0.483** | 0.641 | 0.552 | 0.104 |
| CAISO 2 h | 0.634 | 0.604 | **0.579** | 0.680 | 0.592 | 0.166 |
| CAISO 4 h | 0.730 | 0.719 | **0.718** | 0.723 | 0.623 | 0.203 |

v1.3 wins only ERCOT 4 h (2023 +$12.9/kW-yr, CI [+3.9, +24.7]; 2024 +$4.9, [+3.0, +6.6]; 2025 −$0.7, [−5.5, +2.9]).
It is worse than v1.1 on both pooled rows although it was better on dev. The dev gain came from one year: 2021 (with
Uri) 0.48-0.50 vs 0.37-0.38, while 2019 fell (0.77-0.78 vs 0.81-0.82) and so did CAISO Aug-Dec 2023 (0.73 / 0.81 vs
0.78 / 0.84). The pooled dev gate let the Uri year outweigh the others; 2023-25 had no Uri. 1 h batteries and CAISO got
worse, as they had on 2019 and CAISO dev. About 222 (ERCOT) and 215
(CAISO) cycles a year vs 301 and 282 for the optimiser.

## Limits
- Loses to the classical day-ahead optimiser on the pooled mean in every version (v1.3: ERCOT 0.671 vs 0.685, CAISO
  0.594 vs 0.681; v1.0: 0.667 and 0.642). It wins only with 4 h ERCOT batteries (v1.3 0.785 vs 0.742) and loses with
  1 h batteries and in CAISO.
- v1.3 was chosen on a pooled dev gate that one year (2021, Uri) dominated; it was worse than v1.1 on dev 2019 and on
  CAISO dev, and is worse on the exam.
- GridSim physics 1.2 under-rates following the DA plan: on twins of dev 2019 the rolling optimiser captures 55-56 % vs
  79 % real. The RT scarcity has no CAISO fit.
- Twin test: on simulated twins of dev 2019 it captures 91–94 %, on real 2019 80 %; Uri 2021: 64–68 % vs 38 %.
  Real spikes are less predictable from DA prices than simulated ones.
- Energy arbitrage only, hourly settlement (mean of 15-min RT), no ancillary services, no 5-min prices, price taker,
  copper-plate simulator (no congestion; nodal differences untrained).
- CAISO family is hand-set from 5 dev months, more volatile than real CAISO.
- Calibration hit its grid edges (μ, σ, cold f_max). Simulated DA runs 10–20 % above RT (real 1–5 %).
- Wear parameters are assumptions; calendar ageing ignored.
- v1.0 trained on exam-period outage rates (GridSim physics 1.0); fixed in v1.1 (section above). The v1.1 outage rates
  are PJM class averages, not ERCOT's own pre-2023 rates, and the v1.1 scarcity calibration again ends on a grid edge.
- Single node-years are sensitive to the last decimals of the prices (a hold-or-trade decision near a tie can flip and
  the state of charge then follows another path): read one node-year as ±2-3 %. Pooled shares move by 0.001 at most.

## In this repository (14 Nov 2026)
- Files: `batterypilot.py` (model, battery physics, executor, baselines; numpy + scipy), `batterypilot_weights.npz`
  (v1.1, rung R2 of the v1.1 ladder; the default), `batterypilot_weights_v1_0.npz` (v1.0, rung R4), `demo.py`, `examples/` (3 real exam node-years), `training/` (every script as run, paths through
  `training/_paths.py`; `training/checks/` and `training/ratchet.json` are the release's logs), `lab/` (the original
  page), `web/export_instrument.py` (the site's live instrument).
- With the v1.0 weights the demo reproduces the release demo line for line; it now runs the v1.1 default (golden
  `tests/golden/batterypilot.txt` updated). The v1.0 exam was re-scored from the v1.0 weights on
  `shared/grid/data` with `training/exam.py` + `training/report.py`: every baseline reproduces in all 144
  node-year-battery rows; ERCOT 0.667 (reproduced); CAISO 0.641 vs 0.642 released (close); BatteryPilot rows differ by
  at most $0.29/kW-yr, and one CAISO case flips against the spread rule (beats it in 78 % instead of 81 %). The
  released numbers above are unchanged; NOTEBOOK.md has the details.

## Data and licences
ERCOT DAM/RTM settlement point prices (public ERCOT market data); CAISO OASIS LMPs (public); EIA-930 (public domain);
NASA POWER (cite NASA LaRC POWER); EIA-860/923 via PUDL (CC-BY-4.0); fleets, outages and fuel prices via GridSim.
Code and weights: MIT (ML Ratchet).
