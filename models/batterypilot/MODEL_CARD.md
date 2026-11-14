# BatteryPilot v1.0 — model card

**What it does.** A causal dispatch policy for a grid-scale battery doing energy arbitrage. Before each hour it
returns an offer curve: the marginal value of one stored MWh at 0, 25, 50, 75 and 100 % state of charge. The battery
sells while the real-time price (after losses and wear) beats that value and buys while the stored MWh is worth more
than its price. Works for 1, 2 and 4 h batteries; trained for ERCOT (primary) and CAISO (secondary).

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

## Ratchet (frozen score = share of perfect-foresight net revenue on the frozen simulated set, mean over path-years; dev = same on 10 real dev node-years)
Rule: keep a rung if frozen > incumbent + 0.01 and dev ≥ incumbent − 0.01. When the simulator changed, a new frozen set
was built and every rung re-scored in order (v2: R0 0.489, R1 0.533, R2 0.767; v3: R0 0.447, R1 0.570, R2 0.747).

| Rung | Change | Frozen set | Frozen capture | Dev capture | Decision |
|---|---|---|---|---|---|
| R0 | classical rolling-horizon LP on day-ahead prices (incumbent) | v1_frozen | 0.102 | 0.593 | kept: starting incumbent |
| R1 | MLP value-curve policy trained by policy search on sim v1 (GridSim merit-order prices, no scarcity curve, no outage weather, no RT noise); 45 path-years | v1_frozen | 0.593 | 0.483 | rejected: frozen +0.491 but dev gate failed: dev capture 0.483 vs incumbent 0.593 (-0.110); the v1 prices have no scarcity, so the policy cycles on small spreads and is empty when spikes come |
| R2 | same network and policy search, trained on sim v2 (80 ERCOT path-years from 2018/2020/2022 inputs with dev-calibrated scarcity) | v2_frozen | 0.767 | 0.598 | kept: frozen 0.767 vs incumbent R0 0.489 (+0.278 > 0.01); dev 0.598 vs 0.593 (gate: not more than 0.01 below) -> kept, new incumbent |
| R3 | train on all 120 ERCOT path-years + 45 CAISO-like path-years (CISO fleet, duck-curve solar, carbon price, $1000-2000 caps), 2,500 steps | v3_frozen | 0.749 | 0.623 | rejected: frozen 0.7489 vs incumbent R2 0.7465 (+0.0024, below the 0.01 margin) -> rejected, although dev improved (0.623 vs 0.598) |
| R4 | fine-tune R3 on the same 165 path-years with longer windows (21 days), batch 64, lower learning rate (900 steps) | v3_frozen | 0.758 | 0.637 | kept: frozen 0.7580 vs incumbent R2 0.7465 (+0.0115 > 0.01); dev 0.637 vs 0.598 -> kept; released |

## Exam (real prices, never used for choosing) — $ per kW-year net of wear, mean over nodes
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
It cycles less (about 245 vs 300 cycles a year). Its worst month was never worse than −$0.21/kW (CAISO, December).
Baselines: perfect foresight = annual LP on RT (upper bound); DA optimiser = rolling LP that re-plans at 14:00 with DA
prices as the forecast and follows the schedule; spread rule = GridData's (d cheapest DA hours, d dearest later);
threshold = GridData's (dev-period RT quartiles of the node).
Error bars are on revenue (bootstrap), not on a predicted quantity, so the coverage/widening step of the recipe does not apply.

The **real-price variant** (same network trained on real ERCOT 2018/2020/2022 node prices) breaks the physics-only rule
and is not released; it is reported only to show what the simulator costs or gains.

## Limits
- Roughly ties, does not beat, the classical day-ahead optimiser; loses with 1 h batteries and in CAISO.
- Twin test: on simulated twins of dev 2019 it captures 91–94 %, on real 2019 80 %; Uri 2021: 64–68 % vs 38 %.
  Real spikes are less predictable from DA prices than simulated ones.
- Energy arbitrage only, hourly settlement (mean of 15-min RT), no ancillary services, no 5-min prices, price taker,
  copper-plate simulator (no congestion; nodal differences untrained).
- CAISO family is hand-set from 5 dev months, more volatile than real CAISO.
- Calibration hit its grid edges (μ, σ, cold f_max). Simulated DA runs 10–20 % above RT (real 1–5 %).
- Wear parameters are assumptions; calendar ageing ignored. GridSim's outage rates come from Texas RE 2024/2025
  published fleet averages (LIMITS D of GridSim): a possible mild leak of exam-period fleet statistics.
  Not fixed in this release; the platform plan is to replace them with pre-2023 NERC GADS values and re-score.
- Single node-years are sensitive to the last decimals of the prices (a hold-or-trade decision near a tie can flip and
  the state of charge then follows another path): read one node-year as ±2-3 %. Pooled shares move by 0.001 at most.

## In this repository (14 Nov 2026)
- Files: `batterypilot.py` (model, battery physics, executor, baselines; numpy + scipy), `batterypilot_weights.npz`
  (rung R4), `demo.py`, `examples/` (3 real exam node-years), `training/` (every script as run, paths through
  `training/_paths.py`; `training/checks/` and `training/ratchet.json` are the release's logs), `lab/` (the original
  page), `web/export_instrument.py` (the site's live instrument).
- The demo reproduces the release demo line for line. The exam was re-scored from the shipped weights on
  `shared/grid/data` with `training/exam.py` + `training/report.py`: every baseline reproduces in all 144
  node-year-battery rows; ERCOT 0.667 (reproduced); CAISO 0.641 vs 0.642 released (close); BatteryPilot rows differ by
  at most $0.29/kW-yr, and one CAISO case flips against the spread rule (beats it in 78 % instead of 81 %). The
  released numbers above are unchanged; NOTEBOOK.md has the details.

## Data and licences
ERCOT DAM/RTM settlement point prices (public ERCOT market data); CAISO OASIS LMPs (public); EIA-930 (public domain);
NASA POWER (cite NASA LaRC POWER); EIA-860/923 via PUDL (CC-BY-4.0); fleets, outages and fuel prices via GridSim.
Code and weights: MIT (ML Ratchet).
