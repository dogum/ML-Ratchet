<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# BatteryPilot <sub>v1.0</sub>

> **When should a grid battery charge and discharge, knowing only the past?**

*Before each hour, what is one stored MWh worth to this battery — and so should it charge, hold or sell at the real-time price?*

BatteryPilot is a causal dispatch policy for a 1 MW grid battery doing energy arbitrage. Before each hour a small network returns a value curve: the worth of one stored MWh at 0, 25, 50, 75 and 100 % full. The battery uses that curve as its offer for the hour, cleared against the real-time price, with 85 % round trip and rainflow wear. It was trained only on simulated price paths (GridSim fleets plus a scarcity model calibrated on dev years) and examined once on real ERCOT 2023-25 and CAISO 2024-25 prices. It earns 67 % (ERCOT) and 64 % (CAISO) of perfect hindsight net of wear. That beats the daily-spread and threshold rules but not a rolling day-ahead optimiser (69 % and 68 %).

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/batterypilot/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/batterypilot/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Hourly and causal: real-time prices of the last 24 h, day-ahead prices for the rest of today (and tomorrow after 14:00 local; 36 h, masked where not yet published), the RT-DA spread of the last 6 h, the balancing authority's day-ahead demand forecast for 24 h (EIA-930, relative to its 30-day max), the last temperature, hour, day of year, weekend, and the battery (duration, wear cost / price scale, price scale). Prices are scaled by the 7-day median DA price and squashed with asinh; 135 features. |
| **Returns** | Value curve λ(SOC): $/MWh of stored energy at 5 state-of-charge knots, strictly decreasing<br>Executed by the offer rule: discharge down to the SOC where λ = η_d (p − c_wear), charge up to where λ = p / η_c, within power and energy limits; cleared against the hour's RT price<br>Settled: gross revenue, rainflow wear cost, net revenue ($) and cycles; no error bar on the curve (the exam's intervals are on revenue, from a weekly block bootstrap) |
| **Learned from** | Battery: 1 MW grid side, usable energy = P × duration (5-95 % of nameplate), 85 % round trip (√ each way), wear by rainflow cycle counting (ASTM E1049) on a Wöhler depth curve (depth^1.3 / 6,000 per cycle) with LiFePO4 temperature and C-rate dependence (Wang et al. 2011) in an HVAC container, at $130/kWh nameplate. Prices: the shared GridSim fleet dispatch (unmodified) for the real ERCOT and CISO fleets of 2018, 2020 and 2022, wrapped by BatteryPilot's price simulator (training/pricesim.py) with a merit stack, scarcity and day-ahead prices. |
| **Noise** | Simulated real-time noise (Student-t, AR(1), scaled by net-load ramps) and real EIA-930 day-ahead demand-forecast errors laid on the simulated day-ahead; scenario randomisation of demand, renewables, storage, gas price, retirements and scarcity parameters |
| **Network** | MLP 135 → 128 → 128 → 5 (SiLU), 34,565 weights; outputs the top knot and 4 softplus steps (a strictly decreasing curve). Trained by direct policy search: the battery physics (offer execution, limits, losses, linear wear) runs inside the loss in torch, maximising revenue + value of energy left − value at the start over 14-21-day windows, each divided by its own 2 h perfect-foresight value. |
| **Physics also enters** | inputs: The battery's own wear cost (temperature-dependent linear proxy from the rainflow model) and duration; prices scaled by a causal 7-day median<br>output: The network's value curve feeds a physics executor: an offer rule with round-trip losses, power and energy limits, and rainflow wear; the battery physics sits inside the training loss |
| **Real data's jobs** | calibration, gate, exam (never training labels) |
| **Dev set** | GridData's frozen time split (data/proc/splits.json, sha256 895b4e9a…, seed 20261113): dev = ERCOT 2019 and 2021 at HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_WEST; CAISO Aug-Dec 2023 at TH_NP15 and TH_SP15 (OASIS starts 2023-08). Simulator inputs only from the train-input years 2018, 2020, 2022. |
| **Exam** | ERCOT 2023, 2024, 2025 at 4 hubs + 8 load zones (36 node-years); CAISO 2024, 2025 at TH_NP15, TH_SP15, TH_ZP26 and DLAP PG&E, SCE, SDG&E (12 node-years); 1, 2 and 4 h batteries. Never used for choosing anything; 2026 is held in reserve. |

## Run it

```bash
cd models/batterypilot
pip install numpy scipy
python demo.py        # about 15 s
```

or from anywhere in the repo: `mlratchet demo batterypilot`.

```python
import csv, numpy as np
import batterypilot as bp
net = bp.PolicyNet()                                   # batterypilot_weights.npz next to batterypilot.py
bat = bp.Battery(P=1.0, duration_h=2.0)                # 1 MW / 2 MWh usable, 85 % round trip
rows = list(csv.DictReader(open("examples/ERCOT_HB_WEST_2024.csv")))
path = {k: np.array([float(r[k]) for r in rows]) for k in ("rt", "da", "df", "T")}
path |= {k: np.array([int(r[k]) for r in rows]) for k in ("hour", "day", "dow", "doy")}
ch, dis, soc, V = bp.run_policy(net, path, bat, bat.c_lin(path["T"]))
res = bp.settle(bat, path["rt"], ch, dis, 0.0, path["T"])
print(res["net"] / 1000, "$/kW net of wear;", V[0], "$/MWh at 0/25/50/75/100 % full")
```

## The exam

Share of perfect-hindsight revenue net of wear (each node-year and battery's net revenue divided by an annual LP on the same RT prices, then averaged), on real prices, run once with the released network. Every strategy settles at the hourly RT price with rainflow wear at the BA's temperature. DA optimiser = a rolling LP that re-plans at 14:00 with day-ahead prices as the forecast; spread rule and threshold rule are GridData's baselines (the threshold uses the node's dev-period RT quartiles). Revenue intervals come from a weekly block bootstrap (2,000 draws). Re-scored in this repository from the shipped weights on shared/grid/data (notebook). Every number here is on real data that never chose anything.

| Metric | BatteryPilot | Baselines | |
|---|---|---|---|
| Share of perfect hindsight, ERCOT 2023-25, all batteries (36 node-years × 1/2/4 h)<br><sub>Beats the spread rule in 89 % of node-year-battery cases, the threshold rule in all, the DA optimiser in 57 %. About 244 cycles a year vs 300 for the optimiser.</sub> | **66.7%** | Rolling day-ahead LP optimiser (classical): 68.5%<br>Daily spread rule (GridData) (classical): 56.1%<br>Threshold rule, dev RT quartiles (GridData) (dumb baseline): 11.4% | ❌ loses |
| Share of perfect hindsight, CAISO 2024-25, all batteries (12 node-years × 1/2/4 h)<br><sub>Beats the spread rule in 81 % of cases, the threshold rule in all, the DA optimiser in 19 %.</sub> | **64.2%** | Rolling day-ahead LP optimiser (classical): 68.1%<br>Daily spread rule (GridData) (classical): 58.9%<br>Threshold rule, dev RT quartiles (GridData) (dumb baseline): 15.8% | ❌ loses |
| Share of perfect hindsight, ERCOT, 1 h battery (36 node-years)<br><sub>Beats the optimiser in 25 % of node-years. ERCOT 2023 1 h: −$9.4/kW-yr vs the optimiser, 95 % CI [−18.2, −1.6].</sub> | **55.9%** | Rolling day-ahead LP optimiser (classical): 62.9%<br>Daily spread rule (GridData) (classical): 53.1%<br>Threshold rule (GridData) (dumb baseline): 1.8% | ❌ loses |
| Share of perfect hindsight, ERCOT, 2 h battery (36 node-years)<br><sub>Beats the optimiser in 67 % of node-years but loses on the mean; every year's $ difference interval includes zero.</sub> | **67.1%** | Rolling day-ahead LP optimiser (classical): 68.4%<br>Daily spread rule (GridData) (classical): 57%<br>Threshold rule (GridData) (dumb baseline): 10.9% | ❌ loses |
| Share of perfect hindsight, ERCOT, 4 h battery (36 node-years)<br><sub>Beats the optimiser in 81 % of node-years. ERCOT 2024 4 h: +$3.3/kW-yr, 95 % CI [+1.1, +5.4]; 2023 +$9.0 [−3.0, +21.5]; 2025 −$0.9 [−5.9, +2.9].</sub> | **76.9%** | Rolling day-ahead LP optimiser (classical): 74.2%<br>Daily spread rule (GridData) (classical): 58.3%<br>Threshold rule (GridData) (dumb baseline): 21.6% | ✅ wins |
| Share of perfect hindsight, CAISO, 1 h battery (12 node-years)<br><sub>Loses to the optimiser in all 12 node-years; 2024 1 h: −$1.5/kW-yr, 95 % CI [−2.2, −0.8].</sub> | **56.3%** | Rolling day-ahead LP optimiser (classical): 64.1%<br>Daily spread rule (GridData) (classical): 55.2%<br>Threshold rule (GridData) (dumb baseline): 10.4% | ❌ loses |
| Share of perfect hindsight, CAISO, 2 h battery (12 node-years)<br><sub>Beats the optimiser in 1 of 12 node-years.</sub> | **63.4%** | Rolling day-ahead LP optimiser (classical): 68%<br>Daily spread rule (GridData) (classical): 59.2%<br>Threshold rule (GridData) (dumb baseline): 16.6% | ❌ loses |
| Share of perfect hindsight, CAISO, 4 h battery (12 node-years)<br><sub>A thin win (+0.7 points): beats the optimiser in 6 of 12 node-years, and both years' $ difference intervals include zero (2024 −$1.1 [−3.2, +0.7]; 2025 +$0.9 [−0.5, +2.4]).</sub> | **73%** | Rolling day-ahead LP optimiser (classical): 72.3%<br>Daily spread rule (GridData) (classical): 62.3%<br>Threshold rule (GridData) (dumb baseline): 20.3% | ✅ wins |
| Comparison only: the same network trained on real ERCOT 2018/2020/2022 node prices (breaks the physics-only rule; not released)<br><sub>What the simulator costs in ERCOT (−2.3 points) and gains in CAISO (+7.8 points), where no real training prices exist.</sub> | **ERCOT 69.0, CAISO 56.4%** | BatteryPilot (simulator-trained, released) (classical): ERCOT 66.7, CAISO 64.2% | ℹ️ |

### Where it wins

- Beats GridData's daily-spread rule (ERCOT 0.667 vs 0.561 of perfect hindsight; CAISO 0.642 vs 0.589) in 89 % / 81 % of node-year-battery cases, and the threshold rule in every case.
- 4 h batteries: ERCOT 0.769 vs 0.742 for the rolling day-ahead optimiser (beats it in 81 % of node-years; ERCOT 2024 +$3.3/kW-yr, CI [+1.1, +5.4]); CAISO 0.730 vs 0.723.
- Cycles less: about 245 cycles a year vs 300 (ERCOT) and 282 (CAISO) for the optimiser; its worst month at the median node was never below −$0.17/kW (any node: −$0.21/kW, CAISO December).
- Trained without a single real price path: real dev years only calibrated the simulator and gated rungs.

### Where it loses

- Loses to the classical rolling day-ahead optimiser on the mean: ERCOT 0.667 vs 0.685, CAISO 0.642 vs 0.681.
- 1 h batteries lose clearly: ERCOT 0.559 vs 0.629 (ERCOT 2023: −$9.4/kW-yr, CI [−18.2, −1.6]); CAISO 0.563 vs 0.641, every node-year.
- CAISO 1 h and 2 h lose to the optimiser in 23 of 24 node-years; the CAISO-like simulator family is hand-set from 5 dev months.
- Real spikes are less predictable from day-ahead prices than simulated ones: on twins of dev 2019 it captures 91-94 %, on real 2019 80 %; Uri twins 64-68 % vs real 38 %.

## Limits

- Energy arbitrage only: hourly settlement (mean of the 15-min RT), no ancillary services, no 5-min prices, price taker; copper-plate simulator (no congestion, nodal differences untrained).
- The CAISO-like family is hand-set from 5 dev months (Aug-Dec 2023) and is more volatile than real CAISO.
- Calibration hit its grid edges (μ, σ, cold f_max). Simulated day-ahead runs 10-20 % above RT (real 1-5 %).
- Wear parameters are assumptions from the literature; calendar ageing is not charged.
- Possible mild exam leak: GridSim's outage-rate defaults cite Texas RE's 2024/2025 reliability-assessment aggregates, published during the exam years. Not fixed in this release; the platform plan is to replace them with pre-2023 NERC GADS values and re-score.
- No error bar on the value curve itself; the exam's intervals are on revenue (weekly block bootstrap), so the recipe's coverage step does not apply.

## The ratchet

**Objective:** Capture = net revenue / perfect-foresight net revenue (net of wear), summed over the 1, 2 and 4 h batteries of each simulated path-year and averaged over the path-years of the frozen simulated set (each path-year counts the same)  
**Rule:** Keep a rung if its frozen capture beats the incumbent's by more than 0.01 AND its capture on the 10 real dev node-years (ERCOT 2019 and 2021 at 4 hubs, CAISO Aug-Dec 2023 at NP15 and SP15; all 3 durations) is not more than 0.01 below the incumbent's. When the simulator changed, a new frozen set was built and every logged rung re-scored and re-decided in order.

| Rung | Change | Frozen-set score | Decision | Why | v1 frozen | v2 frozen | v3 frozen | Dev capture |
|---|---|---|---|---|---|---|---|---|
| R0 | Classical rolling-horizon LP on day-ahead prices (re-plans at 14:00, follows the plan) | 0.102 | ● kept | Starting incumbent. | 0.1018 | 0.4894 | 0.4472 | 0.5927 |
| R1 | MLP value-curve policy trained by policy search on sim v1 (GridSim merit-order prices, no scarcity curve, no outage weather, no RT noise); 45 path-years | 0.593 | ○ rejected | Frozen +0.491 but the dev gate failed: dev capture 0.483 vs the incumbent's 0.593 (−0.110). The v1 prices have no scarcity, so the policy cycles on small spreads and is empty when spikes come. Still rejected on v2 (+0.043 frozen, dev −0.110) and v3. | 0.5926 | 0.5326 | 0.5696 | 0.4828 |
| R2 | Same network and policy search, trained on sim v2 (80 ERCOT path-years from 2018/2020/2022 inputs with dev-calibrated scarcity) | 0.767 | ● kept | Frozen 0.767 vs R0 0.489 on v2 (+0.278); dev 0.598 vs 0.593 → new incumbent. Re-scored on v3: 0.7465 vs R0 0.4472, still kept. | — | 0.7671 | 0.7465 | 0.5979 |
| R3 | Train on all 120 ERCOT path-years + 45 CAISO-like path-years (CISO fleet, duck-curve solar, carbon price, $1,000-2,000 caps), 2,500 steps | 0.749 | ○ rejected | Frozen 0.7489 vs R2 0.7465 on v3 (+0.0024, below the 0.01 margin), although dev improved (0.623 vs 0.598). | — | — | 0.7489 | 0.6228 |
| R4 | Fine-tune R3 on the same 165 path-years with longer windows (21 days), batch 64, lower learning rate (900 steps) | 0.758 | **◆ released** | Frozen 0.7580 vs R2 0.7465 on v3 (+0.0115 > 0.01); dev 0.637 vs 0.598. Released as v1.0. | — | — | 0.758 | 0.6372 |

<sub>Columns after *Why*: Frozen capture on each version of the frozen set (re-scores from training/ratchet.json) and dev capture on the real dev node-years.</sub>

v1 = GridSim merit-order prices only; v2 = + dev-calibrated ORDC scarcity, cold outages, gas-turbine derate, RT noise and DA expectation (24 ERCOT path-years); v3 = v2 + 9 CAISO-like path-years on the real CISO fleet. Each rung's sim score is on the set it was judged on. A comparison network trained on real ERCOT 2018/2020/2022 prices breaks the physics-only rule; it was never a rung and is reported only in the exam table.

Raw per-rung logs are in [`training/`](training/).

## Physics

Battery: 1 MW grid side, usable energy = P × duration (5-95 % of nameplate), 85 % round trip (√ each way), wear by rainflow cycle counting (ASTM E1049) on a Wöhler depth curve (depth^1.3 / 6,000 per cycle) with LiFePO4 temperature and C-rate dependence (Wang et al. 2011) in an HVAC container, at $130/kWh nameplate. Prices: the shared GridSim fleet dispatch (unmodified) for the real ERCOT and CISO fleets of 2018, 2020 and 2022, wrapped by BatteryPilot's price simulator (training/pricesim.py) with a merit stack, scarcity and day-ahead prices.

**Simulator**

- GridSim (shared/grid, unmodified): real ERCOT / CISO fleets of 2018, 2020, 2022 (heat-rate curves, EIA-923 fuel prices + Henry Hub daily, Markov forced and planned outages, the dev-2019 coal offer adder)
- Merit-order price from pmin blocks and 3 heat-rate segments per cluster; slow units needed for the day's peak stay at pmin (80 %); below that, wind and solar set a negative price
- Gas-turbine ambient derate (0.6 %/K CT, 0.4 %/K CC around the 35 °C rating); cold-weather outages f(T) = 0.55 Φ((−3 °C − T)/3 K) persisting with a 36 h time constant, fitted on Uri (dev 2021)
- ERCOT-style ORDC adder (VOLL − λ) × LOLP(R − X), μ −4,000 MW, σ 5,000 MW, fitted on dev 2019 + 2021 without Uri; a storage fleet that flattens the daily net load
- Real-time noise (Student-t, AR(1), scaled by net-load ramps); day-ahead price = forecast inputs (real EIA-930 demand-forecast errors, AR(1) wind/solar errors, outages known at noon the day before) + the expected ORDC adder
- Scenario draws: demand ×0.97-1.22, wind ×0.8-1.6, solar up to 45 % of peak, storage 0-20 % of peak, gas ×0.6-1.8, retirements 0-8 %, caps $5,000/$9,000; a CAISO-like family on the CISO fleet (solar 35-80 % of peak, imports ×0.4-0.95, carbon $10-40/t, caps $1,000/$2,000)
- Banks: 120 ERCOT + 45 CAISO-like simulated path-years for training; 24 + 9 frozen

**Checked against independent references**

- GridData's perfect-foresight, daily-spread and threshold baselines reproduced on 89 dev days at two hubs to 1e-13 $ (one spread day differs by $1.65 at HB_WEST: a price tie) (training/checks/battery_checks.json)
- Rainflow counting matches the rainflow package exactly on 200 random series; executor energy balance 1.1e-16 MWh; rainflow wear of a summer-2019 PF schedule $2,414 vs the linear proxy $2,416
- Uri 2021 (dev): 180 and 166 simulated hours above $1,000 for two outage draws vs 153 real; 92-95 % of the real ones covered (training/checks/calibration.json)
- CAISO-like family vs real Aug-Dec 2023 (dev): hourly medians $18-25 midday and $59-61 evening vs real $21-25 and $66-72
- Twin test: the released policy captures 91-94 % of perfect foresight on simulated twins of dev 2019 but 80 % on real 2019; Uri 2021 twins 64-68 % vs real 38 % (training/checks/twin_r4_v3.json)

## Help wanted

- CAISO OASIS serves only about 3 years (data from 2023-08): an archive of CAISO hub and DLAP prices for 2018-2022 would give the CAISO family real dev years and a longer exam.
- Pre-2023 ERCOT forced-outage rates by technology (NERC GADS or Texas RE) to replace GridSim's 2024/2025 aggregates.
- ERCOT and CAISO ancillary-service clearing prices, and ERCOT resource-node prices in West Texas, to add ancillary services and a congestion basis.

## Next rungs

- Make simulated real-time spikes less predictable from day-ahead prices (unforecast outage events, 15-minute ramp scarcity), fitted until the dev twin capture matches the real one (91-94 % twin vs 80 % real).
- A 1 h specialist or a 15-minute execution head: BatteryPilot loses to the day-ahead optimiser on 1 h batteries and in CAISO.
- Nodal basis: a West Texas congestion model; a real CAISO import and hydro supply curve; ancillary services.
- Re-train and re-score once GridSim's outage-rate defaults use pre-2023 NERC GADS values (removes the possible exam leak).
- Upstream to GridSim behind flags: the ORDC adder, the cold-outage curve, the gas-turbine ambient derate, the day-ahead price generator and the single-LP perfect-foresight baseline.

## Files

| Path | What |
|---|---|
| `batterypilot.py` | Inference — numpy/scipy only |
| `batterypilot_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [ERCOT historical DAM / RTM settlement point prices (report types 13060, 13061), via GridData](https://www.ercot.com/mp/data-products) | dev (calibration and gate, 2019 and 2021), exam (2023-2025, 4 hubs + 8 load zones) | Public ERCOT market data (cite ERCOT) |
| [CAISO OASIS LMPs (PRC_LMP DAM, PRC_RTPD_LMP), via GridData](http://oasis.caiso.com) | dev (Aug-Dec 2023, CAISO-like family check and gate), exam (2024-2025, 3 trading hubs + 3 DLAPs) | Public CAISO data (cite CAISO) |
| [EIA-930 Hourly Electric Grid Monitor (demand, day-ahead demand forecast, generation by fuel)](https://www.eia.gov/electricity/gridmonitor/) | physics (simulator inputs and forecast errors, train-input years), model input (demand forecast) | Public domain (U.S. Government) |
| [NASA POWER hourly temperature (BA population-weighted)](https://power.larc.nasa.gov) | physics (simulator weather, wear temperature), model input | Free use; cite NASA LaRC POWER Project |
| [EIA-860 / EIA-923 generator fleets and fuel costs, via PUDL (through GridSim)](https://catalyst.coop/pudl/) | physics (fleets, heat rates, fuel prices) | CC BY 4.0 (PUDL); EIA public domain |
| [GridSim outage-rate defaults (Texas RE reliability-assessment aggregates for 2024/2025)](https://www.texasre.org) | physics (forced-outage rates) — published during the exam years; a possible mild exam leak (limits) | Public reports (cited by GridSim) |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
