# GridSim: known limits

What the simulator gets wrong, by how much where we measured it, and what to watch in Phase 2. Numbers are from
`grid/checks/*.json`.

## Dispatch (ERCOT checks against EIA-930, CAMPD and ERCOT prices)

Configurations: `economic` (pure data-driven costs: EIA-923 heat rates and fuel costs, Henry Hub daily, ERCOT GADS
availability, observed nuclear), `calibrated` (+ coal self-commitment and a coal offer adder of −$6.62/MWh, fitted on
the dev year 2019 only), `merit` (hourly merit order). 2022 was never used for fitting.

A. **Coal versus gas is the main dispatch error, and it flips sign with the gas price.** Annual coal error:
   2019 (gas ~$2.3/MMBtu) economic −43 %, calibrated +6 %; 2022 (gas ~$6.2) economic +23 %, calibrated +28 %.
   Gas mirrors it (2019 +19 % / −2 %; 2022 −10 % / −11 %). Real ERCOT coal output barely moved between those years
   (78 → 72 TWh) while relative fuel costs flipped. Real coal therefore follows things the model does not have: unit
   self-commitment, captive lignite mines and contracts in cheap-gas years, and coal conservation in 2022 (rail and
   supply shortages). **The 2019 calibration does not transfer to 2022.** Treat coal/gas splits as uncertain to about
   ±25-45 % per year. Hourly shape is good for gas (hourly r 0.97-0.99) but weaker for coal (r 0.5-0.8).
B. **CO2 is better than the fuel split** because coal and gas errors partly cancel: −15 % (2019 economic), −1 %
   (2019 calibrated), +0.3 % / +1.8 % (2022) vs CAMPD for ERCOT plants; hourly r 0.93-0.98. CAMPD covers fossil units
   ≥ 25 MW; small engines and biomass are excluded from both sides.
C. **Prices: shape right, scarcity wrong.** Median price $18.6 vs $20.5 (2019) and $46.6 vs $48.0 (2022), median
   absolute hourly error $2.6 / $7.0, rank correlation 0.87-0.88, monthly means r 0.9 (2022). Hourly Pearson r is low
   (0.1-0.5) because scarcity spikes happen at the wrong hours: the sim has 61-99 hours above $200 vs 119-238 real.
   There is no ERCOT ORDC (only a 3-block reserve shortage curve), no forced outages at the real times, no bidding.
   The copper plate has no West Texas congestion. Mean prices are therefore too low ($30 vs $38 in 2019,
   $54 vs $64 in 2022).
D. **Outages are expected values** (seasonal derates from ERCOT GADS EAF), not the real outage timing. Real refuelling
   of nuclear is supplied as observed output in the checks; in pure-model runs nuclear is derated on average.
   **Note on data use:** the GADS availability figures came from Texas RE's 2024/2025 reports (published fleet-wide
   annual statistics, not hourly labels). 2023-2025 are exam years in `splits.json`, so this was a leak of
   exam-period information into every dispatch default. **Fixed in physics 1.1 (14 Nov 2026):** forced shares from PJM's
   2017-2021 class averages (May 2022), planned shares from Texas RE's 2017 assessment (`fleet.OUTAGE_DEFAULTS`);
   the dev calibration and ERCOT damping were re-fitted. The leaked values matched reality better: with clean
   defaults the fleet is more available, scarcity is rarer and the held-out 2022 coal error grows. v1.0 -> v1.1:

   | Check (ERCOT, calibrated unless stated) | v1.0 (leaked) | v1.1 (clean) | Real |
   |---|---|---|---|
   | Dev-2019 coal offer adder | −6.62 $/MWh | −5.73 $/MWh | — |
   | ERCOT effective damping (5 dev events) | 2.5 %/% | 2.75 %/% | — |
   | 2019 coal / gas / CO2 | +5.6 / −2.2 / −1.0 % | +5.9 / −2.2 / −2.7 % | — |
   | 2022 (held out) coal / gas / CO2 | +27.7 / −11.4 / +1.85 % | +34.4 / −14.0 / +2.51 % | — |
   | Hours above $200, 2019 / 2022 | 61 / 99 | 23 / 54 | 119 / 238 |
   | Median price 2022 | $46.6 | $45.0 | $48.0 |
   | Hours below 150 GW·s, 2019 / 2022 | 127 / 949 | 241 / 1,229 | 79 / 492 |
   | Days re-solved without history, 2019 / 2022 | 22 / 15 | 16 / 14 | — |
   | Economic config coal 2019 / 2022 | −42.6 / +23.1 % | −49.1 / +29.1 % | — |
E. **Inertia from the commitment** matches ERCOT's 2019 minimum (131 vs 134.6 GW s) but not 2022 (76 vs 115 GW s;
   949 vs 492 hours below 150 GW s). At low net load with lower GADS availability the MILP commits fewer synchronous
   units than ERCOT did. ERCOT also commits units for reliability (RUC) and reserves the model does not require.
   GridPulse states built from dispatched hours should be checked against ERCOT's reported inertia.
F. **Carried history sometimes makes a day infeasible.** In 4-11 % of days (15-40 a year) the first solve was
   infeasible and the day was re-solved without the carried minimum up/down and ramp history (`n_days_relaxed`). The
   cause was not isolated: it depends on the carried state and did not reproduce from another start date. Effect:
   minimum up/down times may be broken across those midnights. Fix: soft history constraints with a penalty slack.
   *14 Nov 2026: done, behind `dispatch(..., soft_history=True)` (off by default).* A penalised slack (10 x VOLL x
   unit size per unit-hour on min up/down, 10 x VOLL per MW on the first-hour ramp) replaces the re-solve. ERCOT 2022,
   calibrated, physics 1.1: days re-solved without history 14 -> 0; coal +34.36 -> +34.34 %, gas -14.02 -> -14.01 %,
   median price $45.04 -> $45.04, CO2 +2.51 -> +2.49 % (`checks/ercot_dispatch_2022.json`, `calibrated_soft`).
   Random-outage unit-level runs (`checks/check_soft_history.py`, 5 seeds x 5 days) needed no slack and matched the
   hard runs, so the infeasibility PlantDetective's build met with `outages='random'` did not reproduce there.
G. **Clustering**: similar units share one integer commitment (≤ 50 units per cluster integer, larger clusters
   continuous); unit-level results need `cluster=False` (slower). Combined cycles are blocks (no 1×1 vs 2×1 modes);
   no duct firing; no start-up types (hot/warm/cold). Hydro is a daily energy budget.
H. **No network in the ERCOT checks**, no imports optimisation (DC ties are fixed at their real values), and storage
   is one aggregate (ERCOT 2019/2022 had little storage; the 2023+ battery fleet is not validated).
I. **Real-time scarcity the day-ahead market cannot see (physics 1.2, opt-in, 15 Nov 2026).** GridSim prices one
   market. Real ERCOT spikes mostly come in real time and are not in the day-ahead price: on the training-input years
   the DA price covers 24-30 % of the RT value of hours above $300/MWh (HB_HUBAVG; 2019 31 %, 2021 without Uri 13 %).
   BatteryPilot's wrapper had DA prices that covered 44-84 % (median 71 %) in its simulated ERCOT frozen years, so a
   DA-anchored policy looked worse in simulation than on real prices. `gridsim.market` adds the missing part as a
   point process on the RT - DA spread (README, "Physics versions"), fitted on ERCOT 2018, 2020 and 2022 prices only
   (`checks/fit_rt_scarcity.py`: 169 events, 333 event hours) and checked on dev (`checks/rt_scarcity.json`, 200
   draws a year driven by the year's real demand-forecast misses and DA prices).

   **Three fits, all on the training years; the third is the default.** (a) `ERCO_first_fit` placed events by clock
   hour, month and forecast miss. BatteryPilot's twin test on dev 2019 then showed what it missed: in real 2019 the
   rolling DA optimiser, discharging in the DA's dearest hours, earned $47.8k/MW from hours with RT - DA > $100; on
   twins with this fit only $9.6-11.0k, and its capture fell to 0.46-0.50 against 0.79 real. Real unforecast spikes
   fall in the hours the DA already marks as tight: on the training years the start rate is 29-33 per 1,000 h in each
   day's dearest 5 % of hours against 0-0.5 in its cheapest quarter. (b) `ERCO_shape_only` replaces the clock-hour
   profile by the DA price's rank within the day and its level against the past week (gamma 2.28 and 1.32; AIC
   1,712 vs 1,879 for (a) and 1,745 with both). The same twin test still gave the optimiser only $9.3-10.7k: real
   spikes are also larger on days the DA prices high (mean excess over $100 $117 in the lowest quartile of DA level
   among event hours, $756 in the highest). (c) `ERCO` adds that size term: the generalised Pareto scale is
   $60.7 x exp(1.11 level) (xi 0.45; AIC 4,331 vs 4,404 without it). The DA knows when the system is tight and
   roughly how tight, not whether the spike comes.

   | ERCOT HB_HUBAVG, default fit (c) | 2018 | 2020 | 2022 | 2019 (dev) | 2021 (dev) | 2021 without Uri |
   |---|---|---|---|---|---|---|
   | Hours RT - DA > $100, real | 87 | 81 | 165 | 108 | 224 | 99 |
   | sim mean [5-95 %] | 112 [85-140] | 125 [99-155] | 106 [81-133] | 117 [85-148] | 106 [80-136] | 104 [80-132] |
   | Hours RT - DA > $1,000, real / sim | 6 / 6.4 | 1 / 7.9 | 23 / 8.5 | 17 / 11.5 | 63 / 7.1 | 7 / 5.8 |
   | Events, real / sim | 51 / 55 | 44 / 62 | 74 / 52 | 56 / 58 | 75 / 52 | 61 / 52 |
   | $ of RT - DA in event hours (k$/MW), real / sim | 31 / 39 | 22 / 47 | 95 / 44 | 82 / 56 | 192 / 41 | 36 / 36 |
   | Share of that $ in the day's 3 dearest DA hours, real / sim (first fit) | 0.73 / 0.54 (0.23) | 0.66 / 0.56 (0.24) | 0.53 / 0.50 (0.21) | 0.86 / 0.57 (0.21) | 0.18 / 0.45 (0.21) | 0.62 / 0.51 (0.22) |
   | Share of event hours in those 3 hours, real / sim | 0.56 / 0.39 | 0.58 / 0.41 | 0.46 / 0.37 | 0.67 / 0.41 | 0.32 / 0.35 | 0.57 / 0.37 |
   | Hour-of-day profile corr (first fit) | 0.82 (0.82) | 0.89 (0.89) | 0.79 (0.95) | 0.82 (0.80) | 0.88 (0.87) | 0.89 (0.87) |
   | Month profile corr | 0.23 | 0.78 | 0.60 | 0.94 | 0.86 | 0.72 |

   What it gets wrong: (1) **years differ more than the draws do.** The rate knows the DA shape, month and forecast
   miss, each normalised within the year, not how tight the year is: real years range 81-165 hours (dev 2021 with Uri
   224) against a draw range of about 80-155, and the $ size ranges 22-192 k$/MW against 36-56 k$/MW. A caller that
   wants year-to-year spread must scale the rate per scenario (`scale=`). (2) **Uri is not in it**: 125 of 2021's 224
   hours come from 8-24 Feb; cold-weather outages belong to the caller's fleet model (BatteryPilot's cold-outage
   curve). (3) Still too little of the spike $ in the DA's dearest hours (0.50-0.57 vs 0.53-0.86 real outside Uri)
   and too few events in the afternoon (0.53-0.70 of event hours vs 0.68-0.82). (4) The forecast-miss driver is the
   demand miss only (EIA-930 has no wind or solar forecasts): beta 0.57 per standard deviation; real start rates
   0.6-2.9 vs 8.6-13.7 per 1,000 h in the bottom vs top fifth of misses on the training years (2022 is flat above the
   bottom fifth). (5) Upward spikes only: RT below DA by more than $50 (60-430 h a year) is left to the caller's
   noise. (6) The geometric duration (mean 1.97 h) has too many 2-hour events (25 % vs 16 % real); the top 1 % of
   event hours is too high in 2018 and 2020 ($2,594 and $2,836 vs $2,034 and $969) and too low in 2022 and 2019.
   (7) **No CAISO fit**: CAISO prices exist only from Aug 2023 (the OASIS history limit), so no training-year prices
   exist; `market` returns zeros for CISO, and CAISO users stay on 1.1 behaviour. With physics 1.0 or 1.1 `market`
   returns zeros, and 1.2 leaves every other output unchanged (same outage defaults and calibration file; a
   merit-order ERCOT week is identical: `rt_scarcity.json`, `identity`).

## Frequency

1. **One aggregated frequency per area.** No inter-machine oscillations, no voltage dynamics, no network. Against a
   full 4-machine ANDES simulation (Kundur two-area, 200 MW step) the nadir is 24 mHz shallower (0.400 vs 0.423 Hz
   deviation, 6 %) and the RMS difference is 15 mHz: network losses and voltage effects are missing. Multi-area runs
   have tie lines but no line limits or inter-area HVDC control.
2. **Load damping is a fitted, effective parameter in both systems.** No citable GB value was found (literature
   1-2.5 %/Hz). GB: 1.5 %/Hz was fitted on the first nadir of the 9 Aug 2019 dev event (49.11 vs 49.1 Hz). The rest
   of that event then follows the NESO 1 s trace up to LFDD (LFDD at 75.6 s vs 75.9 s, minimum within 3 %). After
   LFDD the simulation recovers too fast (49.6 vs ~48.95 Hz at 100 s): the 931 MW of LFDD is gross demand, and
   embedded generation on the shed feeders cut its net effect. The event is a knife edge: whether the 350 MW of RoCoF
   relays trip decides the outcome (initial RoCoF 0.135 Hz/s vs the 0.125 setting); inertia +10 % or damping
   2.5 %/Hz avoid both the RoCoF trips and LFDD.
3. **ERCOT primary response needed an effective damping of 2.5 %/% (4.2 %/Hz).** With BAL-001-TRE-1 deadbands
   (0.034 Hz steam/hydro, 0.0166 Hz others) and textbook load damping (1 %/%), the nadirs of the 5 ERCOT dev events
   were 41-70 % too deep, even though the dispatch-derived inertia of those hours was within −10/+11 % of ERCOT's
   reported values. One damping value fitted on those 5 events brings them to −15..+16 % (MAE 9 %, in-sample); the
   fitted value depends on the dispatch calibration (3.75 with the earlier generic-availability dispatch, 2.5 now). It
   stands for all fast response the model lacks (motor load, fast CT and inverter response) and is not physical load
   damping. No held-out ERCOT event was used (the exam events are untouched), so GridPulse's exam is the first real
   test. GB dev events not used for fitting (the frozen dev list is 9 Aug, 1 Jul, 11 Jul, 3 Dec 2019): 1 Jul −34 % and
   11 Jul −27 % (too deep, with each event's own inertia and loss but 9 Aug's response-holding mix: their real holdings
   are not in the data). 3 Dec is unusable: GridData's loss estimate (1,788 MW) is about twice what the measured
   0.29 Hz drop at 333 GVA s implies (~850 MW). The model gave −33 % with GridData's earlier 855 MW estimate. GB nadirs
   are probably too deep by ~30 % outside the fitted event: GB fast response is stronger than the 9 Aug mix.
   *14 Nov 2026: the 3 Dec label is corrected and the event is usable.* The 1,788 MW was 854.7 MW (Spalding CCGT
   T_SPLN-1, Elexon B1610 output in the half hour before) plus 933 MW read from Elexon FUELINST as an IFA
   interconnector trip; FUELINST shows an hour-boundary ramp (IFA 1,900 → 204 MW over 19:45-20:10 UTC), not a trip.
   GridData now gives 854.7 MW (`data/sources/gb_event_label_corrections.csv`, applied by `gd.events` and by
   `check_frequency_dev_events.py`). Re-run of the GB part of that check with physics 1.0 (scratch copy, not
   `checks/*.json`): 3 Dec nadir error −316 % → **−33 %** (simulated drop 1.21 → 0.39 Hz, measured 0.29 Hz; RMS
   0-60 s 552 → 70 mHz); GB dev summary (4 events) mean error −94 % → −23 %, MAE 95 % → 24 %. 3 Dec now agrees with
   1 Jul (−34 %) and 11 Jul (−27 %): GB nadirs are too deep by ~30 % on all three REMIT-labelled dev events.
4. **Governor and turbine models are generic per technology** (IEEE PES-TR1 typical values): one valve lag, two
   lead-lags, a rate limit and a headroom clip. Missing: gas-turbine output falling with frequency and ambient
   temperature, boiler pressure effects on steam units beyond ~30 s, load controllers that withdraw response
   (Point C′ withdrawal seen in NERC data), nuclear response.
5. **Loads**: constant damping D × P × Δf; no motor stalling or voltage dependence. Embedded generation protection is
   given as relay groups (RoCoF, under-frequency); vector-shift trips must be entered as part of the loss.
6. **Secondary control and operator actions** are a single first-order MW ramp; there is no AGC with ACE, no reserve
   deployment logic, no market re-dispatch. Results after ~60 s are indicative only.
7. **Inertia from dispatch** counts committed units × H × MVA (MVA = nameplate / 0.9 where unknown) with H from
   technology defaults or GridData's fleet; synchronous condensers and demand-side inertia are not included unless given
   as `H_extra_MWs`.

## Network

8. Power flow and DC-OPF match PYPOWER to 1e-10 and the network dispatch matches the DC-OPF exactly, but there is no
   AC-OPF, no N-1 security constraints, no losses in the DC dispatch, and no real ERCOT network. The ERCOT checks
   are copper-plate, so congestion prices (West Texas, Panhandle) are absent from simulated prices.
9. pandapower's MATPOWER converter turns ACTIVSg transformers into impedances (losing taps) and fails on some IEEE
   cases under numpy error checking; PYPOWER was used as the main reference and pandapower only where its conversion
   is faithful (case9, case30, case118).

## Renewables and demand

10. **Wind from reanalysis is too high by 6-8 %** (ERCOT 2019 +5.9 %, 2022 +7.8 %; hourly correlation 0.90-0.91,
    hourly nMAE 19-20 %). NASA POWER (MERRA-2, ~50 km) smooths wind; curtailment and plant outages are not modelled
    (ERCOT curtails a few % a year). One scale factor fitted on 2019 (0.945) brings 2022 to +1.9 %. Specific power
    comes from the online year, not the real turbine model.
11. **Solar** correlates well hourly (0.97-0.98) but 2019 is +20 % (a small fleet still being built; online dates
    are month-level and new plants ramp up) while 2022 is +3.6 %. The 2019 scale factor does not transfer (2022 −13 %);
    use raw output. No soiling, snow, curtailment or module degradation.
12. **Demand from weather**: MAPE 3.8 % (2021, Uri week excluded) and 4.3 % (2022) with actual weather, worse than
    ERCOT's own day-ahead forecast (2.4-2.8 %) which uses forecast weather. The trend term does not capture 2022 load
    growth (−3.4 % bias). Behind-the-meter PV is subtracted only if its capacity is given. Fitted per BA; it is a
    statistical-physical hybrid, not a building-stock model.

## Generator (Phase 2 training data)

13. Generated systems are drawn from technology ranges, not real BAs, so their mix and commitment are plausible but
    not representative of any one system. Generated inertia per MW of demand: median 4.6 s (ERCOT 2019 minimum 4.6 s,
    GB 9 Aug 2019 7.2 s), 5 % below 2.1 s on purpose (future low-inertia cases). 13 % (60 Hz) and 5 % (50 Hz) of random
    events trigger UFLS
    (labels say so); real systems are operated to avoid that for the largest loss.
14. Governors in generated states inherit limit 3: their aggregate response is plausible by construction but has not
    been calibrated on measured events.

## Speed and robustness

15. MILP dispatch of a full ERCOT year takes 13-21 min wall single thread on this shared machine (solver ~1.1 s per
    day, ~80 clusters, 36-hour window, plus model building); LP solves in ~0.43 s per day; merit order ~2 s a year. The ≤ 10 min target is met by LP on an idle core but
    not by MILP here. Days that become infeasible through carried history (e.g. a retirement inside a minimum up
    time) are re-solved without it and counted (see F).
