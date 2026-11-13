# RoadSim: known limits

What the simulator gets wrong, by how much where we measured it, and what to watch in Phase 2. Numbers are from
`road/checks/*.json` (held-out EPA vehicles unless stated).

## Accuracy against real test results

1. **Conventional city vs highway bias.** FTP-75 is over-predicted (median +5.6 %, IQR −4.2 to +8.0 %, range −10.9 to
   +21.4 %) and HWFET under-predicted (median −4.3 %, IQR −5.5 to −3.0 %). The spread comes mostly from using one of
   ten benchmark engine maps for every engine (a 2023 Mazda3 gets the Toyota A25A map and is +13 % on both cycles), and
   from generic transmission losses. The city/highway tilt suggests low-load losses (idle, light-load engine efficiency,
   converter slip) are still a bit low and highway driveline/aux losses a bit high.
2. **P2 parallel hybrids are too good in the city.** FTP +8 to +31 % (median +12 % held-out); HWFET and US06 are fine
   (−6 %, −5 %). The strategy is not the cause (DP, A-ECMS and the rule give the same FTP). Likely missing: torque
   converter in P2 automatics (F-150, Tacoma), engine-start / clutch-engagement losses, and production calibrations
   that keep the engine on for drivability. Treat P2 city results as optimistic.
3. **Honda-type series hybrids (e:HEV) are modelled as pure series** (no direct-drive clutch at highway speed): HWFET
   −9 %, US06 −10 %, FTP −4 %.
4. **Cold weather is too mild.** The 20 °F (−6.7 °C) cold FTP is over-predicted by +10 to +16 % (3 cars). The cold-start
   penalty was fitted at 24 °C (bag 1 vs bag 3) and extrapolates with an oil-viscosity law; it is not enough at −7 °C.
   Driveline oil, tyres on a cold road, and enrichment are lumped. Expect winter fuel use to be underestimated; this is
   the first thing to calibrate on Phase 2 dev trips (VED has Ann Arbor winters).
5. **A/C on conventional cars is too cheap.** SC03 (35 °C, sun, A/C on) +9 to +15 % for conventional cars (3), but
   −5 % (−13 to +14 %) for hybrids. The belt-driven compressor/condenser-fan load and the hot-soak pull-down are likely
   underestimated.
6. **BEVs needed a loss-map scale of 0.47.** With the ALPHA generic/Bolt maps as published, BEV energy use came out
   7-16 % too high; one factor (fitted on 3 dev BEVs, with an assumed 0.88 wall-to-battery charging efficiency) fixes the
   median (held-out −1.7 %, range −7.2 to +6.0 %). Charger efficiency, 12 V load and motor losses are confounded in that
   factor. Tesla-class drive units are still ~2-7 % under-predicted.
7. **Hybrid motors use the generic IPM map**, not the measured 2010 Prius maps: the old Prius maps scaled up made modern
   hybrids 17-27 % too thirsty. This choice was made on the dev hybrids.

## Heavy duty (no per-vehicle real data with traces; published ranges only)

8. The Class 8 diesel reference truck gives 85.8 g CO2/ton-mile on the GEM cycles (6.25 mpg), above the MY2021-23
   standard of 75.7, as expected for a truck without Phase 2 technology packages. This is a plausibility check, not a
   validation. The heavy-duty diesel map is a Willans-line model (peak BTE ~44.5 %), not a measured map.
9. Battery-electric Class 8: 2.5 kWh/mi composite, above the ~1.5-2.2 kWh/mi reported for loaded BEV trucks
   (literature, not verified). Probably motor losses at light load on a 600 kW machine.
10. **Hybrid transit bus too good on CBD14** (+76 % vs the diesel bus; literature 10-40 %; +35 % on the HD transient
    cycle). BEV bus 1.66 kWh/mi vs 1.8-2.15 measured in service at Foothill Transit (their routes have more stops,
    dwell and HVAC). The diesel bus (4.6 mpg CBD14) is in the expected range. Bus accessory loads are guesses (4-6 kW).
11. The CBD14 bus cycle was constructed from the SAE J1376 description (Manhattan/OCTA traces were not reachable).

## Model structure

12. **1 Hz.** Shifts, engine starts, clutch slips and tip-ins happen inside a second; their energy is lumped.
    Instantaneous fuel rate and engine speed are smoother than an OBD log.
13. **Rotating inertia**: engine + transmission input inertia acts in the engaged gear only; energy released at an
    upshift is booked as driveline loss. Wheel/axle inertia is a constant mass fraction.
14. **No wheel slip, no load transfer, no tyre force model**: the stability limit on regen is a fixed deceleration.
    Snow/gravel only raise rolling resistance.
15. **Driver**: perfect anticipation of the next second plus a PI correction. Real drivers lag, overshoot and vary; use
    `rng` for some variety. Trace misses only happen at the powertrain's limits.
16. **Aerodynamics**: steady wind per second, quadratic yaw correction, no gusts, no drafting, no altitude effect beyond
    air density.
17. **Battery**: one RC branch, no aging, no SOC-dependent capacity fade, lumped temperature. Power limits are simple
    voltage/temperature rules.
18. **HVAC**: a one-node cabin with fixed set-points and on/off thresholds; no humidity/defrost, no passenger-door
    losses (important for buses), no seat heaters.
19. **Hybrid controls are generic**: A-ECMS stands in for each maker's calibration. DP and MPC see the future trace
    (benchmarks, not causal). DP ignores minimum on/off times and uses a coarse grid (121 SOC x 21 controls), so a
    causal strategy can occasionally beat it by ~1 %, mostly when it ends with more charge than DP (the fixed
    SOC-correction factor over-credits stored energy).
20. **Engine maps** are resized at constant BMEP from ten ALPHA engines (+ a Willans map for heavy duty); a specific
    engine's real efficiency can differ by ±5-10 %. `eff_scale` exists for that (the generator draws it ±3 %).
21. **Transmission losses** are generic (mesh efficiency + spin torque by type), not measured per transmission.
22. **Coast-down check is a consistency check**: the spec is built from the same EPA targets it reproduces (median
    force error 0.5-1.6 %, 92 % of vehicles inside the 3.1 % spread of repeat EPA targets); it verifies the road-load
    model, wind handling and fitting, not independent CdA/Crr knowledge.

## Generated vehicles

23. Light-duty vehicles bootstrap real EPA test-car records, so masses, road loads and power are realistic. Generated
    hybrids are ~20 % less efficient than the EPA hybrid set (bodies are drawn from all gasoline vehicles, including
    large trucks; real hybrids are mostly efficient cars/SUVs). Generated vans are slow (0-60 mph median 10 s vs ~7-8 s for
    minivans) because van donors include full-size Transit-type vans. Heavy-duty vehicles come from ranges, not real records.
24. PHEV and P2 hardware sizing are rules of thumb; there is no public table of motor/battery specs by model.

## Speed and robustness

25. Hybrids run at ~4,000x real time (candidate scoring in numpy); conventional/BEV at 40,000-70,000x. DP takes ~1 s
    per UDDS, MPC ~1 s. Ledger closure: median ~1e-15, worst 5e-4 over 600 stress runs (none over 0.5 %).
