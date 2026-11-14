# RoadSim: known limits

What the simulator gets wrong, by how much where we measured it, and what to watch. Since 13 Nov 2026 the default
physics is **roadsim-physics-v2** (calibrated on dev data only: [CALIBRATION.md](CALIBRATION.md)); physics v1 stays
reproducible with `options='roadsim-physics-v1'`, and the hybrid fix pass of 14 Nov 2026 runs with
`options='roadsim-physics-v2.1'` (not the default). Numbers are from `shared/vehicle/checks/*.json` (v1: plain names,
v2: `__roadsim-physics-v2`), computed once after v2 was frozen; held-out / exam data unless stated. Errors on EPA
cycles are on fuel economy (+ = the sim burns less); D3 and ZTBus errors are on energy (+ = the sim uses more).

## Held-out results, v1 -> v2 -> v2.1 (median, [IQR], n)

v1 and v2: computed once after the v2 freeze (13 Nov 2026). v2.1 (`roadsim-physics-v2.1`, the hybrid fix pass of 14 Nov
2026, not the default): a **second look** at the same sets, which had been seen before v2.1 was fitted, so they judge
nothing; the fresh judge of v2.1 is exam-2 below. v2.1 changes hybrid and plug-in runs only: every conventional, BEV,
bus and Class 8 number is identical to v2 ("= v2"). The D3 plug-in rows of v2 were recomputed with agent 3's harness
(charge-sustaining tests at the twin's CS level, scored on corrected fuel); v2 as first reported in brackets.

| check | v1 | v2 | v2.1 (second look) | target (V2_PLAN) |
|---|---|---|---|---|
| EPA held-out conventional FTP | +5.6 [-4.2, +7.9] | +0.4 [-7.7, +4.7] n 15 | = v2 | +-3, IQR halved: median met, IQR not (12.1 -> 12.4) |
| ... HWY | -4.3 [-5.5, -3.0] | -1.9 [-3.1, -1.5] n 15 | = v2 | met |
| ... US06 | +0.2 | -1.8 n 5 | = v2 | |
| ... SC03 / 20 F cold FTP | +11.4 / -6.0 | +1.9 / -5.5 n 2 / 2 | = v2 | +-6: met (n 2) |
| EPA held-out P2 FTP | +11.9 [+9.3, +23.7] | -4.0 [-15.1, +8.6] n 5 | -3.1 [-14.0, +8.9] | +-8: median met, spread not |
| ... HWY / US06 / SC03 | -6.0 / -5.1 / -6.1 | -9.7 / -23.6 / -17.7 | -9.7 / -17.6 / -19.8 | |
| ... 20 F cold FTP | +7.1 | +2.0 n 3 | +0.3 | |
| EPA held-out power-split FTP / HWY / US06 | +1.0 / -2.4 / -3.6 | -5.1 / -7.2 / -6.6 n 4 | -5.6 / -8.9 / -6.9 | v2 worse; v2.1 slightly worse again |
| ... 20 F cold FTP | -14.6 | -15.9 n 3 | -17.6 | |
| EPA held-out e:HEV (2024 CR-V) FTP / HWY | -4.0 / -9.5 | -2.0 / -3.8 n 1 | -2.6 / -4.0 | +-6: met |
| EPA held-out BEV UDDS / HWY | -0.5 / -1.7 | -0.4 / -1.8 n 8 | = v2 | |
| D3 exam conventional (all tests) | -5.1, IQR 12.2 | -3.2, IQR 10.9 n 240 | = v2 | |
| ... cold / hot / SC03 | -9.0 / -12.3 / -12.7 | -4.9 / -6.3 / -8.4 | = v2 | |
| D3 exam BEV | -3.5, IQR 19.8 | -2.0, IQR 14.3 n 62 | = v2 | +-10 per trip: median met |
| D3 exam hybrids (mostly 2010-2013 P0/P1/P2 and the C-Max Hybrid) | -20.2 | -17.8 n 111 | -17.8 (P2 -18.2, C-Max -7.8) | |
| D3 exam plug-in power-split (Prius Plug-ins, RAV4 Prime) | -14.9 | -17.8 [-16.2] n 71 | -17.8 | +-15: not met |
| D3 exam plug-in series (2012/13 Volt, i3 REx) | -22.7 | -22.6 [-27.1] n 97 | -22.6 | not met |
| D3 exam engine-on share sim / measured, P2 / plug-in series | | 0.46 / 0.80, 0.07 / 0.36 | 0.50 / 0.80, 0.09 / 0.36 | |
| ZTBus exam legs (bus twin energy per leg) | -13.8 [-24.2, -1.7] | -2.3 [-9.8, +4.9] n 5,847 | = v2 | |
| FASTSim cross-check, conventional UDDS / HWY / US06 | -0.3 / -3.1 / -3.0 | -1.9 / -1.2 / -6.4 | = v2 | (v2 adds physics FASTSim lacks) |
| ... hybrids UDDS / HWY / US06 | +18.0 / +6.7 / -4.3 | +12.5 / +2.3 / -8.3 | +11.2 / +1.9 / -8.2 | |
| Generated hybrids, hot UDDS / HWY median (EPA MY2022-24 hybrids: FTP 59.2, HWY 58.2 mpg) | 48.3 / - | 47.5 / 45.2 mpg | 47.1 / 45.1 mpg | not met |
| Ledger stress, 600 runs | max 5.1e-4 | max 5.6e-4, none > 0.5 % | = v2 | closes: met |
| Speed, UDDS single thread | conv 50-55k x, hybrids 3.8-4.8k x | conv 24-26k x, hybrids 3.2-3.9k x | hybrids 3.1-3.9k x (2-4 % slower) | v2 costs 2x on conventional/BEV |

## Exam-2 (fresh hybrid exam, frozen 14 Nov 2026 before the v2.1 fit; computed once after the v2.1 freeze)

`checks/exam2_hybrids.json` (sha256 fd180962…): data no check had looked at. Results: `checks/exam2_report.json`,
`hybrids_epa__exam2__<preset>.json`; the per-trip VED exam twins (`ved_twin_exam2__<preset>.json`) stay local and untracked (VED exam is the models' exam too).

**EPA test-car hybrids and plug-ins of held-out families, never simulated before** (409 records; FE %, + = the sim
burns less; median, n of the FTP column; "|err|" = median absolute error over every cycle of every record):

| group | n | FTP v2 -> v2.1 | HWY | US06 | SC03 | 20 F cold FTP | median abs err |
|---|---|---|---|---|---|---|---|
| P2 TMED (Hyundai / Kia) | 53 | -10.2 -> -9.8 | -8.1 -> -8.1 | -8.8 -> -6.3 | -12.8 -> -13.6 | -1.3 -> -2.6 | 9.1 -> 8.9 |
| P2 with converter (Ford, BMW, Mercedes, Toyota i-FORCE MAX) | 19 | +27.9 -> +26.1 | +1.5 -> +2.2 | +12.7 -> +12.6 (3) | +37.1 -> +38.3 (3) | +11.1 -> +9.1 (3) | 11.1 -> 9.4 |
| P2 with a clutch (Infiniti, Lexus Hybrid MAX) | 9 | +17.7 -> +15.4 | -2.3 -> -3.0 | | | | 4.3 -> 4.7 |
| mild P0/P1 (Honda IMA, GM BAS, BMW AH7) | 38 | +10.8 -> +8.9 | +11.5 -> +11.0 | +4.2 -> +5.3 (7) | | | 10.8 -> 9.8 |
| Toyota / Lexus power-split | 166 | +18.0 -> +17.2 | +7.1 -> +6.8 | +6.4 -> +5.7 | +18.1 -> +19.0 | -4.5 -> -5.5 | 9.4 -> 9.1 |
| Ford power-split | 10 | +9.0 -> +8.9 | +5.2 -> +4.7 | | | | 7.8 -> 7.3 |
| Honda e:HEV (2019+ Insight) | 4 | -2.9 -> -3.6 | -2.6 -> -3.6 | -2.1 -> -2.2 | +18.7 -> +18.3 | -12.8 -> -14.0 | 4.1 -> 5.0 |
| plug-in power-split, CS (Prius PHV / Prime, RAV4/NX/RX plug-ins, Ford, Pacifica) | 22 | +0.7 -> +0.5 | +2.7 -> +2.2 | +1.6 -> +0.4 | +11.2 -> +11.8 | -28.5 -> -27.3 | 5.3 -> 4.9 |
| plug-in series, CS (ELR, i3 REx, Karma, Clarity, Accord PHEV, Outlander) | 15 | +18.5 -> +19.4 | +5.0 -> +4.8 | +13.8 -> +15.3 | +44.8 -> +45.5 | +1.7 -> +1.5 | 14.9 -> 16.6 |
| plug-in P2, CS (BMW, Mercedes, Porsche, Hyundai/Kia, Jeep, Land Rover) | 67 | +6.9 -> +7.3 | -8.2 -> -8.1 | -8.6 -> -8.3 | +16.6 -> +14.7 | +5.9 -> +4.8 | 10.6 -> 10.8 |

Records of families no check had ever touched (n 296 of 409) tell the same story (Toyota FTP +20.0 -> +19.4, TMED
-8.3 -> -7.8, converter P2 +30.6 -> +30.1).

**VED exam HEV / PHEV twins** (108 cars, 11,157 trips; % on fuel (HEV) or fuel + battery energy (PHEV), + = the sim
uses more; engine-on gap in points):

| group | cars / trips | trip median v2 -> v2.1 | car median | engine-on gap |
|---|---|---|---|---|
| power-split HEV, Toyota-type label | 37 / 3,969 | -1.8 -> +0.5 | -1.1 -> +3.1 | -10.1 -> -3.8 |
| power-split HEV, other label (Ford Escape / Fusion / C-Max, Altima candidates) | 19 / 1,416 | +15.0 -> +17.2 | +16.3 -> +17.3 | +1.0 -> +6.9 |
| mild (CR-Z-type, low-confidence label; run as P1) | 30 / 2,856 | -1.3 -> -0.8 | +2.2 -> +2.7 | -5.0 -> +2.2 |
| plug-in power-split (Energi / Prius PHV) | 11 / 1,939 | -23.2 -> -22.7 | -33.2 -> -32.9 | (no rpm logged) |
| plug-in series (Volt-type) | 11 / 977 | -20.3 -> -20.3 | -20.3 -> -20.3 | (no rpm logged) |

What exam-2 says: v2.1 moves engine operation the right way everywhere it can be measured (VED Toyota engine-on gap
-10 -> -4 points) and the EPA errors by 0.1-2.5 points, mostly toward zero; it does not fix any group. The large errors
are the same as v2's and are now measured on fresh data: older Toyota/Lexus power-splits are 15-20 % too frugal on
the FTP (the TNGA-era Atkinson map stands in for 2008-2017 engines), TMED P2s 8-10 % too thirsty, converter P2s 26-28
% too frugal in the city, plug-in series 19 % too frugal, and plug-ins on VED roads 20-33 % low.

## Accuracy against real test results

1. **Conventional city vs highway bias — fixed in v2.** v1: FTP +5.6 %, HWFET -4.3 %. v2: +0.4 / -1.9 % (held-out),
   D3 exam -3.2 % on all tests. Still open: the per-engine spread (held-out FTP IQR 12.4, EPA dev pool 19.8) comes from
   one of ten benchmark maps per engine; per-map levels were tried and rejected on dev (CALIBRATION.md C3b).
2. **P2 parallel hybrids — partly fixed, now split in two.** v1: city FTP +12 % (+8 to +31). v2 (converter in P2
   automatics, start energy, brake blending 0.75 fitted on the D3 Sonata Hybrid): held-out FTP -4.0 % median, but the
   Hyundai/Kia TMED cars now burn too much (Elantra / Niro / Tucson FTP -16 / -15 / -4, US06 -24 / -24 / -8, SC03
   -18 / -28 / -13 %) while the converter P2 trucks stay too good in the city (F-150 Hybrid +18, Tacoma +9 %). One modern
   P2 in D3 dev was not enough; the brake-blending share is likely car-specific. D3 exam P2 (mostly 2010-2013 mild
   hybrids) -18.6 %. **v2.1 / exam-2 (14 Nov 2026):** the 0.75 share is not a physical limit: at 10 Hz the D3 Sonata
   puts 0.86 of its tyre braking energy into the pack, as the twin does at full regen. The Sonata burns more because
   it runs its engine twice as long at half the power (24 % vs 12 % of the UDDS, launch starts, 19 % of the fuel while
   braking). Every physical replacement tried made every dev P2 more frugal in the city (CALIBRATION.md H12-H20); the
   share stays. Exam-2 confirms the split on fresh cars: TMED FTP -9.8 % (53 records), converter P2 +26.1 % (19).
3. **Honda e:HEV direct drive — fixed.** v2 locks the engine at 22.7 rpm per km/h from 65 km/h (measured on the D3
   2015 Accord Hybrid) and prefers engine drive at cruise: held-out CR-V FTP -2.0, HWY -3.8 % (v1 -4.0 / -9.5). US06 on
   the dev Accord -8.5 % FE: open.
4. **Toyota power-split — on-road fixed, test cycles worse.** VED dev twins: fuel -20.3 -> -0.9 %, engine-on gap -24.6
   -> -10.9 points (engine on above 45 km/h; 1,250 W on-road load). On the held-out EPA cycles v2 burns 5-7 % too much
   (FTP -5.1, HWY -7.2, US06 -6.6; v1 +1.0 / -2.4 / -3.6): the 45 km/h rule fitted on real roads costs dyno cycles that
   the dev RAV4 and Escape did not show. 20 F cold FTP -16 % (v1 -15 %): open. The EPA dev Toyotas of the wider dev
   set (Grand Highlander Hybrid, LC 500h, Altima, RAV4) do not show that deficit (FTP +2.9 %), and the roads still
   want 45 km/h (CALIBRATION.md H16). **Exam-2:** 166 fresh Toyota/Lexus records, 2008-2026, are **too frugal**: FTP
   +17 %, HWY +7 %, SC03 +19 % (v2.1). Most are pre-TNGA engines (2AZ/2AR/2ZR/1NZ, 3.3-5.0 L V6/V8) that the TNGA
   paper map flatters, as the D3 2010 Prius (item 11) already showed. VED exam Toyota-type twins: fuel -1.8 -> +0.5 %
   (trip median), engine-on gap -10.1 -> -3.8 points with v2.1's run-on.
5. **Cold weather.** v1's cold FTP was +10-16 % too good. EPA procedure fixed (heater on, soaked tyres; both versions)
   and v2 cold physics: held-out conventional cold FTP -5.5 % (n 2), D3 exam conventional cold -4.9 % (v1 -9.0).
   Hybrids' held-out cold FTP: power-split -15.9 %, P2 +2.0 % (n 3 each; v1 -14.6 / +7.1).
6. **A/C.** v1 SC03 +9-15 % on conventional cars; v2 +1.9 % held-out (n 2), D3 exam conventional SC03 -8.4 % (v1
   -12.7). Electric A/C COP / Carnot 0.50 matches the logged compressor power of two D3 dev cars; hybrids' SC03 is now
   -18 to -28 % on the held-out TMED P2s and the D3 dev Accord Hybrid / Volt hot tests -17 / -18 %: open. On the wider
   dev set the hybrid A/C cost is too small for most groups (SC03 minus FTP error: Accord +17, plug-in P2 +24, Toyota
   +7 points; TMED -6) and exam-2 agrees (SC03 Toyota +19, plug-in series +46, converter P2 +38 %). Not changed: the
   compressor power is right where it was logged (CALIBRATION.md, hybrid A/C).
7. **Cabin heat (fixed cause, partly).** At -7 C the v2 draft cabin lost too much heat (D3 dev cabins 5-9 K too cold,
   coolant drawn to 45-50 C): conduction share 0.6 -> 0.3. Cabins are now 4-5 K too cold, coolant 66 vs 74 C (CX9 at
   1,200 s). VED dev plug-ins below 10 C are now 27-44 % low (v1 +6 to +16 % for the Volt): VED fragments start with a
   warm 21 C cabin in the twin; open.
8. **BEVs.** One loss-map scale (0.47, v1, fitted on modern EPA dev BEVs) holds: held-out UDDS -0.4, HWY -1.8 %; D3 exam
   -2.0 %. Old BEVs (2012-2015 i-MiEV, Leaf SV, Soul EV) use 10-17 % more than their twins (D3 dev).
9. **Plug-in hybrids — not met.** v2 adds charge-depleting rules, pack power from the machines, engine heat in charge
   sustaining. D3 dev -8 % (power-split) / -11 % (Volt); D3 exam -16 % / -27 %: the 2012-2013 Volts and the i3 REx run
   their engine 7 % of the time in the twin vs 36 % on the dyno (v1 17 %). VED dev Energi -12.6, Volt -10.7 %.
   With agent 3's harness (charge-sustaining tests at the twin's CS level and scored on corrected fuel) v2 is D3 exam
   -17.8 / -22.6 %, v2.1 the same. Exam-2 (charge-sustaining EPA tests of fresh cars): plug-in power-split FTP +0.5,
   HWY +2.2, 20 F cold -27 %; plug-in series FTP +19, SC03 +46 %; plug-in P2 FTP +7, HWY -8 %. VED exam plug-ins
   -20 to -23 % (trip median). The D3 2016 Volt (dev) runs its engine ~100 % above 60 km/h in charge sustaining, the
   twin 15 % overall; a speed rule for it was rejected on dev (H17).
10. **Hybrid motors use the generic IPM map**, not the measured 2010 Prius maps (chosen on the dev hybrids in v1).
11. **Legacy D3 hybrids** (2010 Prius, Fusion Hybrid, 2013 Energis in charge sustaining): 20-28 % more fuel than the
    twins. Their fuel is the legacy measurement and their engines are older than the TNGA-based Atkinson map. Not fitted.

## Heavy duty (no per-vehicle truck traces; ZTBus for buses)

12. **ZTBus (18 m battery-trolleybus) — new check.** Twin with mass from passengers + 19 t curb, nothing fitted:
    exam legs -2.3 % median (v1 -13.8 %), IQR 14.7 (22.5) points, 1.42 vs 1.59 kWh/km. The 19 t curb mass is a
    convention, not verified: on 40 dev legs both the logged power and the logged traction force regress to
    0.67-0.73 x (19 t + 70 kg per passenger) (TripDetective v2.0 NOTEBOOK, 14 Nov 2026). A mass read back from these
    logs has a level that is not known to better than ~30 %.
13. Class 8 diesel reference: 85.3 g CO2/ton-mile (v1 85.8), above the MY2021-23 standard of 75.7 as expected without
    Phase 2 packages; Willans map, plausibility only. Class 8 BEV 2.49 kWh/mi (v1 2.51) vs ~1.5-2.2 reported for loaded
    BEVs: the duty (65 mph, 19 t), not the motor (CALIBRATION.md C8).
14. **Transit buses.** BEV bus CBD14 1.83 kWh/mi (v1 1.66; Foothill in service 1.82-2.15). Hybrid bus +55 % vs diesel on
    CBD14 (v1 +76 %; literature 10-40 % in service): not tuned, no hybrid-bus trace data. Bus heating: the PTC loop
    controller saturates at stops (24 kW standstill at 2.5-12.5 C vs ~9 kW total on ZTBus), although the mean heating
    fell to ~9 kW at 7.5 C.
15. The CBD14 bus cycle was constructed from the SAE J1376 description (Manhattan/OCTA traces were not reachable).

## Model structure

16. **1 Hz by default.** v2 has 10 Hz sub-steps (`substeps=10`, off in the preset: <= 0.1 % trip energy at 10x run
    time). Launch engine starts of P2 hybrids (D3 Sonata: 29 starts per UDDS, sim 10-17) are not modelled.
17. **Rotating inertia**: engine + transmission input inertia acts in the engaged gear only; energy released at an
    upshift is booked as driveline loss.
18. **Tyres**: v2 has a brush tyre model (slip, load transfer, friction-ellipse regen limits) on by default; no per-brand
    tyre data beyond EU label classes.
19. **Driver**: perfect anticipation plus PI by default; v2's human driver model (lag, overshoot) is an option, not
    calibrated.
20. **Aerodynamics**: v2 scales station wind to road height (log profile); gusts and drafting are options (off: no
    dev gain, no gap data).
21. **Battery**: one RC branch; aging (capacity fade, resistance growth) is an input (`batt_age`, `batt_soh_*`), not a
    state that evolves within a trip.
22. **Hybrid controls are generic**: A-ECMS plus production rules stands in for each maker's calibration. Real hybrids
    load-level more gently than ECMS (charging power 2-3x lower when the engine runs; a cap was tried and rejected).
    DP and MPC see the future trace (benchmarks, not causal).
23. **Engine maps** are resized at constant BMEP from ten ALPHA engines (+ a Willans map for heavy duty); a specific
    engine can differ by +-5-10 %. No full-load enrichment model: D3 dev US06 -6 % vs EPA pool US06 +3 % (opposite
    signs, CALIBRATION.md H8).
24. **On-road extra loads** (conventional 700 W, HEV 1,250 W, plug-in and BEV 300 W) are fitted lumps from VED dev
    twins, applied to road runs only. They carry 12 V loads, real roads and the behaviour of trip fragments, not one
    physical load.
25. **Coast-down check is a consistency check** (the spec is built from the same EPA targets).

## Generated vehicles

26. Light-duty vehicles bootstrap real EPA test-car records. Generated hybrids stay less efficient than the EPA hybrid
    set with the default v1 donors (check_generator: HEV hot UDDS 47.5 mpg v2, 48.3 v1; EPA ~59); `donor_model='v2'`
    draws hybrid bodies from real hybrid records (docs/v2_b1.md). PHEV and P2 hardware sizing are rules of thumb.

## Speed and robustness

27. v2 runs at ~24,000-28,000x real time for conventional/BEV (v1 50,000-55,000x) and ~3,200-3,900x for hybrids (v1
    3,800-4,800x), single thread, UDDS; generated fleet 5,700x (v1 7,700x); DP on UDDS 1.6 s (v1 0.8 s). Ledger closure
    over 600 stress runs: max 5.6e-4, none over 0.5 %.
