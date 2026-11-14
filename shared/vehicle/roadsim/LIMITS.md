# RoadSim: known limits

What the simulator gets wrong, by how much where we measured it, and what to watch. Since 14 Nov 2026 the default
physics is **roadsim-physics-v2** (calibrated on dev data only: [CALIBRATION.md](CALIBRATION.md)); physics v1 stays
reproducible with `options='roadsim-physics-v1'`. Numbers are from `shared/vehicle/checks/*.json` (v1: plain names,
v2: `__roadsim-physics-v2`), computed once after v2 was frozen; held-out / exam data unless stated. Errors on EPA
cycles are on fuel economy (+ = the sim burns less); D3 and ZTBus errors are on energy (+ = the sim uses more).

## Held-out results, v1 -> v2 (median, [IQR], n)

| check | v1 | v2 | target (V2_PLAN) |
|---|---|---|---|
| EPA held-out conventional FTP | +5.6 [-4.2, +7.9] | +0.4 [-7.7, +4.7] n 15 | +-3, IQR halved: median met, IQR not (12.1 -> 12.4) |
| ... HWY | -4.3 [-5.5, -3.0] | -1.9 [-3.1, -1.5] n 15 | met |
| ... US06 | +0.2 | -1.8 n 5 | |
| ... SC03 / 20 F cold FTP | +11.4 / -6.0 | +1.9 / -5.5 n 2 / 2 | +-6: met (n 2) |
| EPA held-out P2 FTP | +11.9 [+9.3, +23.7] | -4.0 [-15.1, +8.6] n 5 | +-8: median met, spread not |
| ... HWY / US06 / SC03 | -6.0 / -5.1 / -6.1 | -9.7 / -23.6 / -17.7 | |
| EPA held-out power-split FTP / HWY / US06 | +1.0 / -2.4 / -3.6 | -5.1 / -7.2 / -6.6 n 4 | v2 worse |
| ... 20 F cold FTP | -14.6 | -15.9 n 3 | |
| EPA held-out e:HEV (2024 CR-V) FTP / HWY | -4.0 / -9.5 | -2.0 / -3.8 n 1 | +-6: met |
| EPA held-out BEV UDDS / HWY | -0.5 / -1.7 | -0.4 / -1.8 n 8 | |
| D3 exam conventional (all tests) | -5.1, IQR 12.2 | -3.2, IQR 10.9 n 240 | |
| ... cold / hot / SC03 | -9.0 / -12.3 / -12.7 | -4.9 / -6.3 / -8.4 | |
| D3 exam BEV | -3.5, IQR 19.8 | -2.0, IQR 14.3 n 62 | +-10 per trip: median met |
| D3 exam hybrids (mostly 2010-2013 P0/P1/P2 and the C-Max Hybrid) | -20.2 | -17.8 n 111 | |
| D3 exam plug-in power-split (Prius Plug-ins, RAV4 Prime) | -14.9 | -16.2 n 71 | +-15: not met |
| D3 exam plug-in series (2012/13 Volt, i3 REx) | -22.7 | -27.1 n 97 | not met; v2 worse |
| ZTBus exam legs (bus twin energy per leg) | -13.8 [-24.2, -1.7] | -2.3 [-9.8, +4.9] n 5,847 | |
| FASTSim cross-check, conventional UDDS / HWY / US06 | -0.3 / -3.1 / -3.0 | -1.9 / -1.2 / -6.4 | (v2 adds physics FASTSim lacks) |
| ... hybrids UDDS / HWY / US06 | +18.0 / +6.7 / -4.3 | +12.5 / +2.3 / -8.3 | |
| Ledger stress, 600 runs | max 5.1e-4 | max 5.6e-4, none > 0.5 % | closes: met |
| Speed, UDDS single thread | conv 50-55k x, hybrids 3.8-4.8k x | conv 24-26k x, hybrids 3.2-3.9k x | v2 costs 2x on conventional/BEV |

## Accuracy against real test results

1. **Conventional city vs highway bias — fixed in v2.** v1: FTP +5.6 %, HWFET -4.3 %. v2: +0.4 / -1.9 % (held-out),
   D3 exam -3.2 % on all tests. Still open: the per-engine spread (held-out FTP IQR 12.4, EPA dev pool 19.8) comes from
   one of ten benchmark maps per engine; per-map levels were tried and rejected on dev (CALIBRATION.md C3b).
2. **P2 parallel hybrids — partly fixed, now split in two.** v1: city FTP +12 % (+8 to +31). v2 (converter in P2
   automatics, start energy, brake blending 0.75 fitted on the D3 Sonata Hybrid): held-out FTP -4.0 % median, but the
   Hyundai/Kia TMED cars now burn too much (Elantra / Niro / Tucson FTP -16 / -15 / -4, US06 -24 / -24 / -8, SC03
   -18 / -28 / -13 %) while the converter P2 trucks stay too good in the city (F-150 Hybrid +18, Tacoma +9 %). One modern
   P2 in D3 dev was not enough; the brake-blending share is likely car-specific. D3 exam P2 (mostly 2010-2013 mild
   hybrids) -18.6 %.
3. **Honda e:HEV direct drive — fixed.** v2 locks the engine at 22.7 rpm per km/h from 65 km/h (measured on the D3
   2015 Accord Hybrid) and prefers engine drive at cruise: held-out CR-V FTP -2.0, HWY -3.8 % (v1 -4.0 / -9.5). US06 on
   the dev Accord -8.5 % FE: open.
4. **Toyota power-split — on-road fixed, test cycles worse.** VED dev twins: fuel -20.3 -> -0.9 %, engine-on gap -24.6
   -> -10.9 points (engine on above 45 km/h; 1,250 W on-road load). On the held-out EPA cycles v2 burns 5-7 % too much
   (FTP -5.1, HWY -7.2, US06 -6.6; v1 +1.0 / -2.4 / -3.6): the 45 km/h rule fitted on real roads costs dyno cycles that
   the dev RAV4 and Escape did not show. 20 F cold FTP -16 % (v1 -15 %): open.
5. **Cold weather.** v1's cold FTP was +10-16 % too good. EPA procedure fixed (heater on, soaked tyres; both versions)
   and v2 cold physics: held-out conventional cold FTP -5.5 % (n 2), D3 exam conventional cold -4.9 % (v1 -9.0).
   Hybrids' held-out cold FTP: power-split -15.9 %, P2 +2.0 % (n 3 each; v1 -14.6 / +7.1).
6. **A/C.** v1 SC03 +9-15 % on conventional cars; v2 +1.9 % held-out (n 2), D3 exam conventional SC03 -8.4 % (v1
   -12.7). Electric A/C COP / Carnot 0.50 matches the logged compressor power of two D3 dev cars; hybrids' SC03 is now
   -18 to -28 % on the held-out TMED P2s and the D3 dev Accord Hybrid / Volt hot tests -17 / -18 %: open.
7. **Cabin heat (fixed cause, partly).** At -7 C the v2 draft cabin lost too much heat (D3 dev cabins 5-9 K too cold,
   coolant drawn to 45-50 C): conduction share 0.6 -> 0.3. Cabins are now 4-5 K too cold, coolant 66 vs 74 C (CX9 at
   1,200 s). VED dev plug-ins below 10 C are now 27-44 % low (v1 +6 to +16 % for the Volt): VED fragments start with a
   warm 21 C cabin in the twin; open.
8. **BEVs.** One loss-map scale (0.47, v1, fitted on modern EPA dev BEVs) holds: held-out UDDS -0.4, HWY -1.8 %; D3 exam
   -2.0 %. Old BEVs (2012-2015 i-MiEV, Leaf SV, Soul EV) use 10-17 % more than their twins (D3 dev).
9. **Plug-in hybrids — not met.** v2 adds charge-depleting rules, pack power from the machines, engine heat in charge
   sustaining. D3 dev -8 % (power-split) / -11 % (Volt); D3 exam -16 % / -27 %: the 2012-2013 Volts and the i3 REx run
   their engine 7 % of the time in the twin vs 36 % on the dyno (v1 17 %). VED dev Energi -12.6, Volt -10.7 %.
10. **Hybrid motors use the generic IPM map**, not the measured 2010 Prius maps (chosen on the dev hybrids in v1).
11. **Legacy D3 hybrids** (2010 Prius, Fusion Hybrid, 2013 Energis in charge sustaining): 20-28 % more fuel than the
    twins. Their fuel is the legacy measurement and their engines are older than the TNGA-based Atkinson map. Not fitted.

## Heavy duty (no per-vehicle truck traces; ZTBus for buses)

12. **ZTBus (18 m battery-trolleybus) — new check.** Twin with mass from passengers + 19 t curb, nothing fitted:
    exam legs -2.3 % median (v1 -13.8 %), IQR 14.7 (22.5) points, 1.42 vs 1.59 kWh/km.
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
