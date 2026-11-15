#!/bin/sh
# Round 2 (15 Nov 2026), rule v2.2 (ratchet.py; fixed before R19 was scored). Each rung builds on the incumbent
# (incumbent.py). Settings were chosen on training-distribution probes (r2_explore.py: probe, probe_mild), never on dev
# or exam; every probe is listed in NOTEBOOK.md. Run from WORK with HP_PHYS=roadsim-physics-v3.1.
# usage: r2_ladder.sh R19|R20|R21|R22
PY=${PY:-python}; SRC="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1 HP_WORKERS=${HP_WORKERS:-3} HP_RULE=v2.2
R="$PY $SRC/ratchet.py"
INC() { $PY $SRC/incumbent.py "$@"; }
case "$1" in
R19) $R R19_phev_cs "$(INC ckpt)" "$(INC params '{"cls_fb":{"phev":{"fb_dead":0.0,"Kp":24}},"phev_cs_s":{"phev_powersplit":2.42,"phev_series":2.70}}')" "Plug-in charge-sustaining rule: at its CS level (soc_lo + 0.05) a plug-in is an HEV, so there s is no lower than the DP teacher's median costate of the HEV of the same architecture (power-split 2.42, series 2.70; prior_p31r0, training bank), and its SOC feedback below the CS level has no dead band and Kp 24. Frozen showed plug-ins dipping 0.03-0.07 below the CS level (series PHEV holds 7 of 12, A-ECMS 12 of 12); probe 60 plug-in trips: holds 88 -> 100 %, % of DP 100.00 -> 100.14" > logs/r19_p3.1.log 2>&1 ;;
R20) $R R20_mild_noband "$(INC ckpt)" "$(INC params '{"cls_fb":{"phev":{"fb_dead":0.0,"Kp":24},"mild":{"fb_dead":0.0}}}')" "Mild P1 hybrids: no dead band on the SOC feedback (s0 2.8 kept). The 0.58 kWh mild pack moves 1 % of SOC per second at 6 kW of load-point charging, so the band lets it ride up; dev mild twins end 0.10-0.13 above their start on short trips. Chosen on the new 80-trip mild probe (training mild vehicles): holds 96 -> 98 % (A-ECMS 98 %), % of DP 96.45 -> 96.35" > logs/r20_p3.1.log 2>&1 ;;
R21) $R R21_hev_kp18 "$(INC ckpt)" "$(INC params '{"cls_fb":{"phev":{"fb_dead":0.0,"Kp":24},"hev_parallel_p2":{"Kp":18},"hev_series":{"Kp":18},"hev_powersplit":{"Kp":18}}}')" "HEVs (incl. P2 trucks): Kp 18 beyond the +-0.03 dead band (was 12), so the band's steady offset is corrected faster. Frozen and dev showed P2 HEVs ending 0.12-0.14 below the window middle with the band (3 trips that v2.0 held). Probe 100 HEV trips: holds 92 -> 94 %, % of DP 98.90 -> 98.84" > logs/r21_p3.1.log 2>&1 ;;
R22) $R R22_bus_weak_fb "$(INC ckpt)" "$(INC params '{"cls_fb":{"phev":{"fb_dead":0.0,"Kp":24},"bus_hev_series":{"Kp":3,"Ki":0.0},"bus_hev_parallel_p2":{"Kp":3,"Ki":0.0}}}')" "Buses: weak SOC feedback (Kp 3 beyond the +-0.03 band, no integral term) around the fixed bus s (series 2.8, P2 2.6). A bus pack is large against a leg's energy swings, so Kp 12 pulls s far from the costate and spends charge at a poor price; constant-s ECMS (no feedback) beat v2.1 on frozen P2 buses (100.12 vs 97.40 % of DP). Probe 20 buses: % of DP 100.09 -> 100.26, holds 100 % (ECMS 100.01, holds 95 %)" > logs/r22_p3.1.log 2>&1 ;;
esac
