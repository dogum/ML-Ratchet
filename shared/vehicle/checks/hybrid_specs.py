"""RoadSim specs for EPA test-car hybrids and plug-ins (RoadData's EPA test-car table, data/proc/epa), by rule.

The EPA table gives what the test lab measured on (equivalent test weight, target road-load A/B/C, engine displacement
and rated power, transmission type and gears, N/V and axle ratios) but no electric hardware. HW below adds the hybrid
system of each model line from public spec sheets (machine power and torque, pack energy and voltage, converter or
clutch), rounded; rear axle motors of AWD power-splits are lumped into MG2 (as checks/epa_vehicles.py does). The rules
were written from the model names only, before any of these records was simulated (14 Nov 2026, v2.1 hybrid fix pass).

Groups (reporting units of checks/check_hybrids_epa.py):
  p2_tmed            Hyundai/Kia TMED: P2 motor in place of the converter (6DCT or 6AT without converter)
  p2_converter       P2 automatics that keep a torque converter (Ford 10R80 MHT, Toyota i-FORCE MAX, BMW, Mercedes,
                     Porsche Tiptronic)
  p2_clutch          other P2s without a converter (Infiniti/Nissan 7AT, Audi Q5 8AT, VW DSG, Toyota Hybrid MAX 6AT)
  mild               P0/P1 mild hybrids (Honda IMA, GM BAS, Mercedes S400, BMW ActiveHybrid 7 V8): no electric driving
  toyota_ps          Toyota/Lexus power-split (and Nissan's licensed Altima Hybrid)
  ford_ps            Ford/Lincoln/Mercury power-split
  other_ps           other two-motor input splits (Subaru's 2025 strong hybrid, Lincoln Nautilus HEV)
  honda_series       Honda i-MMD / e:HEV (series with direct-drive clutch)
  plugin_ps          plug-in power-split (Toyota, Ford, Chrysler Pacifica eFlite), charge-sustaining tests
  plugin_series      plug-in series / EREV (Volt, ELR, i3 REx, Karma, Clarity, Accord PHEV, Outlander), charge-sustaining
  plugin_p2          plug-in P2s in charge-sustaining tests, simulated as hev_parallel_p2 with the plug-in pack
Not modelled (excluded, with the reason): GM/Daimler/BMW two-mode, through-the-road and P4 designs (Volvo T8, BMW i8,
Mini, Dodge Hornet, Acura SH-AWD), P2 hybrids with a CVT (Nissan Rogue/Murano/Pathfinder), Porsche 918.

PHEV records: RoadData's FTP_mpg / HWY_mpg / US06 / SC03 / cold of a plug-in are its charge-sustaining tests (2012 Volt
47.5 / 53.6 mpg); the twin starts and is held at its charge-sustaining SOC (soc_cs).

Since 14 Nov 2026 the code lives in the roadsim package (roadsim/roadsim/hybrid_hw.py); this module keeps the old import
path for the checks and EnergyNet's training (models/energynet/training/epa_hybkind.py reads lookup()).
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'roadsim'))
from roadsim.hybrid_hw import *  # noqa: F401,F403
from roadsim.hybrid_hw import _num  # noqa: F401
