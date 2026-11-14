"""RoadSim specs for EPA test-car hybrids and plug-ins (RoadData's EPA test-car table, data/proc/epa), by rule
(roadsim.hybrid_hw; moved here from shared/vehicle/checks/hybrid_specs.py on 14 Nov 2026, which re-exports it).

Stable interface: lookup(make, model, year, disp) -> (kind, hw); lookup_ext (v3 supplementary rules first);
group_of(kind, hw); spec_of(row, ext=False) -> (spec, kind, group); epa_results(row); body_class(row); HW, SUPP_HW.

The EPA table gives what the test lab measured on (equivalent test weight, target road-load A/B/C, engine displacement
and rated power, transmission type and gears, N/V and axle ratios) but no electric hardware. HW below adds the hybrid
system of each model line from public spec sheets (machine power and torque, pack energy and voltage, converter or
clutch), rounded; rear axle motors of AWD power-splits are lumped into MG2 (as checks/epa_vehicles.py does). The rules
were written from the model names only, before any of these records was simulated (14 Nov 2026, calibration agent 3).

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
"""
import math, re, copy
import numpy as np

from .vehicle import GEAR_SETS, spec_from_abc
from .constants import LB, MPH

PICKUP = ('F150', 'F-150', 'SILVERADO', 'SIERRA', 'RAM', 'TACOMA', 'TUNDRA', 'COLORADO', 'CANYON', 'FRONTIER', 'RANGER',
          'GLADIATOR', 'TITAN', 'RIDGELINE', 'MAVERICK', 'SANTA CRUZ', 'K1500', 'C1500', 'K10', 'C10')
VAN = ('SIENNA', 'ODYSSEY', 'PACIFICA', 'CARNIVAL', 'TRANSIT', 'EXPRESS', 'SAVANA')

# (make regex, model regex, first MY, last MY, kind, hardware). First match wins.
# kind: ps | p2 | series | phev_ps | phev_series | phev_p2 | exclude
# ps:     mg2 kW (front + rear lumped), mg1 kW, batt (kWh, V, chem), mg2_rpm, mg2_ratio, mg2_T (Nm, optional)
# p2:     mot kW, mot_T Nm, batt, tc (converter kept), mild (no electric driving)
# series: mot kW, mot_T, gen kW, batt
# eng_kW overrides the EPA rated power (when the table carries the system power)
Y0, Y9 = 1990, 2100
HW = [
    # ---------------- exclusions (no RoadSim model for the architecture)
    ('CHEVROLET|GMC|CADILLAC|CHRYSLER|DODGE|GENERAL MOTORS', r'TAHOE|YUKON|ESCALADE|SILVERADO|SIERRA|C1500|K1500|C15|K15|ASPEN|DURANGO', Y0, Y9, 'exclude', 'GM two-mode'),
    ('MERCEDES', r'ML ?450', Y0, Y9, 'exclude', 'two-mode'),
    ('BMW', r'ACTIVEHYBRID X6|X6', Y0, Y9, 'exclude', 'two-mode'),
    ('CADILLAC', r'CT6', Y0, Y9, 'exclude', 'GM EVT plug-in (two-mode type)'),
    ('VOLVO', r'.', Y0, Y9, 'exclude', 'T8: rear-axle motor (P4, through the road)'),
    ('BMW', r'\bI8\b', Y0, Y9, 'exclude', 'through the road'),
    ('MINI', r'.', Y0, Y9, 'exclude', 'through the road'),
    ('DODGE', r'HORNET', Y0, Y9, 'exclude', 'rear-axle motor (P4)'),
    ('ACURA|HONDA', r'MDX|RLX|NSX', Y0, Y9, 'exclude', 'SH-AWD: P2 plus twin rear motors'),
    ('NISSAN|INFINITI', r'ROGUE|MURANO|PATHFINDER|QX60|JX', Y0, Y9, 'exclude', 'P2 with a CVT (RoadSim P2 needs stepped gears)'),
    ('PORSCHE', r'918', Y0, Y9, 'exclude', 'hypercar, three machines'),
    # ---------------- Toyota / Lexus power-split
    ('TOYOTA|LEXUS', r'PRIME|PLUG-IN|PHV|450H\+|550H\+', Y0, Y9, None, None),      # plug-ins below
    ('TOYOTA', r'PRIUS C\b', Y0, Y9, 'ps', dict(mg2=45, mg1=30, batt=(0.94, 144, 'NiMH'), mg2_rpm=13900, mg2_ratio=2.636, eng_kW=54)),
    ('TOYOTA', r'PRIUS', Y0, 2009, 'ps', dict(mg2=50, mg2_T=400, mg1=30, batt=(1.31, 202, 'NiMH'), mg2_rpm=6700, mg2_ratio=1.0)),
    ('TOYOTA', r'PRIUS', 2010, 2015, 'ps', dict(mg2=60, mg2_T=207, mg1=42, batt=(1.31, 202, 'NiMH'), mg2_rpm=13900, mg2_ratio=2.636)),
    ('TOYOTA', r'PRIUS', 2016, 2022, 'ps', dict(mg2=53, mg2_T=163, mg1=42, batt=(1.0, 207, 'NMC'), mg2_rpm=17000, mg2_ratio=2.636)),
    ('TOYOTA', r'PRIUS', 2023, Y9, 'ps', dict(mg2=83, mg1=55, batt=(0.9, 222, 'NMC'), mg2_rpm=17000)),
    ('LEXUS', r'CT ?200', Y0, Y9, 'ps', dict(mg2=60, mg2_T=207, mg1=42, batt=(1.31, 202, 'NiMH'), mg2_rpm=13900, mg2_ratio=2.636)),
    ('TOYOTA', r'COROLLA CROSS', Y0, Y9, 'ps', dict(mg2=113, mg1=55, batt=(0.9, 222, 'NMC'), mg2_rpm=17000)),
    ('TOYOTA', r'COROLLA', Y0, 2022, 'ps', dict(mg2=53, mg2_T=163, mg1=42, batt=(1.0, 207, 'NMC'), mg2_rpm=17000, mg2_ratio=2.636)),
    ('TOYOTA', r'COROLLA', 2023, Y9, 'ps', dict(mg2=70, mg2_T=185, mg1=42, batt=(0.9, 207, 'NMC'), mg2_rpm=17000, mg2_ratio=2.636)),
    ('TOYOTA|LEXUS|NISSAN', r'CAMRY|AVALON|ES ?300|HS ?250|ALTIMA', Y0, 2011, 'ps', dict(mg2=105, mg2_T=270, mg1=70, batt=(1.6, 245, 'NiMH'), mg2_rpm=14000)),
    ('TOYOTA|LEXUS', r'CAMRY|AVALON|ES ?300', 2012, 2017, 'ps', dict(mg2=105, mg2_T=270, mg1=70, batt=(1.6, 245, 'NiMH'), mg2_rpm=14000)),
    ('TOYOTA|LEXUS', r'CAMRY|AVALON|ES ?300', 2018, 2024, 'ps', dict(mg2=88, mg1=65, batt=(1.1, 259, 'NMC'), mg2_rpm=17000)),
    ('TOYOTA|LEXUS', r'CAMRY|AVALON|ES ?300', 2025, Y9, 'ps', dict(mg2=100, mg1=65, batt=(1.0, 259, 'NMC'), mg2_rpm=17000, awd_rear=40)),
    ('TOYOTA|LEXUS', r'HIGHLANDER.*MAX|TX ?500|RX ?500|CROWN.*MAX', Y0, Y9, 'p2', dict(mot=64, mot_T=292, batt=(1.0, 288, 'NiMH'), tc=False, turbo=True, awd_rear=76)),
    ('TOYOTA|LEXUS', r'GRAND HIGHLANDER', Y0, Y9, 'ps', dict(mg2=134, mg1=88, batt=(1.6, 245, 'NiMH'), mg2_rpm=17000, awd_rear=40)),
    ('TOYOTA|LEXUS', r'HIGHLANDER|RX ?400', Y0, 2010, 'ps', dict(mg2=123, mg2_T=333, mg1=109, batt=(1.9, 288, 'NiMH'), mg2_rpm=12400, awd_rear=50)),
    ('LEXUS', r'RX ?450', 2010, 2022, 'ps', dict(mg2=123, mg2_T=335, mg1=109, batt=(1.9, 288, 'NiMH'), mg2_rpm=14000, awd_rear=50)),
    ('TOYOTA', r'HIGHLANDER', 2011, 2019, 'ps', dict(mg2=123, mg2_T=335, mg1=109, batt=(1.9, 288, 'NiMH'), mg2_rpm=14000, awd_rear=50)),
    ('TOYOTA', r'HIGHLANDER', 2020, Y9, 'ps', dict(mg2=134, mg1=88, batt=(1.6, 245, 'NiMH'), mg2_rpm=17000, awd_rear=40)),
    ('LEXUS', r'RX ?350', Y0, Y9, 'ps', dict(mg2=134, mg1=88, batt=(1.6, 245, 'NiMH'), mg2_rpm=17000, awd_rear=40)),
    ('TOYOTA', r'RAV4', Y0, 2018, 'ps', dict(mg2=105, mg1=65, batt=(1.6, 245, 'NiMH'), mg2_rpm=14000, awd_rear=50)),
    ('TOYOTA', r'RAV4', 2019, Y9, 'ps', dict(mg2=88, mg1=65, batt=(1.6, 245, 'NiMH'), mg2_rpm=17000, awd_rear=40)),
    ('LEXUS', r'NX ?300', Y0, Y9, 'ps', dict(mg2=105, mg1=65, batt=(1.6, 245, 'NiMH'), mg2_rpm=14000, awd_rear=50)),
    ('LEXUS', r'NX ?350', Y0, Y9, 'ps', dict(mg2=134, mg1=88, batt=(1.1, 259, 'NMC'), mg2_rpm=17000, awd_rear=40)),
    ('LEXUS', r'UX', Y0, Y9, 'ps', dict(mg2=80, mg2_T=202, mg1=55, batt=(1.3, 216, 'NiMH'), mg2_rpm=17000, awd_rear=5)),
    ('TOYOTA|LEXUS', r'GS ?450', Y0, Y9, 'ps', dict(mg2=147, mg2_T=275, mg1=110, batt=(1.65, 288, 'NiMH'), mg2_rpm=14000)),
    ('TOYOTA|LEXUS', r'LS ?600', Y0, Y9, 'ps', dict(mg2=165, mg2_T=300, mg1=130, batt=(1.9, 288, 'NiMH'), mg2_rpm=14000)),
    ('TOYOTA|LEXUS', r'LS ?500|LC ?500', Y0, Y9, 'ps', dict(mg2=132, mg2_T=300, mg1=100, batt=(1.1, 310, 'NMC'), mg2_rpm=14000)),
    ('TOYOTA|LEXUS', r'TACOMA|TUNDRA|LX ?700|SEQUOIA', Y0, Y9, 'p2', dict(mot=36, mot_T=250, batt=(1.87, 288, 'NiMH'), tc=True, turbo=True)),
    ('SUBARU', r'CROSSTREK|FORESTER', 2025, Y9, 'ops', dict(mg2=88, mg1=60, batt=(1.1, 259, 'NMC'), mg2_rpm=17000)),
    ('SUBARU', r'.', Y0, Y9, 'exclude', 'Subaru P2 with a CVT'),
    # ---------------- Ford
    ('FORD|LINCOLN', r'AVIATOR', Y0, Y9, 'phev_p2', dict(mot=75, mot_T=300, batt=(13.6, 350, 'NMC'), tc=True, turbo=True)),
    ('FORD', r'F-?150|EXPLORER|INTERCEPTOR UTILITY|POLICE INTERCEPTOR', 2020, Y9, 'p2', dict(mot=35, mot_T=240, batt=(1.5, 300, 'NMC'), tc=True)),
    ('FORD|LINCOLN', r'NAUTILUS', Y0, Y9, 'ops', dict(mg2=100, mg1=70, batt=(1.4, 280, 'NMC'), mg2_rpm=14000, turbo=True)),
    ('FORD|LINCOLN', r'CORSAIR|ESCAPE.*(PHEV|PLUG)|KUGA', Y0, Y9, 'phev_ps', dict(mg2=96, mg1=70, batt=(14.4, 350, 'NMC'), mg2_rpm=14000)),
    ('FORD|LINCOLN', r'ENERGI|C-MAX.*PHEV|FUSION.*PHEV|MKZ.*PHEV', Y0, Y9, 'phev_ps', dict(mg2=88, mg2_T=240, mg1=70, batt=(7.6, 300, 'NMC'), mg2_rpm=14000)),
    ('FORD|MERCURY|MAZDA', r'ESCAPE|MARINER|TRIBUTE', Y0, 2012, 'fps', dict(mg2=70, mg1=45, batt=(1.8, 330, 'NiMH'), mg2_rpm=14000)),
    ('FORD|MERCURY|LINCOLN', r'FUSION|MILAN|MKZ', Y0, 2012, 'fps', dict(mg2=79, mg2_T=228, mg1=70, batt=(1.3, 275, 'NiMH'), mg2_rpm=14000)),
    ('FORD|LINCOLN', r'C-MAX|FUSION|MKZ|POLICE RESPONDER', 2013, Y9, 'fps', dict(mg2=88, mg2_T=240, mg1=70, batt=(1.4, 280, 'NMC'), mg2_rpm=14000)),
    ('FORD', r'ESCAPE|MAVERICK|BRONCO SPORT', 2020, Y9, 'fps', dict(mg2=88, mg1=70, batt=(1.1, 280, 'NMC'), mg2_rpm=14000)),
    # ---------------- Hyundai / Kia (TMED)
    ('HYUNDAI|KIA', r'IONIQ.*(PHEV|PLUG)|NIRO.*(PHEV|PLUG)', Y0, 2022, 'phev_p2', dict(mot=44.5, mot_T=170, batt=(8.9, 360, 'NMC'), tc=False)),
    ('HYUNDAI|KIA', r'NIRO.*(PHEV|PLUG)', 2023, Y9, 'phev_p2', dict(mot=62, mot_T=170, batt=(11.1, 360, 'NMC'), tc=False)),
    ('HYUNDAI|KIA', r'(SONATA|OPTIMA|K5).*(PHEV|PLUG)', Y0, Y9, 'phev_p2', dict(mot=50, mot_T=205, batt=(9.8, 360, 'NMC'), tc=False)),
    ('HYUNDAI|KIA', r'(TUCSON|SANTA FE|SORENTO|SPORTAGE).*(PHEV|PLUG)', Y0, Y9, 'phev_p2', dict(mot=67, mot_T=304, batt=(13.8, 360, 'NMC'), tc=False, turbo=True)),
    ('HYUNDAI|KIA', r'SONATA|OPTIMA|K5', Y0, 2015, 'p2', dict(mot=30, mot_T=205, batt=(1.4, 270, 'NMC'), tc=False, tmed=True)),
    ('HYUNDAI|KIA', r'SONATA|OPTIMA|K5', 2016, 2019, 'p2', dict(mot=38, mot_T=205, batt=(1.62, 270, 'NMC'), tc=False, tmed=True)),
    ('HYUNDAI|KIA', r'SONATA|OPTIMA|K5', 2020, Y9, 'p2', dict(mot=39, mot_T=205, batt=(1.76, 270, 'NMC'), tc=False, tmed=True)),
    ('HYUNDAI|KIA', r'IONIQ|NIRO', Y0, 2022, 'p2', dict(mot=32, mot_T=170, batt=(1.56, 240, 'NMC'), tc=False, tmed=True)),
    ('HYUNDAI|KIA', r'NIRO|ELANTRA', Y0, Y9, 'p2', dict(mot=32, mot_T=170, batt=(1.32, 240, 'NMC'), tc=False, tmed=True)),
    ('HYUNDAI|KIA', r'TUCSON|SANTA FE|SORENTO|SPORTAGE', Y0, Y9, 'p2', dict(mot=44, mot_T=264, batt=(1.49, 270, 'NMC'), tc=False, turbo=True, tmed=True)),
    ('HYUNDAI|KIA', r'CARNIVAL', Y0, Y9, 'p2', dict(mot=54, mot_T=264, batt=(1.49, 270, 'NMC'), tc=False, turbo=True, tmed=True)),
    ('HYUNDAI|KIA', r'PALISADE|TELLURIDE', Y0, Y9, 'p2', dict(mot=54, mot_T=264, batt=(1.65, 270, 'NMC'), tc=False, turbo=True, tmed=True)),
    # ---------------- Honda / Acura
    ('HONDA', r'CLARITY', Y0, Y9, 'phev_series', dict(mot=135, mot_T=315, gen=100, batt=(17.0, 310, 'NMC'), dd=True)),
    ('HONDA', r'ACCORD.*(PHEV|PLUG)', Y0, Y9, 'phev_series', dict(mot=124, mot_T=307, gen=105, batt=(6.7, 320, 'NMC'), dd=True)),
    ('HONDA', r'ACCORD', 2014, 2016, 'series', dict(mot=124, mot_T=307, gen=105, batt=(1.3, 259, 'NMC'))),
    ('HONDA', r'ACCORD', 2017, Y9, 'series', dict(mot=135, mot_T=315, gen=105, batt=(1.06, 260, 'NMC'))),
    ('HONDA', r'INSIGHT', 2019, Y9, 'series', dict(mot=96, mot_T=267, gen=80, batt=(1.1, 173, 'NMC'))),
    ('HONDA', r'CR-V|CIVIC', 2017, Y9, 'series', dict(mot=135, mot_T=335, gen=105, batt=(1.06, 260, 'NMC'))),
    ('HONDA|ACURA', r'INSIGHT|CR-Z', Y0, 2016, 'p2', dict(mot=10, mot_T=78, batt=(0.58, 101, 'NiMH'), tc=False, mild=True, cvt_as='DCT')),
    ('HONDA|ACURA', r'CIVIC|ILX', Y0, 2011, 'p2', dict(mot=15, mot_T=103, batt=(0.87, 158, 'NiMH'), tc=False, mild=True, cvt_as='DCT')),
    ('HONDA|ACURA', r'CIVIC|ILX', 2012, 2016, 'p2', dict(mot=17, mot_T=106, batt=(0.8, 144, 'NMC'), tc=False, mild=True, cvt_as='DCT')),
    # ---------------- GM
    ('CHEVROLET|CADILLAC', r'VOLT|ELR', Y0, 2015, 'phev_series', dict(mot=111, mot_T=370, gen=55, batt=(16.5, 360, 'NMC'), eng_kW=63)),
    ('CADILLAC', r'ELR', 2016, Y9, 'phev_series', dict(mot=117, mot_T=373, gen=55, batt=(17.1, 360, 'NMC'), eng_kW=63)),
    ('CHEVROLET', r'VOLT', 2016, Y9, 'phev_series', dict(mot=111, mot_T=398, gen=75, batt=(18.4, 360, 'NMC'), eng_kW=75)),
    ('CHEVROLET|SATURN|BUICK|GENERAL MOTORS', r'MALIBU|VUE|AURA|LACROSSE|REGAL', Y0, 2015, 'p2', dict(mot=11, mot_T=110, batt=(0.5, 115, 'NMC'), tc=True, mild=True)),
    # ---------------- BMW
    ('BMW', r'I3', Y0, 2016, 'phev_series', dict(mot=125, mot_T=250, gen=28, batt=(18.8, 360, 'NMC'), eng_kW=28)),
    ('BMW', r'I3', 2017, Y9, 'phev_series', dict(mot=127, mot_T=250, gen=28, batt=(27.2, 360, 'NMC'), eng_kW=28)),
    ('BMW', r'ACTIVEHYBRID 7', Y0, 2012, 'p2', dict(mot=15, mot_T=210, batt=(0.4, 120, 'NMC'), tc=True, mild=True, turbo=True)),
    ('BMW', r'ACTIVEHYBRID', Y0, Y9, 'p2', dict(mot=40, mot_T=210, batt=(1.35, 317, 'NMC'), tc=True, turbo=True)),
    ('BMW', r'330E', Y0, 2019, 'phev_p2', dict(mot=65, mot_T=250, batt=(7.6, 293, 'NMC'), tc=True, turbo=True)),
    ('BMW', r'330E|X3', 2020, Y9, 'phev_p2', dict(mot=80, mot_T=265, batt=(12.0, 350, 'NMC'), tc=True, turbo=True)),
    ('BMW', r'530E|740E|X5 XDRIVE40E', Y0, 2019, 'phev_p2', dict(mot=83, mot_T=250, batt=(9.2, 350, 'NMC'), tc=True, turbo=True)),
    ('BMW', r'530E|745E|X5 XDRIVE45E', 2020, Y9, 'phev_p2', dict(mot=83, mot_T=265, batt=(12.0, 350, 'NMC'), tc=True, turbo=True)),
    ('BMW', r'550E|750E|X5 XDRIVE50E', Y0, Y9, 'phev_p2', dict(mot=145, mot_T=280, batt=(19.4, 400, 'NMC'), tc=True, turbo=True)),
    # ---------------- Mercedes, Audi, Porsche, Bentley, Land Rover, Jeep, VW, Infiniti
    ('MERCEDES', r'S ?400', Y0, 2013, 'p2', dict(mot=15, mot_T=160, batt=(0.8, 120, 'NMC'), tc=True, mild=True)),
    ('MERCEDES', r'E ?400', Y0, Y9, 'p2', dict(mot=20, mot_T=250, batt=(0.8, 126, 'NMC'), tc=True)),
    ('MERCEDES', r'C ?350', Y0, Y9, 'phev_p2', dict(mot=60, mot_T=340, batt=(6.2, 290, 'NMC'), tc=True, turbo=True)),
    ('MERCEDES', r'GLC ?350', Y0, 2020, 'phev_p2', dict(mot=85, mot_T=340, batt=(8.7, 350, 'NMC'), tc=True, turbo=True)),
    ('MERCEDES', r'S ?5[56]0', Y0, Y9, 'phev_p2', dict(mot=90, mot_T=440, batt=(13.5, 400, 'NMC'), tc=True, turbo=True)),
    ('MERCEDES', r'GLC ?350|GLE ?450 ?E', 2021, Y9, 'phev_p2', dict(mot=100, mot_T=440, batt=(24.8, 400, 'NMC'), tc=True, turbo=True)),
    ('MERCEDES', r'AMG.*E ?53|E ?53', Y0, Y9, 'phev_p2', dict(mot=120, mot_T=480, batt=(21.2, 400, 'NMC'), tc=True, turbo=True)),
    ('AUDI', r'A3', Y0, Y9, 'phev_p2', dict(mot=75, mot_T=330, batt=(8.8, 350, 'NMC'), tc=False, turbo=True)),
    ('AUDI', r'A8', Y0, Y9, 'phev_p2', dict(mot=100, mot_T=350, batt=(14.1, 380, 'NMC'), tc=True, turbo=True)),
    ('AUDI', r'Q5.*(PHEV|TFSI ?E|PA)', Y0, Y9, 'phev_p2', dict(mot=105, mot_T=350, batt=(17.9, 380, 'NMC'), tc=False, turbo=True)),
    ('AUDI', r'Q5', Y0, Y9, 'p2', dict(mot=40, mot_T=210, batt=(1.3, 266, 'NMC'), tc=False, turbo=True)),
    ('PORSCHE', r'PANAMERA', Y0, 2012, 'p2', dict(mot=34, mot_T=300, batt=(1.7, 288, 'NiMH'), tc=True)),
    ('PORSCHE', r'PANAMERA', 2017, 2020, 'phev_p2', dict(mot=100, mot_T=400, batt=(14.1, 380, 'NMC'), tc=False, turbo=True)),
    ('PORSCHE', r'PANAMERA', 2021, 2023, 'phev_p2', dict(mot=100, mot_T=400, batt=(17.9, 380, 'NMC'), tc=False, turbo=True)),
    ('PORSCHE', r'PANAMERA', 2024, Y9, 'phev_p2', dict(mot=140, mot_T=450, batt=(25.9, 400, 'NMC'), tc=False, turbo=True)),
    ('PORSCHE', r'PANAMERA|CAYENNE', Y0, 2018, 'phev_p2', dict(mot=70, mot_T=310, batt=(10.8, 380, 'NMC'), tc=True, turbo=False)),
    ('PORSCHE', r'PANAMERA|CAYENNE', 2019, 2023, 'phev_p2', dict(mot=100, mot_T=400, batt=(17.9, 380, 'NMC'), tc=True, turbo=True)),
    ('PORSCHE', r'PANAMERA|CAYENNE', 2024, Y9, 'phev_p2', dict(mot=130, mot_T=460, batt=(25.9, 400, 'NMC'), tc=True, turbo=True)),
    ('BENTLEY', r'.', Y0, Y9, 'phev_p2', dict(mot=100, mot_T=400, batt=(17.3, 380, 'NMC'), tc=True, turbo=True)),
    ('LAND ROVER|JAGUAR', r'.', Y0, 2022, 'phev_p2', dict(mot=85, mot_T=275, batt=(13.1, 300, 'NMC'), tc=True, turbo=True)),
    ('LAND ROVER|JAGUAR', r'.', 2023, Y9, 'phev_p2', dict(mot=105, mot_T=275, batt=(38.2, 400, 'NMC'), tc=True, turbo=True)),
    ('JEEP', r'4XE|PHEV', Y0, Y9, 'phev_p2', dict(mot=107, mot_T=245, batt=(17.0, 400, 'NMC'), tc=True, turbo=True)),
    ('VOLKSWAGEN', r'JETTA', Y0, Y9, 'p2', dict(mot=20, mot_T=150, batt=(1.1, 220, 'NMC'), tc=False, turbo=True)),
    ('INFINITI|NISSAN', r'Q50|Q70|M35|M ?HYBRID', Y0, Y9, 'p2', dict(mot=50, mot_T=270, batt=(1.4, 346, 'NMC'), tc=False)),
    # ---------------- other plug-ins
    ('CHRYSLER', r'PACIFICA', Y0, Y9, 'phev_ps', dict(mg2=84, mg1=68, batt=(16.0, 350, 'NMC'), mg2_rpm=14000)),
    ('MITSUBISHI', r'OUTLANDER', Y0, 2018, 'phev_series', dict(mot=120, mot_T=332, gen=70, batt=(12.0, 300, 'NMC'), dd=True)),
    ('MITSUBISHI', r'OUTLANDER', 2019, 2022, 'phev_series', dict(mot=130, mot_T=332, gen=70, batt=(13.8, 300, 'NMC'), dd=True)),
    ('MITSUBISHI', r'OUTLANDER', 2023, Y9, 'phev_series', dict(mot=200, mot_T=450, gen=100, batt=(20.0, 350, 'NMC'), dd=True)),
    ('FISKER|KARMA', r'.', Y0, Y9, 'phev_series', dict(mot=300, mot_T=1300, gen=175, batt=(20.0, 350, 'NMC'), turbo=True)),
]
# plug-in power-splits (matched after the TOYOTA|LEXUS placeholder above)
HW_PHEV_TOYOTA = [
    (r'PRIUS', Y0, 2016, dict(mg2=60, mg2_T=207, mg1=42, batt=(4.4, 207, 'NMC'), mg2_rpm=13900, mg2_ratio=2.636)),
    (r'PRIUS', 2017, 2022, dict(mg2=53, mg2_T=163, mg1=23, batt=(8.8, 352, 'NMC'), mg2_rpm=17000, mg2_ratio=2.636)),
    (r'PRIUS', 2023, Y9, dict(mg2=120, mg1=55, batt=(13.6, 352, 'NMC'), mg2_rpm=17000)),
    (r'RAV4|NX ?450|RX ?450|TX ?550', Y0, Y9, dict(mg2=174, mg2_T=270, mg1=100, batt=(18.1, 355, 'NMC'), mg2_rpm=17000)),
]
GROUP = {'ps': 'toyota_ps', 'fps': 'ford_ps', 'ops': 'other_ps', 'series': 'honda_series', 'phev_ps': 'plugin_ps',
         'phev_series': 'plugin_series', 'phev_p2': 'plugin_p2'}


HYBRID_MAX = dict(mot=64, mot_T=292, batt=(1.0, 288, 'NiMH'), tc=False, turbo=True, awd_rear=76)

# Supplementary rules (14 Nov 2026, RoadSim v3 pass), written from public spec sheets before any of these records was
# simulated. They cover the exam-3 records (checks/exam3_hybrids.py): MY2000-2007 hybrids (EPA files RoadData's table
# does not hold) and hybrids RoadData labels conventional because the model name has no hybrid word. lookup() is
# unchanged (EnergyNet's epa_hybkind.py reads it); lookup_ext() tries these first, then lookup().
SUPP_HW = [
    ('TOYOTA', r'PRIUS', Y0, 2003, 'ps', dict(mg2=33, mg2_T=350, mg1=15, batt=(1.79, 274, 'NiMH'), mg2_rpm=5600, mg2_ratio=1.0, eng_kW=52)),
    ('HONDA', r'CIVIC', Y0, 2005, 'p2', dict(mot=10, mot_T=62, batt=(0.86, 144, 'NiMH'), tc=False, mild=True, cvt_as='DCT')),
    ('TOYOTA', r'VENZA', 2021, Y9, 'ps', dict(mg2=88, mg1=65, batt=(1.1, 259, 'NMC'), mg2_rpm=17000, awd_rear=40)),
    ('TOYOTA', r'CROWN', 2023, Y9, 'ps', dict(mg2=88, mg1=65, batt=(1.6, 245, 'NiMH'), mg2_rpm=17000, awd_rear=40)),
    ('MAZDA', r'CX-50', 2025, Y9, 'ps', dict(mg2=88, mg1=65, batt=(1.6, 245, 'NiMH'), mg2_rpm=17000, awd_rear=40)),
    ('HONDA', r'PRELUDE', 2026, Y9, 'series', dict(mot=135, mot_T=315, gen=105, batt=(1.06, 260, 'NMC'))),
    ('SUBARU', r'CROSSTREK', 2019, 2023, 'phev_ps', dict(mg2=88, mg1=65, batt=(8.8, 351, 'NMC'), mg2_rpm=17000)),
    ('TOYOTA', r'SIENNA', 2021, Y9, 'ps', dict(mg2=134, mg1=88, batt=(1.6, 245, 'NiMH'), mg2_rpm=17000, awd_rear=40)),
    ('FORD', r'MAVERIK', 2026, Y9, 'fps', dict(mg2=88, mg1=70, batt=(1.1, 280, 'NMC'), mg2_rpm=14000)),
    ('AUDI', r'A7.*TFSI ?E', Y0, Y9, 'phev_p2', dict(mot=105, mot_T=350, batt=(14.1, 380, 'NMC'), tc=False, turbo=True)),
    ('CHEVROLET', r'MALIBU', 2016, Y9, 'exclude', 'GM two-motor EVT (two-mode type)'),
]


def lookup_ext(make, model, year, disp=None):
    """lookup() with the supplementary rules of the v3 pass tried first."""
    mk = str(make).upper(); md = re.sub(r'\s+', ' ', str(model).upper())
    for mre, dre, y0, y1, kind, hw in SUPP_HW:
        if re.search(mre, mk) and re.search(dre, md) and y0 <= year <= y1:
            return kind, hw
    return lookup(make, model, year, disp)


def lookup(make, model, year, disp=None):
    """-> (kind, hw) or ('exclude', reason) or (None, 'no rule')."""
    mk = str(make).upper(); md = re.sub(r'\s+', ' ', str(model).upper())
    if (re.search('TOYOTA|LEXUS', mk) and year >= 2023 and disp is not None and 2.3 <= disp <= 2.45
            and not re.search(r'TACOMA|TUNDRA|PLUG|PRIME|\+', md)):
        return 'p2', HYBRID_MAX            # 2.4 L turbo "Hybrid MAX": P2 with a wet start clutch, 6AT, rear eAxle
    for mre, dre, y0, y1, kind, hw in HW:
        if not re.search(mre, mk) or not re.search(dre, md) or not (y0 <= year <= y1):
            continue
        if kind is None:          # Toyota / Lexus plug-in
            for dre2, a, b, h in HW_PHEV_TOYOTA:
                if re.search(dre2, md) and a <= year <= b:
                    return 'phev_ps', h
            return None, 'no plug-in rule'
        return kind, hw
    return None, 'no rule'


def group_of(kind, hw):
    if kind == 'p2':
        if hw.get('mild'):
            return 'mild'
        if hw.get('tmed'):
            return 'p2_tmed'
        return 'p2_converter' if hw.get('tc') else 'p2_clutch'
    return GROUP.get(kind)


def trans_of(t, n):
    t = str(t)
    if 'Continuously' in t:
        return 'CVT', n
    if 'Automated Manual' in t or t == 'Manual':
        return 'DCT', n
    return 'AT', n


def body_class(r):
    model = str(r.model).upper()
    if r.veh_type == 'Car':
        return 'car'
    if any(k in model for k in PICKUP):
        return 'pickup'
    if any(k in model for k in VAN):
        return 'van'
    return 'suv'


def _num(x, d=None):
    try:
        x = float(x)
        return x if math.isfinite(x) else d
    except (TypeError, ValueError):
        return d


def spec_of(r, ext=False):
    """EPA test-car row (realdata.epa_testcars(), a pandas row) -> (spec, kind, group) or (None, 'exclude'|None, reason).
    ext: use lookup_ext (the supplementary rules of the v3 pass first). The spec carries the model year
    ('model_year'; read only by the v3 era option, see roadsim/options.py)."""
    year = int(r.model_year)
    kind, hw = (lookup_ext if ext else lookup)(r.make, r.model, year, _num(r.displacement_L))
    if kind in (None, 'exclude'):
        return None, kind, hw
    if hw.get('mild') and kind != 'p2':
        return None, None, 'mild rule on a non-P2'
    cls = body_class(r)
    mass = float(r.etw_lb) * LB
    disp = _num(r.displacement_L, 2.0)
    P = _num(r.rated_hp, 100.0) * 0.7457
    nv = _num(r.nv_ratio, 0.0); axle = _num(r.axle_ratio, 0.0)
    awd = 'All' in str(r.drive) or '4-Wheel' in str(r.drive)
    spec = {'id': 'epahyb_%s' % r.name, 'name': f'{year} {r.make} {r.model}', 'class': cls, 'mass_kg': mass, 'model_year': year,
            'rot_mass_frac': 0.015, 'tire': {'K_t': 0.0, 'temp_effects': False}}
    rr = 0.33 if cls == 'car' else 0.36 if cls in ('suv', 'van') else 0.39
    spec['wheel_radius_m'] = rr
    k_nv = 2 * math.pi * rr / (60 * MPH)
    Pe = float(hw.get('eng_kW', P))
    turbo = hw.get('turbo', False)
    c, V, ch = hw['batt']
    plug = kind.startswith('phev')
    if kind in ('ps', 'fps', 'ops', 'phev_ps'):
        spec['architecture'] = 'phev_powersplit' if plug else 'hev_powersplit'
        bm = 'gas_turbo_ford_27' if turbo else 'gas_atk_toyota_tnga_hev_paper'
        spec['engine'] = dict(fuel='gasoline', displacement_L=disp, max_power_kW=Pe, bsfc_map=bm)
        spec['transmission'] = dict(type='eCVT', ring_teeth=78, sun_teeth=30, mg2_ratio=hw.get('mg2_ratio', 3.0))
        spec['final_drive'] = axle if axle > 2 else 3.3
        mg2 = hw['mg2'] + (hw.get('awd_rear', 0) if awd else 0)
        m2 = dict(role='MG2', max_power_kW=float(mg2), eff_map='ipm_generic_150', max_rpm=hw.get('mg2_rpm', 14000))
        if hw.get('mg2_T') and not (awd and hw.get('awd_rear')):
            m2['max_torque_Nm'] = float(hw['mg2_T'])
        spec['motors'] = [dict(role='MG1', max_power_kW=float(hw['mg1']), eff_map='ipm_generic_150', max_rpm=13500), m2]
        spec['battery'] = dict(capacity_kWh=c, nominal_V=V, chemistry=ch, soc_window=[0.15, 0.95] if plug else [0.4, 0.8])
        if plug:
            spec['battery']['max_power_kW'] = round(min(mg2 / 0.9, 9 * c), 1)
    elif kind in ('p2', 'phev_p2'):
        spec['architecture'] = 'hev_parallel_p2'
        if turbo:
            bm = 'gas_turbo_honda_15' if disp < 2.2 else 'gas_turbo_ford_27'
        elif hw.get('mild') or disp >= 3.0:
            bm = 'gas_v6_gm_43' if disp >= 3.0 else 'gas_na_toyota_a25a'
        else:
            bm = 'gas_atk_toyota_tnga_hev_paper'
        if disp >= 4.5 and not turbo:
            bm = 'gas_v8_ford_73'
        spec['engine'] = dict(fuel='gasoline', displacement_L=disp, max_power_kW=Pe, bsfc_map=bm)
        n = int(_num(r.gears, 6) or 6)
        t, n = trans_of(r.trans_type, n)
        if t == 'CVT' or n < 4:
            t, n = hw.get('cvt_as', 'DCT'), 6
        t = 'AT' if hw.get('tc') else 'DCT'
        gears = GEAR_SETS.get((t, n)) or GEAR_SETS.get(('AT', n)) or list(np.round(np.geomspace(4.6, 0.65, n), 3))
        spec['transmission'] = dict(type=t, gears=list(gears))
        top = gears[-1]
        spec['final_drive'] = nv * k_nv / top if nv > 15 else (axle if axle > 2 else 3.5)
        mot = hw['mot'] + (hw.get('awd_rear', 0) if awd else 0)
        spec['motors'] = [dict(role='P2', max_power_kW=float(mot), max_torque_Nm=float(hw['mot_T']), eff_map='ipm_generic_150', max_rpm=7000)]
        if plug:
            spec['battery'] = dict(capacity_kWh=c, nominal_V=V, chemistry=ch, soc_window=[0.15, 0.95],
                                   max_power_kW=round(min(mot / 0.9, 9 * c), 1))
            spec['control'] = {'strategy': 'aecms', 'params': {'soc_target': 0.2}}      # charge-sustaining level
        else:
            spec['battery'] = dict(capacity_kWh=c, nominal_V=V, chemistry=ch, soc_window=[0.3, 0.8])
        if hw.get('mild'):
            spec['control'] = {'strategy': 'aecms', 'params': {'engine_on_moving': True}}
    elif kind in ('series', 'phev_series'):
        spec['architecture'] = 'phev_series' if plug else 'hev_series'
        bm = 'gas_turbo_ford_27' if turbo else 'gas_atk_toyota_tnga_hev_paper'
        spec['engine'] = dict(fuel='gasoline', displacement_L=disp, max_power_kW=Pe, bsfc_map=bm)
        spec['transmission'] = dict(type='single')
        spec['motors'] = [dict(role='traction', max_power_kW=float(hw['mot']), max_torque_Nm=float(hw['mot_T']), eff_map='ipm_generic_150', max_rpm=14000),
                          dict(role='generator', max_power_kW=float(hw['gen']), eff_map='ipm_generic_150', max_rpm=14000)]
        spec['final_drive'] = 14000 * math.pi / 30 * rr / (180 / 3.6)
        spec['battery'] = dict(capacity_kWh=c, nominal_V=V, chemistry=ch, soc_window=[0.15, 0.95] if plug else [0.3, 0.8])
        if plug:
            spec['battery']['max_power_kW'] = round(min(hw['mot'] / 0.9, 9 * c), 1)
    else:
        return None, None, f'unknown kind {kind}'
    A, B, C = _num(r.A_lbf), _num(r.B_lbf_mph), _num(r.C_lbf_mph2)
    if A is None or C is None or B is None:
        return None, None, 'no target road load'
    spec = spec_from_abc(spec, A, B, C, mass_kg=mass)
    return spec, kind, group_of(kind, hw)


CYCLES = (('FTP', 'FTP_mpg'), ('HWY', 'HWY_mpg'), ('US06', 'US06_mpg'), ('SC03', 'SC03_mpg'), ('COLD', 'FTP_cold_mpg'))


def epa_results(r):
    out = {}
    for k, c in CYCLES:
        x = _num(r[c])
        if x is not None and 3 < x < 200:
            out[k] = x
    return out
