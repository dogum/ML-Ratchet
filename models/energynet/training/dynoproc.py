"""EPA test conditions for the bank's SC03 and 20 F cold-FTP dynamometer draws (rung v2-r14, 15 Nov 2026).

Up to v2.0 the bank (bank.py) ran these two cycles with HVAC on but otherwise like any dyno draw: no sun, the cabin
starting at 21 C (warm_temps), RoadSim's default HVAC settings (set point 21.5 C; 80% recirculated air when cooling, 30%
when heating). The real tests run under 40 CFR 1066 conditions, which roadsim.epa emulates and RoadSim's own dev check
(shared/vehicle/checks/epa_dev_pool.json, physics v2) uses: SC03 median twin error -0.3% (n 47 dev vehicles), 20 F
cold FTP +2.6% (n 43). This module gives the bank the same conditions, from roadsim.epa's constants and soak model:

  SC03          35 C, 850 W/m2 solar, A/C maximum (recirculated air, automatic 72 F); a preconditioning drive over the
                first 505 s of the UDDS under the lamps, a 10-minute soak with the engine off, cabin air and interior at
                45 C at the start (roadsim.epa.sc03).
  FTP_cold_20F  heat in automatic 72 F with outside air (roadsim.epa.COLD_HVAC); the start state stays the bank's own
                draw (cold soak at -6.7 C 60% of the time, else warm, which stands in for the hot bag).

Only the simulated run changes; the observed trip the network reads is the same as before."""
import copy
import numpy as np
from roadsim import simulate, epa
from roadsim.trips import cycle, make_trip
import physics as PHY

SC03_T, SC03_SOLAR = 35.0, 850.0
CYCLES = ('SC03', 'FTP_cold_20F')


def test_spec(spec):
    """dyno spec (no tyre-temperature effects, HVAC on) with the 72 F set point, as roadsim.epa.test_spec."""
    return epa.test_spec(PHY.dyno_spec(spec, hvac=True), hvac=True, T_set=epa.T_SET_72F)


def run(spec, tru, name, init_soc, init_temps, seed):
    """simulate one SC03 / FTP_cold_20F dyno draw under the test's conditions -> roadsim output (record=True)."""
    s = test_spec(spec)
    if name == 'SC03':
        opts = epa.merged(epa.SC03_HVAC, PHY.PHYSICS)
        T = SC03_T
        st = dict(engine=88.0, cabin=T, battery=T, tire_warm=1.0, driveline=80.0)
        pre = make_trip(cycle('udds')[:506], T_amb=T, rho_air=epa.RHO_REF, solar=SC03_SOLAR)
        op = simulate(s, pre, options=opts, init_soc=init_soc, init_temps=st, record=False, check=False)
        fs = epa.soak(s, op['final_state'], T, 600, epa.SC03_HVAC, PHY.PHYSICS)
        fs['cabin'] = fs['cabin_mass'] = epa.SC03_CABIN_SOAK_C
        trip = dict(tru); trip['solar'] = np.full(len(tru['v']), SC03_SOLAR, np.float32)
        return simulate(s, trip, options=opts, init_soc=init_soc, init_temps=fs, record=True, rng=seed)
    if name == 'FTP_cold_20F':
        opts = epa.merged(epa.COLD_HVAC, PHY.PHYSICS)
        return simulate(s, tru, options=opts, init_soc=init_soc, init_temps=init_temps, record=True, rng=seed)
    raise ValueError(name)
