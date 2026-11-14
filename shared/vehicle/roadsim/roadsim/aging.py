"""Battery aging for RoadSim (physics v2, option `batt_age`): state of health from calendar time and charge throughput.

    soh(chem, years, efc, T_C=25, soc_mean=0.6, dod=0.6) -> {'capacity': Q/Q_new, 'resistance': R/R_new}

Semi-empirical pack-level laws (cell data, so a real pack with better cells or thermal management may age slower; the
`cal_scale` / `cyc_scale` factors are for the calibration pass):
- NMC/NCA/LMO: Schmalstieg et al. 2014, J. Power Sources 257, 325 ("holistic aging model", NMC 18650 cells):
  calendar  alpha_cap = (7.543 V - 23.75) 1e6 exp(-6976 / T), loss = alpha_cap t^0.75   (t in days)
            alpha_res = (5.270 V - 16.32) 1e5 exp(-5986 / T), growth = alpha_res t^0.75
  cycling   beta_cap = 7.348e-3 (V - 3.667)^2 + 7.60e-4 + 4.081e-3 DOD, loss = beta_cap sqrt(Q_Ah)
            beta_res = 2.153e-4 (V - 3.725)^2 - 1.521e-5 + 2.798e-4 DOD, growth = beta_res Q_Ah
  with V the cell voltage at the mean SOC and Q_Ah the charge throughput of their 2.05 Ah cell (charge + discharge).
- LFP: cycling from Wang et al. 2011, J. Power Sources 196, 3942 (A123 2.3 Ah): loss % = B exp(-(31700 - 370.3 C) / RT)
  Ah^0.55 (B = 30330 at C/2); calendar ~1.2 % sqrt(years) at 25 C, Arrhenius 50 kJ/mol, more at high SOC (Naumann et al.
  2018, J. Energy Storage 17, 153: order of magnitude). Resistance growth ~2x the relative capacity loss.
- NiMH: ~2 %/year calendar + 1e-5 per equivalent full cycle; resistance growth 3x the capacity loss.
CellSense (models/cellsense) is the detailed cell-level reference (SEI ~ sqrt(t), lost graphite, plating knee); this module
is the pack-level stand-in a vehicle simulation needs. Capacity knees (end of life) are not modelled.
"""
import math


def _v_nmc(soc):
    return 3.45 + 0.75 * soc                 # NMC/graphite cell OCV, rough linear fit 10-90 % SOC


CAL_SCALE = 0.5     # EV-grade cells and pack thermal management age slower than the lab cells: these two factors bring a
CYC_SCALE = 0.3     # typical NMC EV (25 C, 100 equivalent cycles a year) to ~2 %/year, the fleet average in telematics
                    # studies (Geotab 2020-24: ~1.8-2.3 %/year). Calibration may refit them.


def soh(chem='NMC', years=0.0, efc=0.0, T_C=25.0, soc_mean=0.6, dod=0.6, c_rate=0.5, cal_scale=None, cyc_scale=None):
    cal_scale = CAL_SCALE if cal_scale is None else cal_scale
    cyc_scale = CYC_SCALE if cyc_scale is None else cyc_scale
    T = T_C + 273.15
    t_d = max(years, 0.0) * 365.25
    if chem in ('NMC', 'NCA', 'LMO'):
        V = _v_nmc(soc_mean)
        a_c = max((7.543 * V - 23.75) * 1e6 * math.exp(-6976.0 / T), 0.0)
        a_r = max((5.270 * V - 16.32) * 1e5 * math.exp(-5986.0 / T), 0.0)
        Q = max(efc, 0.0) * 2.0 * 2.05
        b_c = max(7.348e-3 * (V - 3.667) ** 2 + 7.60e-4 + 4.081e-3 * dod, 0.0)
        b_r = max(2.153e-4 * (V - 3.725) ** 2 - 1.521e-5 + 2.798e-4 * dod, 0.0)
        cap = 1.0 - cal_scale * a_c * t_d ** 0.75 - cyc_scale * b_c * math.sqrt(Q)
        res = 1.0 + cal_scale * a_r * t_d ** 0.75 + cyc_scale * b_r * Q
    elif chem == 'LFP':
        Ah = max(efc, 0.0) * 2.0 * 2.3
        cyc = 30330.0 * math.exp(-(31700.0 - 370.3 * c_rate) / (8.314 * T)) * Ah ** 0.55 / 100.0
        cal = 0.012 * math.sqrt(max(years, 0.0)) * math.exp(-50e3 / 8.314 * (1.0 / T - 1.0 / 298.15)) * (1.0 + 0.5 * (soc_mean - 0.5))
        loss = cal_scale * cal + cyc_scale * cyc
        cap = 1.0 - loss; res = 1.0 + 2.0 * loss
    else:   # NiMH
        loss = cal_scale * 0.02 * max(years, 0.0) * math.exp(-40e3 / 8.314 * (1.0 / T - 1.0 / 298.15)) + cyc_scale * 1e-5 * max(efc, 0.0)
        cap = 1.0 - loss; res = 1.0 + 3.0 * loss
    return {'capacity': min(max(cap, 0.3), 1.0), 'resistance': max(res, 1.0)}


def efc(throughput_kWh, capacity_kWh):
    """equivalent full cycles from discharge energy throughput."""
    return throughput_kWh / max(capacity_kWh, 1e-9)


def apply(bat, opt):
    """set the battery's per-run state from the options: state of health, SOC window, pack-heater threshold (defaults
    restore the spec pack exactly)."""
    a = opt.get('batt_age')
    if a:
        s = soh(bat.chem, **a)
        cap, res = s['capacity'], s['resistance']
    else:
        cap, res = opt['batt_soh_capacity'], opt['batt_soh_resistance']
    bat.set_aging(cap, res, opt['batt_soc_window'], opt['batt_heat_T_on'])
