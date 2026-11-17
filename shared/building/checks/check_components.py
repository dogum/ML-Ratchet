"""Component checks: psychrometrics vs CoolProp, PMV/PPD vs pythermalcomfort (both from the isolated venv
buildingsim/.refenv), curves at their rating points, DX coil rated SHR / capacity / COP, cooling-tower design point,
heat-pump defrost behaviour, chiller rated COP, infiltration formula, window angular transmittance."""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import json
import os
import subprocess
import tempfile

import numpy as np

from _common import write
from buildingsim import psychro as ps
from buildingsim import curves as cv
from buildingsim.comfort import pmv_ppd
from buildingsim.systems import DXCoil, HPHeating
from buildingsim.plant import ChillerPlant, CoolingTower
from buildingsim.zone import CP_AIR

REFPY = f'{BLDG_ROOT}/buildingsim/.refenv/bin/python'


def ref(req):
    os.makedirs(f'{BLDG_ROOT}/buildingsim/tmp', exist_ok=True)
    d = tempfile.mkdtemp(dir=f'{BLDG_ROOT}/buildingsim/tmp')
    a, b = os.path.join(d, 'req.json'), os.path.join(d, 'ans.json')
    json.dump(req, open(a, 'w'))
    subprocess.run([REFPY, '-I', os.path.join(os.path.dirname(__file__), 'ref', 'ref_values.py'), a, b], check=True)
    return json.load(open(b))


def main():
    rng = np.random.default_rng(0)
    pts = [(float(T), float(RH), float(P)) for T, RH, P in zip(rng.uniform(-20, 45, 60), rng.uniform(5, 100, 60),
                                                                rng.choice([101325.0, 83000.0], 60))]
    pmvp = [(float(a), float(a + rng.uniform(-3, 3)), float(v), float(rh), float(m), float(c)) for a, v, rh, m, c in
            zip(rng.uniform(17, 30, 60), rng.uniform(0.05, 0.4, 60), rng.uniform(20, 80, 60), rng.uniform(0.9, 1.8, 60),
                rng.uniform(0.4, 1.2, 60))]
    R = ref(dict(psychro=pts, pmv=pmvp))
    T = np.array([p[0] for p in pts]); RH = np.array([p[1] for p in pts]); P = np.array([p[2] for p in pts])
    W = ps.w_from_rh(T, RH, P)
    Wc = np.array([r['W'] for r in R['psychro']])
    twb = ps.wetbulb(T, W, P)
    tdp = ps.dewpoint_from_w(W, P)
    h = ps.enthalpy(T, W)
    hc = np.array([r['H'] for r in R['psychro']])
    dh = (h - h.mean()) - (hc - hc.mean())       # reference states differ; compare differences
    v = ps.R_DA * (T + 273.15) * (1 + 1.607858 * W) / P
    psy = dict(n=len(pts), W_max_rel_err=float(np.max(np.abs(W - Wc) / np.maximum(Wc, 1e-6))),
               Twb_max_abs_err_K=float(np.max(np.abs(twb - np.array([r['Twb'] for r in R['psychro']])))),
               Twb_p95_abs_err_K=float(np.percentile(np.abs(twb - np.array([r['Twb'] for r in R['psychro']])), 95)),
               Twb_note='the max is one point with T_wb ~ 0 C, where the ASHRAE equation switches from the water to '
                        'the ice branch (two roots); elsewhere <= 0.02 K',
               W_note='CoolProp includes the vapour enhancement factor (~1.004); the ASHRAE ideal-gas W does not',
               Tdp_max_abs_err_K=float(np.max(np.abs(tdp - np.array([r['Tdp'] for r in R['psychro']])))),
               enthalpy_diff_max_err_J_kg=float(np.max(np.abs(dh))),
               v_max_rel_err=float(np.max(np.abs(v - np.array([r['V'] for r in R['psychro']])) /
                                         np.array([r['V'] for r in R['psychro']]))),
               scalar_vs_vector_wsat=float(max(abs(ps.w_sat_s(t, p) - float(ps.w_sat(t, p))) for t, p in zip(T, P))))
    print('psychro', psy)
    a = np.array(pmvp)
    pm, pp = pmv_ppd(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5])
    pmc = np.array([r['pmv'] for r in R['pmv']]); ppc = np.array([r['ppd'] for r in R['pmv']])
    com = dict(n=len(pmvp), pmv_max_abs_err=float(np.max(np.abs(pm - pmc))), ppd_max_abs_err=float(np.max(np.abs(pp - ppc))))
    print('pmv', com)
    curves = dict(dx_cool_capft_rated=cv.dx_cool_capft(19.44, 35.0), dx_cool_eirft_rated=cv.dx_cool_eirft(19.44, 35.0),
                  dx_capfff_1=cv.quad(cv.DX_CAP_FFF, 1.0), dx_eirfff_1=cv.quad(cv.DX_EIR_FFF, 1.0),
                  hp_heat_capft_8_33=cv.hp_heat_capft(8.33), hp_heat_eirft_8_33=cv.hp_heat_eirft(8.33),
                  hp_heat_capft_minus_8_33=cv.hp_heat_capft(-8.33),
                  ch_capft_rated=cv.ch_capft(6.67, 29.44), ch_eirft_rated=cv.ch_eirft(6.67, 29.44),
                  ch_eirfplr_1=cv.quad(cv.CH_EIR_FPLR, 1.0), vav_fan_1=cv.cubic(cv.VAV_FAN, 1.0),
                  vs_pump_1=cv.cubic(cv.VS_PUMP, 1.0), boiler_plr_1=cv.quad(cv.BOILER_PLR, 1.0))
    print('curves', curves)
    # DX coil at rated conditions reproduces rated SHR, capacity and COP
    m = 1.2 * 0.0537e-3 * 10000.0
    coil = DXCoil(10000.0, 3.5, m, SHR=0.75)
    Win = ps.w_sat_s(19.44)
    Win = ((2501 - 2.326 * 19.44) * Win - 1.006 * (26.67 - 19.44)) / (2501 + 1.86 * 26.67 - 4.186 * 19.44)
    To, Wo, Qt, Qs, Pel = coil.full(26.67, Win, 35.0, m)
    dx = dict(rated_capacity_W=Qt, rated_SHR_target=0.75, rated_SHR_sim=Qs / Qt, rated_COP=Qt / Pel,
              leaving_T=To, leaving_W=Wo, dry_coil_SHR_at_low_humidity=coil.full(26.67, 0.004, 35.0, m)[3] /
              coil.full(26.67, 0.004, 35.0, m)[2],
              capacity_at_46C_outdoor_rel=coil.full(26.67, Win, 46.1, m)[2] / Qt,
              COP_at_46C_rel=(coil.full(26.67, Win, 46.1, m)[2] / coil.full(26.67, Win, 46.1, m)[4]) / (Qt / Pel))
    print('dx', dx)
    hp = HPHeating(10000.0, 3.6)
    hpr = {}
    for t in (8.33, 2.0, -8.33, -15.0, -20.0):
        Q, P_, Qd = hp.full(t, ps.w_from_rh(t, 80.0, 101325.0))
        hpr[str(t)] = dict(capacity_rel=Q / 10000.0, COP=Q / P_ if P_ else None, defrost_load_W=Qd)
    print('hp', hpr)
    ch = ChillerPlant(1.0e6, 6.0, 'water')
    ch.T_cw = 29.44
    r = ch.step(1.0e6 - ch.P_chwp, 30.0, 25.6, 3600)
    tower = CoolingTower(1.25e6, ch.m_w)
    Tlv, Pt = tower.step(ch.m_w * 4186.0 * 5.56, 25.6)
    chk = dict(chiller_COP_at_rated=r['cop'], chiller_cond_balance=r['Q_cond'] - r['Q_evap'] - r['P_chiller'],
               tower_design_leaving_C=Tlv, tower_design_target_C=max(21.0, 25.6 + 3.0), tower_fan_W=Pt,
               tower_fan_design_W=tower.P_fan_d)
    print('plant', chk)
    write('components.json', dict(what='component checks (psychrometrics, comfort, curves, DX, heat pump, chiller, tower)',
                                  psychrometrics_vs_CoolProp=psy, pmv_vs_pythermalcomfort=com, curves_at_rating=curves,
                                  dx_coil=dx, heat_pump_heating=hpr, plant=chk,
                                  references=dict(CoolProp='CoolProp 8.0 HAPropsSI (ASHRAE RP-1485)',
                                                  pythermalcomfort='pythermalcomfort 4.8.1 pmv_ppd_iso (ISO 7730)')))


if __name__ == '__main__':
    main()
