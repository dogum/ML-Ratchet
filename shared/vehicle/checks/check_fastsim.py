"""Check: cross-check against NREL FASTSim 3.1.0 (pip 'fastsim') on the same 47 EPA vehicles and cycles.

FASTSim is given the same test mass, the same road load (our A + B v + C v^2 least-squares fitted onto FASTSim's
a + c v^2 form at FASTSim's own air density), the same engine / motor / battery ratings and the same auxiliary load,
but keeps its own component models (efficiency-vs-power curves, constant transmission efficiency, its HEV controller
with built-in SOC balancing) from its shipped templates: 2012 Ford Fusion (conventional), 2016 Toyota Prius Two (HEV),
2022 Renault Zoe ZE50 (BEV). Both models run hot-start UDDS, HWFET and US06.
Variant B ('matched components'): FASTSim additionally gets efficiency curves derived from roadsim's own engine
(efficiency along its best-BSFC line vs power), motor (efficiency vs power at 35 % of max speed) and driveline, so the
remaining difference reflects model structure (gear/shift operation off the best-BSFC line, regen and blending, aux,
limits) rather than component data. The engine power ramp lag of the templates is set to 1 s (US06 otherwise fails). We report roadsim/FASTSim - 1 and each
model's error vs EPA (hot UDDS is compared with EPA FTP for fuel vehicles, so both carry the same cold-start offset).
"""
import os, sys, json, copy, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'roadsim'))
sys.path.insert(0, HERE)
import _opts
_opts.init()          # --options NAME|JSON (physics options for every simulate() call; default v1)
import fastsim as fsim
from roadsim import cycle
from roadsim.sim import get_vehicle
from roadsim.constants import G, MILE, GAL, FUELS, EPA_KWH_PER_GAL, KWH
from roadsim.powertrain import ETA_ALT, ETA_DCDC
from epa_procedures import hot_cycle, test_spec, ETA_CHARGE, ETA_BATT_EQ

V = json.load(open(os.path.join(HERE, 'epa_vehicles.json')))
TEMPL = {'conv': '2012_Ford_Fusion.yaml', 'hev': '2016_TOYOTA_Prius_Two.yaml', 'bev': '2022_Renault_Zoe_ZE50_R135.yaml'}
RHO_FS = 1.2


def fs_cycle(name):
    if name in ('udds', 'hwfet'):
        return fsim.Cycle.from_resource(name + '.csv')
    v = cycle(name)
    n = len(v)
    d = fsim.Cycle.from_resource('udds.csv').to_dict()
    d.update(time_seconds=list(map(float, range(n))), speed_meters_per_second=list(map(float, v)),
             dist_meters=list(np.concatenate([[0], np.cumsum(0.5 * (v[1:] + v[:-1]))]).astype(float)),
             grade=[0.0] * n, elev_meters=[d['init_elev_meters']] * n, pwr_max_chrg_watts=[0.0] * n,
             temp_amb_air_kelvin=[d['temp_amb_air_kelvin'][0]] * n, pwr_solar_load_watts=[0.0] * n)
    return fsim.Cycle.from_pydict(d)


def _set_curve(interp, xs, ys):
    interp['data']['grid'][0]['data'] = [float(x) for x in xs]
    interp['data']['grid'][0]['dim'] = [len(xs)]
    interp['data']['values']['data'] = [float(y) for y in ys]
    interp['data']['values']['dim'] = [len(ys)]


def matched_curves(veh, pt, grp):
    """variant B: give FASTSim efficiency curves derived from roadsim's own components."""
    if veh.engine is not None:
        e = veh.engine
        fr = np.array(pt['fc']['eff_interp_from_pwr_out']['data']['grid'][0]['data'])
        P = np.maximum(fr * e.P_max, 1.0)
        eff = np.array([p / e.ool_f(p) if e.ool_f(p) > 0 else 0.05 for p in P])
        eff[0] = max(eff[1] * 0.5, 0.05)
        _set_curve(pt['fc']['eff_interp_from_pwr_out'], fr, np.clip(eff, 0.05, 0.5))
        if grp == 'conv':
            c0, c1, eg = 1.6, 0.0025, 0.975
            pt['transmission']['eff_interp'] = float(eg * veh.eta_dl * 0.97)
    if 'em' in pt:
        m = [mm for r, mm in veh.motors.items() if r in ('MG2', 'P2', 'traction')][0]
        ach = pt['em']['eff_interp_achieved']
        fr = np.array(ach['data']['grid'][0]['data'])
        w = 0.35 * m.w_max
        T = np.minimum(np.maximum(fr * m.P_max / w, 1e-3), m.t_lim(w))
        Pm = T * w
        eff = Pm / (Pm + np.array([m.loss(w, t) for t in T]))
        eff[fr < 1e-6] = eff[fr >= 1e-6][0]
        eff = np.clip(eff, 0.3, 0.99)
        _set_curve(ach, fr, eff)
        _set_curve(pt['em']['eff_interp_at_max_input'], fr / eff, eff)
        if grp != 'conv':
            pt['transmission']['eff_interp'] = float(veh.eta_dl)


def fs_vehicle(spec, matched=False):
    veh = get_vehicle(test_spec(spec))
    arch = spec['architecture']
    grp = 'conv' if arch == 'conv' else ('bev' if arch == 'bev' else 'hev')
    d = fsim.Vehicle.from_resource(TEMPL[grp]).to_dict()
    m = veh.mass
    # road load: fit a + c v^2 to our A + B v + C v^2 over 0-35 m/s
    A = m * G * veh.Crr; B = A * veh.Crr_v; C = 0.5 * 1.1767 * veh.CdA
    vs = np.linspace(0, 35, 71)
    X = np.stack([np.ones_like(vs), vs ** 2], 1)
    a, c = np.linalg.lstsq(X, A + B * vs + C * vs ** 2, rcond=None)[0]
    ch = d['chassis']
    ch['wheel_rr_coef'] = float(a / (m * G))
    ch['frontal_area_square_meters'] = float(c / (0.5 * RHO_FS) / ch['drag_coef'])
    d['mass_kilograms'] = float(m)
    pt = d['pt_type'][list(d['pt_type'].keys())[0]]
    if 'fc' in pt:
        pt['fc']['pwr_ramp_lag_seconds'] = 1.0       # no artificial engine power ramp (would fail US06)
    if matched:
        matched_curves(veh, pt, grp)
    if grp == 'conv':
        pt['fc']['pwr_out_max_watts'] = veh.engine.P_max
        pt['fc']['pwr_out_max_init_watts'] = veh.engine.P_max / 2
        pt['alt_eff'] = 1.0
        d['pwr_aux_base_watts'] = veh.aux_base / ETA_ALT
    elif grp == 'hev':
        pt['fc']['pwr_out_max_watts'] = veh.engine.P_max
        pt['fc']['pwr_out_max_init_watts'] = veh.engine.P_max / 2
        em = max(mm.P_max for r, mm in veh.motors.items() if r in ('MG2', 'P2', 'traction'))
        pt['em']['pwr_out_max_watts'] = em
        pt['res']['energy_capacity_joules'] = veh.battery.E_nom
        d['pwr_aux_base_watts'] = veh.aux_base / ETA_DCDC
    else:
        pt['em']['pwr_out_max_watts'] = veh.motors['traction'].P_max
        pt['res']['energy_capacity_joules'] = veh.battery.E_nom
        d['pwr_aux_base_watts'] = veh.aux_base / ETA_DCDC
    return fsim.Vehicle.from_pydict(d), grp


def fs_run(fv, grp, name):
    sp = fsim.SimParams.default().to_dict(); sp['trace_miss_opts'] = 'Allow'
    sd = fsim.SimDrive(fv, fs_cycle(name), fsim.SimParams.from_pydict(sp))
    sd.walk()
    d = sd.to_dict()['veh']
    st = d['state']
    dist_mi = st['dist_meters'] / MILE
    pt = d['pt_type'][list(d['pt_type'].keys())[0]]
    met = bool(st.get('cyc_met_overall', True))
    if grp == 'bev':
        E = pt['res']['state']['energy_out_chemical_joules'] if 'energy_out_chemical_joules' in pt['res']['state'] else None
        if E is None:
            E = sum(v for k, v in pt['res']['state'].items() if k == 'energy_out_chemical_joules')
        kwh_mi = E / KWH / dist_mi / ETA_CHARGE
        return dict(MPGe=EPA_KWH_PER_GAL / kwh_mi, kWh_100mi_ac=100 * kwh_mi, cyc_met=met)
    Ef = pt['fc']['state']['energy_fuel_joules']
    Eb = 0.0
    if grp == 'hev':
        Eb = pt['res']['state'].get('energy_out_chemical_joules', 0.0)
    gal = (Ef + Eb / ETA_BATT_EQ) / (FUELS['gasoline']['lhv'] * FUELS['gasoline']['density'] / 1e3 * GAL)
    return dict(FE=dist_mi / gal, cyc_met=met)


def main():
    t0 = time.time()
    rows = []
    for v in V:
        spec = v['spec']; e = v['epa']; arch = spec['architecture']
        try:
            fv, grp = fs_vehicle(spec)
            fvm, _ = fs_vehicle(spec, matched=True)
        except Exception as ex:
            print('skip', v['name'], ex); continue
        row = dict(name=v['name'], architecture=arch, split=v['split'], roadsim={}, fastsim={}, epa={}, diff_pct={}, err_rs_pct={},
                   err_fs_pct={}, fastsim_matched={}, diff_matched_pct={})
        cycles = [('udds', 'UDDS'), ('hwfet', 'HWY'), ('us06', 'US06')]
        for cyc, key in cycles:
            try:
                f = fs_run(fv, grp, cyc)
            except Exception as ex:
                row.setdefault('fastsim_errors', {})[key] = str(ex)[:200]; continue
            r = hot_cycle(test_spec(spec), cyc)
            try:
                fm = fs_run(fvm, grp, cyc)
                fmv = fm['MPGe'] if grp == 'bev' else fm['FE']
            except Exception as ex:
                fmv = None
            if grp == 'bev':
                rs, fsv = r['MPGe'], f['MPGe']
                ref = e.get({'UDDS': 'CD_UDDS', 'HWY': 'CD_HWY'}.get(key, 'x'))
                if ref and v['bev_unit'] != 'MPGe':
                    ref = EPA_KWH_PER_GAL / (ref / 100.0)
            else:
                rs, fsv = r['FE'], f['FE']
                ref = e.get({'UDDS': 'FTP', 'HWY': 'HWY', 'US06': 'US06'}[key])
            row['roadsim'][key] = round(rs, 2); row['fastsim'][key] = round(fsv, 2)
            row['diff_pct'][key] = round(100 * (rs / fsv - 1), 2)
            if fmv:
                row['fastsim_matched'][key] = round(fmv, 2); row['diff_matched_pct'][key] = round(100 * (rs / fmv - 1), 2)
            row['fastsim_cyc_met'] = f['cyc_met']
            if ref:
                row['epa'][key] = ref
                row['err_rs_pct'][key] = round(100 * (rs / ref - 1), 2); row['err_fs_pct'][key] = round(100 * (fsv / ref - 1), 2)
        rows.append(row)
        print(f"{arch:16s} {v['name'][:32]:32s} diff " + ' '.join(f"{k} {x:+.1f}%" for k, x in row['diff_pct'].items())
              + ' | FS vs EPA ' + ' '.join(f"{k} {x:+.1f}%" for k, x in row['err_fs_pct'].items()), flush=True)

    def st(x):
        x = np.asarray(x)
        return dict(n=int(len(x)), median_pct=round(float(np.median(x)), 2), p25_pct=round(float(np.percentile(x, 25)), 2),
                    p75_pct=round(float(np.percentile(x, 75)), 2), mean_abs_pct=round(float(np.mean(np.abs(x))), 2)) if len(x) else None
    summ = {}
    for g, f in {'conv': lambda a: a == 'conv', 'hev': lambda a: a.startswith('hev'), 'bev': lambda a: a == 'bev'}.items():
        rr = [r for r in rows if f(r['architecture'])]
        summ[g] = {}
        for key in ('UDDS', 'HWY', 'US06'):
            d = [r['diff_pct'][key] for r in rr if key in r['diff_pct']]
            ers = [r['err_rs_pct'][key] for r in rr if key in r['err_rs_pct'] and r['split'] == 'heldout']
            efs = [r['err_fs_pct'][key] for r in rr if key in r['err_fs_pct'] and r['split'] == 'heldout']
            dm = [r['diff_matched_pct'][key] for r in rr if key in r['diff_matched_pct']]
            summ[g][key] = dict(roadsim_vs_fastsim=st(d), roadsim_vs_fastsim_matched_components=st(dm),
                                roadsim_vs_epa_heldout=st(ers), fastsim_vs_epa_heldout=st(efs))
    out = dict(check='cross-check vs NREL FASTSim 3.1.0', fastsim_version=fsim.__version__, method=__doc__.strip(),
               note='UDDS row compares hot-start UDDS with EPA FTP-75 (cold start) for fuel vehicles; both models carry that offset.',
               summary=summ, vehicles=rows, wall_s=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(HERE, _opts.out('fastsim_crosscheck.json')), 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
