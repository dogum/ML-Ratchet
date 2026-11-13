"""Classical baselines for the exam, every fitted quantity fitted on DEV only.
VED targets (q_label_ok trips): ICE/HEV trip fuel [g]; EV net battery energy [kWh]; PHEV total energy [MJ]
(fuel x 42.0 MJ/kg + battery kWh x 3.6). Predictions use only seconds with valid==1 (the label is summed there too).
B1 label: fueleconomy.gov label MPG (or kWh/100 mi) x distance, city/highway mix by trip mean speed.
B2 VSP: MOVES-like operating-mode (speed x VSP bin) mean rates per architecture from dev, scaled per vehicle by
   (displacement/2 L)^gamma with gamma and a constant fitted on dev vehicles.
B3 class median: dev median energy per km for (VED type, body class), falling back to VED type.
Also: EPA test-car exam baselines and HD baselines."""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, _VEH)
import realdata as rd
ROOT = _VEH + '/data'; OUT = f'{ROOT}/proc/baselines'; os.makedirs(OUT, exist_ok=True)
LHV = 42.0e-3      # MJ/g (E10 gasoline)
GAL = 3.785411784; MI = 1609.344
CITY_V, HWY_V = 21.2, 48.3   # mph: mean speeds of UDDS (FTP) and HWFET


def vsp(v, grade):
    a = np.gradient(v)
    return v * (1.1 * a + 9.81 * grade + 0.132) + 0.000302 * v ** 3, a


SPD = [0.447, 11.18, 22.35]                   # 1, 25, 50 mph
VSPB = [-np.inf, 0, 3, 6, 9, 12, 18, 24, 30, np.inf]


def opmode(v, grade):
    p, a = vsp(v, grade)
    om = np.full(len(v), -1)
    idle = v < SPD[0]
    brake = (~idle) & (a <= -0.894)            # <= -2 mph/s
    sb = np.digitize(v, SPD[1:])               # 0,1,2
    vb = np.digitize(p, VSPB[1:-1])            # 0..8
    om = np.where(idle, 0, np.where(brake, 1, 2 + sb * 9 + vb))
    return om                                   # 0..28


def target(meta, arr, ved_type):
    ok = arr['valid'] == 1
    if ved_type in ('ICE', 'HEV'):
        return np.where(ok, np.nan_to_num(arr['fuel_rate']), 0.0), 'g'
    pe = np.nan_to_num(arr['batt_current'] * arr['batt_voltage']) / 1000.0   # kW
    if ved_type == 'EV':
        return np.where(ok, pe / 3600.0, 0.0), 'kWh'                          # kWh per second
    return np.where(ok, np.nan_to_num(arr['fuel_rate']) * LHV + pe / 1000.0, 0.0), 'MJ'   # MJ per second


def collect(split):
    idx = rd.ved_index(); veh = rd.ved_vehicles()
    d = idx[(idx.split == split) & idx.q_label_ok]
    rows, secs = [], []
    for meta, arr in rd.load_trips(split, label_ok=True):
        vt = meta['ved_type']
        y_s, unit = target(meta, arr, vt)
        ok = arr['valid'] == 1
        om = opmode(arr['v'].astype(float), np.nan_to_num(arr['grade']).astype(float))
        cnt = np.bincount(om[ok], minlength=29)
        rows.append(dict(trip_id=meta['trip_id'], vehicle_id=meta['vehicle_id'], ved_type=vt, arch=meta['architecture'],
                         y=float(y_s.sum()), unit=unit, dist_m=float(np.sum(arr['v'][ok])), dist_all_m=meta['distance_m'],
                         dur_valid_s=int(ok.sum()), v_mean_mph=meta['distance_m'] / meta['duration_s'] / 0.44704,
                         T_amb=meta['T_amb_mean'], **{f'om{k}': int(c) for k, c in enumerate(cnt)},
                         **{f'y_om{k}': float(y_s[ok][om[ok] == k].sum()) for k in range(29)}))
    t = pd.DataFrame(rows).join(veh[['displacement_L', 'cls', 'label_city_mpg', 'label_hwy_mpg', 'label_city_kWh_per_100mi',
                                     'label_hwy_kWh_per_100mi', 'match_confidence']], on='vehicle_id')
    return t


def fit_vsp(dev):
    model = {}
    for vt, g in dev.groupby('ved_type'):
        n = g[[f'om{k}' for k in range(29)]].sum().values.astype(float)
        e = g[[f'y_om{k}' for k in range(29)]].sum().values
        rate = np.where(n > 0, e / np.maximum(n, 1), np.nan)
        # empty bins: nearest filled bin in the same speed band, else overall mean
        allm = e.sum() / n.sum()
        rate = np.where(np.isnan(rate), allm, rate)
        pred0 = g[[f'om{k}' for k in range(29)]].values @ rate
        gv = g.assign(p0=pred0).groupby('vehicle_id').agg(y=('y', 'sum'), p0=('p0', 'sum'), displ=('displacement_L', 'first'))
        gamma, c = 0.0, 0.0
        if vt in ('ICE', 'HEV') and gv.displ.notna().sum() >= 3 and gv.displ.nunique() > 1:
            X = np.log(gv.displ / 2.0); Y = np.log(gv.y / gv.p0)
            gamma, c = np.polyfit(X, Y, 1)
        model[vt] = dict(rate=rate.tolist(), gamma=float(gamma), c=float(c))
    return model


def pred_vsp(t, model):
    out = np.full(len(t), np.nan)
    for vt, m in model.items():
        s = t.ved_type == vt
        p0 = t.loc[s, [f'om{k}' for k in range(29)]].values @ np.array(m['rate'])
        sc = np.exp(m['c'] + m['gamma'] * np.log(t.loc[s, 'displacement_L'].fillna(2.0) / 2.0)) if vt in ('ICE', 'HEV') else 1.0
        out[s.values] = p0 * sc
    return out


def pred_label(t, phev_fe):
    mph = t.v_mean_mph.values
    w = np.clip((HWY_V - mph) / (HWY_V - CITY_V), 0, 1)
    mi = t.dist_m.values / MI
    gal = mi * (w / t.label_city_mpg.values + (1 - w) / t.label_hwy_mpg.values)
    fuel_g = gal * GAL * 745.0
    kwh = mi * (w * t.label_city_kWh_per_100mi.values + (1 - w) * t.label_hwy_kWh_per_100mi.values) / 100.0
    out = np.where(t.ved_type.isin(['ICE', 'HEV']), fuel_g,
                   np.where(t.ved_type == 'EV', kwh, phev_fe * kwh * 3.6 + (1 - phev_fe) * fuel_g * LHV))
    return out


def errors(t, col):
    r = {}
    for vt, g in t.groupby('ved_type'):
        g = g[(g.y > {'ICE': 1, 'HEV': 1, 'EV': 0.01, 'PHEV': 0.05}[vt]) & g[col].notna()]
        ape = np.abs(g[col] - g.y) / g.y
        gv = g.groupby('vehicle_id')[[col, 'y']].sum()
        vape = np.abs(gv[col] - gv.y) / gv.y
        r[vt] = dict(n_trips=int(len(g)), n_vehicles=int(len(gv)), trip_MdAPE=float(ape.median()), trip_MAPE=float(ape.mean()),
                     trip_bias=float(np.median((g[col] - g.y) / g.y)), vehicle_total_MdAPE=float(vape.median()), unit=g.unit.iloc[0])
    return r


def main():
    dev = collect('dev'); exam = collect('exam')
    model = fit_vsp(dev)
    # PHEV electric share for the label baseline (dev): fraction of PHEV energy that came from the battery
    ph = dev[dev.ved_type == 'PHEV']
    phev_fe = 0.5
    if len(ph):
        el = []
        for meta, arr in rd.load_trips('dev', arch='PHEV', label_ok=True):
            ok = arr['valid'] == 1
            pe = np.nansum((arr['batt_current'] * arr['batt_voltage'])[ok]) / 1e6
            ft = np.nansum(arr['fuel_rate'][ok]) * LHV
            el.append((pe, ft))
        el = np.array(el); phev_fe = float(np.clip(el[:, 0].sum() / el.sum(), 0, 1))
    # class medians (dev): energy per metre by (ved_type, cls) and ved_type
    dev['epm'] = dev.y / dev.dist_m
    m1 = dev.groupby(['ved_type', 'cls']).apply(lambda g: g.y.sum() / g.dist_m.sum()) if False else dev.groupby(['ved_type', 'cls']).epm.median()
    m2 = dev.groupby('ved_type').epm.median()
    for t in (dev, exam):
        t['pred_label'] = pred_label(t, phev_fe)
        t['pred_vsp'] = pred_vsp(t, model)
        key = list(zip(t.ved_type, t.cls))
        t['pred_class'] = [(m1.get(k, m2.get(k[0])) * dm) for k, dm in zip(key, t.dist_m)]
    keep = ['trip_id', 'vehicle_id', 'ved_type', 'arch', 'unit', 'y', 'dist_m', 'dur_valid_s', 'v_mean_mph', 'T_amb', 'displacement_L', 'cls',
            'match_confidence', 'pred_label', 'pred_vsp', 'pred_class']
    exam[keep].to_parquet(f'{OUT}/ved_exam.parquet', index=False)
    dev[keep].to_parquet(f'{OUT}/ved_dev.parquet', index=False)
    summary = {'VED_exam': {c: errors(exam, c) for c in ['pred_label', 'pred_vsp', 'pred_class']},
               'VED_dev_in_sample': {c: errors(dev, c) for c in ['pred_label', 'pred_vsp', 'pred_class']},
               'fitted_on_dev': {'vsp_model': model, 'phev_electric_energy_share': phev_fe,
                                 'class_median_energy_per_m': {f'{a}|{b}': float(v) for (a, b), v in m1.items()},
                                 'type_median_energy_per_m': {a: float(v) for a, v in m2.items()}},
               'notes': 'APE = |pred - y| / y per trip (trips with y below a small floor excluded); vehicle_total = per-vehicle sums.'}
    summary['EPA_exam'] = epa_baselines()
    summary['HD_exam'] = hd_baselines()
    json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1)
    for k in ['pred_label', 'pred_vsp', 'pred_class']:
        print(k, {vt: (r['n_trips'], round(r['trip_MdAPE'], 3), round(r['trip_MAPE'], 3), round(r['trip_bias'], 3), round(r['vehicle_total_MdAPE'], 3)) for vt, r in summary['VED_exam'][k].items()})
    print(json.dumps(summary['EPA_exam'], indent=1)); print(json.dumps(summary['HD_exam'], indent=1))


def epa_baselines():
    """EPA test cars: predict FTP and HWY fuel per mile (conv, hev) and CD_UDDS/CD_HWY kWh/mi (bev).
    (a) dev median per architecture; (b) log-linear regression on dev: log(y) ~ log ETW + log displ + log hp + log road-load force at 50 mph."""
    e = rd.epa_testcars()
    e = e.assign(F50=e.A_N + e.B_N_per_mps * 22.352 + e.C_N_per_mps2 * 22.352 ** 2)
    res = {}
    for arch, cyc, ycol in [('conv', 'FTP', 'FTP_mpg'), ('conv', 'HWY', 'HWY_mpg'), ('conv', 'US06', 'US06_mpg'),
                            ('hev', 'FTP', 'FTP_mpg'), ('hev', 'HWY', 'HWY_mpg'),
                            ('bev', 'CD_UDDS', 'CD_UDDS_kWh_per_mi'), ('bev', 'CD_HWY', 'CD_HWY_kWh_per_mi')]:
        d = e[(e.architecture == arch) & e[ycol].notna() & (e[ycol] > 0) & e.F50.gt(0) & e.test_mass_kg.gt(0)].copy()
        if arch != 'bev': d = d[d.displacement_L.gt(0) & d.rated_hp.gt(0) & d.fuel.eq('gasoline')]
        d['y'] = 1 / d[ycol] if ycol.endswith('mpg') else d[ycol]        # gal/mi or kWh/mi
        dv, ex = d[d.split == 'dev'], d[d.split == 'exam']
        if len(dv) < 10 or len(ex) < 10: continue
        Xc = ['test_mass_kg', 'F50'] + ([] if arch == 'bev' else ['displacement_L', 'rated_hp'])
        X = lambda q: np.c_[np.ones(len(q)), np.log(q[Xc].values.astype(float))]
        beta, *_ = np.linalg.lstsq(X(dv), np.log(dv.y.values), rcond=None)
        p_reg = np.exp(X(ex) @ beta); p_med = np.full(len(ex), dv.y.median())
        ape = lambda p: np.abs(p - ex.y.values) / ex.y.values
        res[f'{arch}_{cyc}'] = dict(target='gal/mi' if ycol.endswith('mpg') else 'kWh/mi', n_dev=len(dv), n_exam=len(ex),
                                    median_MdAPE=float(np.median(ape(p_med))), regression_MdAPE=float(np.median(ape(p_reg))),
                                    regression_MAPE=float(np.mean(ape(p_reg))), regression_features=Xc, beta=beta.tolist())
    return res


def hd_baselines():
    h = rd.hd_data(); res = {}
    L = h['ztbus_legs']; L = L[L.q_label_ok]
    dv, ex = L[L.split == 'dev'], L[L.split == 'exam']
    k = dv.elec_kWh.sum() / (dv.distance_m.sum() / 1000)
    p = k * ex.distance_m / 1000; ape = np.abs(p - ex.elec_kWh) / ex.elec_kWh
    res['ZTBus_legs_kWh'] = dict(method='dev kWh/km x distance', kWh_per_km_dev=float(k), n_exam=len(ex), MdAPE=float(ape.median()), MAPE=float(ape.mean()))
    M = h['ztbus_missions']; dvm, exm = M[M.split == 'dev'], M[M.split == 'exam']
    k2 = dvm.kWh.sum() / (dvm.drivenDistance.sum() / 1000)
    ape = np.abs(k2 * exm.drivenDistance / 1000 - exm.kWh) / exm.kWh
    res['ZTBus_missions_kWh'] = dict(method='dev kWh/km x distance', kWh_per_km_dev=float(k2), n_exam=len(exm), MdAPE=float(ape.median()), MAPE=float(ape.mean()))
    m = h['mersin_trips']; m = m[m.q_label_ok]
    for fuel, g in m.groupby('fuel'):
        dv, ex = g[g.split == 'dev'], g[g.split == 'exam']
        p = np.full(len(ex), dv.fuel_g.median()); ape = np.abs(p - ex.fuel_g) / ex.fuel_g
        res[f'Mersin_{fuel}_trip_fuel'] = dict(method='dev median fuel per trip (fixed route)', n_exam=len(ex), MdAPE=float(np.median(ape)), MAPE=float(np.mean(ape)))
    t = h['icct_trucks']; t = t[t.cycle_available]
    dv = t[t.split == 'dev'].set_index('cycle').L_per_100km; ex = t[t.split == 'exam']
    ape = np.abs(ex.cycle.map(dv).values - ex.L_per_100km.values) / ex.L_per_100km.values
    res['ICCT_trucks_L_per_100km'] = dict(method='dev truck value on the same cycle', n_exam=len(ex), MdAPE=float(np.median(ape)), MAPE=float(np.mean(ape)))
    return res


if __name__ == '__main__':
    main()
