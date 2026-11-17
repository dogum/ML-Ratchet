"""Twin tests on the 20 dev runs: simulate each dev unit's day with BuildingSim (same outdoor temperature trace,
same operating hours, same fault where BuildingSim has it) and compare the twin's trend log and FaultFinder's
reading with the real one. Usage: python3 twin.py [model_name] [calib.json]
Writes twin/<tag>_stats.json and twin/<tag>_dev_twins.png."""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, RI, RAW, sim_raw, to_day, load_real_runs, EXAM2CLASS  # noqa: E402
import buildingsim as bs  # noqa: E402

SITE = {'Iowa Energy Resource Station (ERS)': 'Chicago', 'LBNL FLEXLAB X3A (Berkeley)': 'San Francisco',
        'ORNL Flexible Research Platform 2 (Oak Ridge)': 'Baltimore',
        'Nesbitt Hall, Drexel University (Philadelphia)': 'New York'}
FAULT = {'hc_valve_stuck': lambda s: dict(type='heating_valve_stuck', position=_pct(s)),
         'hc_valve_leak': lambda s: dict(type='heating_valve_leak', leak=_pct(s)),
         'cc_valve_stuck': lambda s: dict(type='cooling_valve_stuck', position=_pct(s)),
         'oa_damper_stuck': lambda s: dict(type='oa_damper_stuck', position=_pct(s))}


def _pct(s):
    v = float(str(s).replace('%', ''))
    return v / 100.0 if v > 1 else v


def city_of(name):
    for c in bs.weather.EPW_CLIMATE_ZONE:
        if name.replace(' ', '.') in c or name in c:
            return c
    return [c for c in bs.weather.EPW_CLIMATE_ZONE if name.split()[0] in c][0]


def twin_spec(row, x, calib=None):
    unit = row.unit
    on = np.nan_to_num(x[RI['fan_on']]) > 0.5
    occ = x[RI['occ']]
    flag = np.nan_to_num(occ) > 0.5 if np.isfinite(occ).any() else on
    hrs = np.where(flag)[0] / 4.0
    open_h, close_h = (float(hrs.min()), float(hrs.max()) + 0.25) if len(hrs) else (7.0, 18.0)
    if unit == 'RTU':
        system, area = 'packaged_rtu', 900.0
    else:
        system, area = 'vav_chiller_boiler', 1500.0 if 'ERS' in row.facility or 'FLEX' in row.facility else 4000.0
    spec = dict(primary_use='office', floor_area_m2=area, n_floors=1, year_built=2005, system=system,
                occupancy_schedule=dict(type='commercial', weekday=[open_h, close_h], saturday=[open_h, close_h],
                                        sunday=[open_h, close_h], early_start=0.0),
                hvac=dict(cooling_source='chiller'))
    sp = x[RI['T_sa_sp']]
    if np.isfinite(sp).any() and system != 'packaged_rtu':
        spec['hvac']['sat_cool'] = float(np.nanmin(sp[on])) if on.any() and np.isfinite(sp[on]).any() else 12.8
        spec['hvac']['sat_reset'] = False
    if calib and calib.get('twin_min_oa'):
        spec['ventilation'] = dict(L_s_m2=calib['twin_min_oa'])
    return spec


def run_twin(row, x, fault=True, calib=None):
    city = city_of(SITE[row.facility])
    w, meta = bs.read_epw(bs.epw_path(city))
    tz = meta['tz']
    day = pd.Timestamp(row.date)
    tday = pd.Timestamp(f'2019-{day.month:02d}-{day.day:02d}')
    Toa = x[RI['T_oa']]
    w = w.copy()
    if np.isfinite(Toa).any():
        hourly = np.nanmean(np.where(np.isfinite(Toa), Toa, np.nan).reshape(24, 4), axis=1)
        hourly = pd.Series(hourly).interpolate(limit_direction='both').to_numpy()
        idx = pd.date_range(tday, periods=24, freq='h').tz_localize(tz).tz_convert('UTC')
        sel = w.index.isin(idx)
        if sel.sum() == 24:
            w.loc[sel, 'T_out'] = hourly
            w.loc[sel, 'dewpoint'] = np.minimum(w.loc[sel, 'dewpoint'], hourly - 0.5)
    spec = twin_spec(row, x, calib)
    start = (tday - pd.Timedelta(days=2)).tz_localize(tz).tz_convert('UTC').tz_localize(None)
    end = (tday + pd.Timedelta(days=1)).tz_localize(tz).tz_convert('UTC').tz_localize(None)
    f = None
    if fault and row.fault_type in FAULT:
        try:
            f = [FAULT[row.fault_type](row.severity)]
        except ValueError:
            f = None
    out = bs.simulate(spec, (w, meta), str(start), str(end), dt='15min', faults=f)
    return to_day(sim_raw(out, tz), tday), f


def stats(x):
    on = np.nan_to_num(x[RI['fan_on']]) > 0.5
    g = lambda c: x[RI[c], on]
    s = {}
    with np.errstate(all='ignore'):
        s['SA-SP'] = float(np.nanmean(g('T_sa') - g('T_sa_sp')))
        s['SA-MA'] = float(np.nanmean(g('T_sa') - g('T_ma')))
        s['MA-RA'] = float(np.nanmean(g('T_ma') - g('T_ra')))
        s['damper'] = float(np.nanmean(g('damper_oa')))
        s['cc'] = float(np.nanmean(g('valve_cc')))
        s['hc'] = float(np.nanmean(g('valve_hc')))
        s['fan'] = float(np.nanmean(g('fan_speed')))
        s['on_h'] = float(on.sum() / 4)
    return s


def main(model_name=None, calib=None, tag='twin'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(f'{ROOT}/twin', exist_ok=True)
    runs = load_real_runs('dev')
    model = None
    if model_name:
        import torch
        from model import FFNet
        from features import featurize
        import evaluate as ev
        sd = torch.load(f'{ROOT}/models/{model_name}.pt')
        w = sd['c1.0.weight'].shape[0]
        model = FFNet(w=w)
        model.load_state_dict(sd)
    rows = []
    fig, axs = plt.subplots(len(runs), 1, figsize=(13, 2.6 * len(runs)))
    for i, (row, x) in enumerate(runs):
        xt, f = run_twin(row, x, True, calib)
        r = dict(run_id=row.run_id, fault_type=row.fault_type, twin_fault=f[0]['type'] if f else None,
                 real=stats(x), twin=stats(xt))
        if model is not None:
            # same channel availability as the real log
            xt2 = xt.copy()
            for c in RAW:
                if not np.isfinite(x[RI[c]]).any():
                    xt2[RI[c]] = np.nan
            lo, _, _ = ev.predict(model, np.stack([featurize(x), featurize(xt2)]))
            p = 1 / (1 + np.exp(-ev.det_logit(lo)))
            r['p_real'], r['p_twin'] = float(p[0]), float(p[1])
            from common import CLASSES
            r['cls_real'] = CLASSES[1 + int(lo[0, 1:].argmax())]
            r['cls_twin'] = CLASSES[1 + int(lo[1, 1:].argmax())]
        rows.append(r)
        ax = axs[i]
        h = np.arange(96) / 4
        for c, col in (('T_oa', 'C0'), ('T_ra', 'C1'), ('T_ma', 'C2'), ('T_sa', 'C3')):
            ax.plot(h, x[RI[c]], color=col, lw=1, label=c + ' real')
            ax.plot(h, xt[RI[c]], color=col, lw=1, ls='--')
        ax2 = ax.twinx()
        for c, col in (('damper_oa', 'C4'), ('valve_cc', 'C5'), ('valve_hc', 'C6')):
            ax2.plot(h, x[RI[c]], color=col, lw=0.8)
            ax2.plot(h, xt[RI[c]], color=col, lw=0.8, ls='--')
        ax.set_title(f"{row.run_id} {row.fault_type} twin={r['twin_fault']} " +
                     (f"p_real={r['p_real']:.2f} p_twin={r['p_twin']:.2f}" if model else ''), fontsize=8)
        print(r['run_id'], r['fault_type'], {k: (round(r['real'][k], 2), round(r['twin'][k], 2)) for k in r['real']},
              r.get('p_real'), r.get('p_twin'), r.get('cls_real'), r.get('cls_twin'), flush=True)
    plt.tight_layout()
    plt.savefig(f'{ROOT}/twin/{tag}_dev_twins.png', dpi=50)
    json.dump(rows, open(f'{ROOT}/twin/{tag}_stats.json', 'w'), indent=1)
    return rows


if __name__ == '__main__':
    mn = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != '-' else None
    cal = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    main(mn, cal, tag=sys.argv[3] if len(sys.argv) > 3 else 'twin')
