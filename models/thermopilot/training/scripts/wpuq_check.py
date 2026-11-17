"""Sanity check of the simulator against real heat-pump homes (WPuQ, Lower Saxony, 2019). REPORT ONLY: nothing
here is used to train, tune or choose ThermoPilot.

  wpuq_check.py signature          real vs simulated heating signature per home (generic twin), and a per-home
                                   'calibrated twin' whose envelope is scaled so its slope matches the real one
  wpuq_check.py rank WEIGHTS.npz   controllers on generic vs calibrated twins (Jan 2019): does the ranking hold?
"""
import os, sys, json, copy, pickle
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # training/: tp, _paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORK as ROOT, PROC   # data/ and results/ live in WORK; PROC = BuildingData's processed tables
import numpy as np
import pandas as pd
import tp
import buildingsim as bs
from tp import env
from tp.episode import prep_spec

OUT = os.path.join(ROOT, 'results', 'wpuq')


def real_daily():
    w = pd.read_parquet(os.path.join(PROC, 'other', 'wpuq_60min.parquet'), columns=['path', 'time_utc', 'P_TOT'])
    w = w[w.path.astype(str).str.endswith('HEATPUMP')]
    w['path'] = w.path.astype(str)
    piv = w.pivot_table(index='time_utc', columns='path', values='P_TOT') / 1000.0      # kW
    piv = piv[(piv.index >= '2019-01-01') & (piv.index < '2020-01-01')]
    return piv


def signature(daily_kWh, T):
    """Heating signature on days with daily mean T < 12 C: slope (kWh/day per K, positive = more when colder),
    and kWh/day at 0 C."""
    x = pd.concat([daily_kWh.rename('E'), T.rename('T')], axis=1).dropna()
    x = x[x['T'] < 12]
    if len(x) < 40:
        return None
    b, a = np.polyfit(x['T'], x['E'], 1)
    return dict(slope=float(-b), E0=float(a), n=int(len(x)))


def scale_env(spec, f):
    s = copy.deepcopy(spec)
    for k in ('wall_U', 'roof_U', 'window_U', 'ach50', 'infiltration_ach'):
        if s.get(k) is not None:
            s[k] = float(s[k]) * f
    g = s.get('geometry') or {}
    for k in ('floor_U', 'basement_U', 'slab_F'):
        if g.get(k) is not None:
            g[k] = float(g[k]) * f
    return s


def sim_daily(spec, wx, pref):
    s = prep_spec(spec, pref)
    out = bs.simulate(s, wx, '2019-01-01', '2020-01-01', dt='1h')
    ts = out['ts']
    hp = ts['heating_electricity_kW'] + ts['cooling_electricity_kW'] + ts['fans_electricity_kW']
    return hp


def cmd_signature():
    os.makedirs(OUT, exist_ok=True)
    cases = pickle.load(open(os.path.join(ROOT, 'data', 'cases', 'exam_wpuq.pkl'), 'rb'))
    piv = real_daily()
    wdf, meta = env.weather('wpuq')
    wx = (wdf, meta)
    Td = wdf['T_out'].tz_convert('Europe/Berlin').resample('D').mean()
    rows = []
    calib = {}
    for c in cases:
        path = [p for p in piv.columns if p.split('/')[1] == c['unit'].split('_')[1]][0]
        hr = piv[path]
        hr_loc = hr.copy(); hr_loc.index = hr_loc.index.tz_convert('Europe/Berlin')
        cnt = hr_loc.resample('D').count()
        dr = hr_loc.resample('D').sum().where(cnt >= 22)
        sr = signature(dr, Td)
        if sr is None:
            rows.append(dict(unit=c['unit'], ok=False)); continue
        hp = sim_daily(c['spec_true'], wx, c['pref'])
        hp.index = hp.index.tz_convert('Europe/Berlin')
        ds = hp.resample('D').sum()
        ss = signature(ds, Td)
        f = 1.0
        spec = c['spec_true']
        for it in range(3):       # fixed-point: slope ~ proportional to UA
            f *= float(np.clip(sr['slope'] / max(ss['slope'] if it == 0 else sc['slope'], 1e-3), 0.3, 3.0)) ** 0.9
            f = float(np.clip(f, 0.2, 5.0))
            hpc = sim_daily(scale_env(spec, f), wx, c['pref'])
            hpc.index = hpc.index.tz_convert('Europe/Berlin')
            sc = signature(hpc.resample('D').sum(), Td)
        # January hourly statistics (local)
        jan = lambda s: s[(s.index.month == 1)]
        hj, sj, cj = jan(hr_loc).dropna(), jan(hp), jan(hpc)
        def hstats(s):
            prof = s.groupby(s.index.hour).mean()
            return dict(mean_kW=float(s.mean()), p95_kW=float(s.quantile(0.95)), off_frac=float((s < 0.05).mean()),
                        profile=(prof / max(prof.mean(), 1e-9)).round(3).tolist())
        r = dict(unit=c['unit'], ok=True, path=path, real=sr, sim=ss, calib=sc, f_env=f, real_jan=hstats(hj),
                 sim_jan=hstats(sj), calib_jan=hstats(cj),
                 profile_r_sim=float(np.corrcoef(hstats(hj)['profile'], hstats(sj)['profile'])[0, 1]),
                 real_annual_kWh=float(hr.sum() * (8760 / max(hr.count(), 1))), sim_annual_kWh=float(hp.sum()),
                 calib_annual_kWh=float(hpc.sum()))
        rows.append(r)
        calib[c['unit']] = f
        print(c['unit'], 'real slope %.2f E0 %.1f | sim %.2f %.1f | calib f %.2f -> %.2f %.1f' % (
            sr['slope'], sr['E0'], ss['slope'], ss['E0'], f, sc['slope'], sc['E0']), flush=True)
    ok = [r for r in rows if r.get('ok')]
    summ = dict(n=len(ok),
                slope_ratio_sim_real_median=float(np.median([r['sim']['slope'] / r['real']['slope'] for r in ok])),
                slope_ratio_iqr=[float(np.percentile([r['sim']['slope'] / r['real']['slope'] for r in ok], q)) for q in (25, 75)],
                E0_ratio_median=float(np.median([r['sim']['E0'] / r['real']['E0'] for r in ok])),
                f_env_median=float(np.median([r['f_env'] for r in ok])),
                calib_E0_ratio_median=float(np.median([r['calib']['E0'] / r['real']['E0'] for r in ok])),
                jan_off_frac_real=float(np.median([r['real_jan']['off_frac'] for r in ok])),
                jan_off_frac_sim=float(np.median([r['sim_jan']['off_frac'] for r in ok])),
                jan_peak_to_mean_real=float(np.median([r['real_jan']['p95_kW'] / max(r['real_jan']['mean_kW'], 1e-6) for r in ok])),
                jan_peak_to_mean_sim=float(np.median([r['sim_jan']['p95_kW'] / max(r['sim_jan']['mean_kW'], 1e-6) for r in ok])),
                profile_r_median=float(np.median([r['profile_r_sim'] for r in ok])))
    json.dump(dict(summary=summ, homes=rows), open(os.path.join(OUT, 'signature.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))


def cmd_rank(wpath):
    from evaluate import run_baselines, thermopilot, clean
    from tp import policy as P
    from tp.episode import Episode
    from scipy.stats import kendalltau
    W = P.load(wpath)
    sig = json.load(open(os.path.join(OUT, 'signature.json')))
    fmap = {r['unit']: r['f_env'] for r in sig['homes'] if r.get('ok')}
    cases = pickle.load(open(os.path.join(ROOT, 'data', 'cases', 'exam_wpuq.pkl'), 'rb'))
    names = ['fixed', 'setback', 'setback_std', 'precool', 'mpc_c', 'oracle', 'thermopilot']
    f = os.path.join(OUT, 'rank.json')
    res = json.load(open(f)) if os.path.exists(f) else {}
    for c in cases:
        if c['unit'] not in fmap or c['unit'] in res:
            continue
        out = {}
        for variant in ('generic', 'calibrated'):
            true = c['spec_true'] if variant == 'generic' else scale_env(c['spec_true'], fmap[c['unit']])
            nom = c['spec_nom'] if variant == 'generic' else scale_env(c['spec_nom'], fmap[c['unit']])
            ep = Episode(true, nom, c['wsrc'], c['year'], c['month'], c['tariff'], c['pref'], seed=c['seed'])
            b = run_baselines(ep, ['fixed', 'setback', 'setback_std', 'precool', 'mpc_c', 'oracle'])
            m, _ = ep.run(thermopilot(ep, W, c).act)
            b['thermopilot'] = clean(m)
            out[variant] = {k: dict(J=v['J'], bill=v['bill'], DH=v['DH'], kWh=v['kWh'], peak_kW=v['peak_kW']) for k, v in b.items()}
        res[c['unit']] = out
        json.dump(res, open(f, 'w'), indent=1)
        print(c['unit'], ' '.join('%s %.0f/%.0f' % (k, out['generic'][k]['J'], out['calibrated'][k]['J']) for k in names), flush=True)
    taus, flips = [], 0
    sav = {k: dict(generic=[], calibrated=[]) for k in names}
    for u, o in res.items():
        a = [o['generic'][k]['J'] for k in names]; b = [o['calibrated'][k]['J'] for k in names]
        taus.append(kendalltau(a, b)[0])
        for k in names:
            for v in ('generic', 'calibrated'):
                sav[k][v].append(1 - o[v][k]['J'] / o[v]['fixed']['J'])
        if (o['generic']['thermopilot']['J'] < o['generic']['setback']['J']) != (o['calibrated']['thermopilot']['J'] < o['calibrated']['setback']['J']):
            flips += 1
    summ = dict(n=len(res), kendall_tau_median=float(np.nanmedian(taus)),
                tp_vs_setback_order_flips=flips,
                mean_J_saving_vs_fixed={k: {v: float(np.mean(x)) for v, x in d.items()} for k, d in sav.items()})
    json.dump(dict(summary=summ, homes=res), open(f, 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'signature':
        cmd_signature()
    else:
        cmd_rank(sys.argv[2])
