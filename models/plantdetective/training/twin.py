"""Twin test on the 40 dev units (dev years 2019 and 2021 only).

Each dev unit's twin gets the unit's own observed parameters (pmax, min stable load, heat-rate curve, ramp, minimum
up/down from its dev-year CAMPD record), is self-committed against the real ERCOT day-ahead hub price of the same
year (or run as a cogeneration unit if it is a CHP unit operated near-continuously), and goes through the same
emulators as the training bank.  Then the twin's operating pattern is compared with the real one.

usage: python twin.py <bank_version> [tag]
"""
from common import *
import json, zlib
import numpy as np, pandas as pd
from pdcore import truth, runs, model_input
from selfcommit import schedule, outages
from emulate import record, fuel_fn
from evaluate import real_unit, DEV_YEARS
import realdata, calib

STATS = ['starts_yr', 'med_up', 'min_up', 'min_dn', 'cf', 'pmin_frac', 'ramp_up', 'ramp_dn', 'hot', 'warm', 'cold',
         'first_h', 'second_h', 'load_p10', 'load_p50', 'load_p90', 'at_max', 'flat']


def pattern(rec):
    tr = truth(rec['op'], rec['P'], rec['HI'], rec['meas'])
    x = model_input(rec['op'], rec['P'])
    on = x > 0
    s, L, v = runs(on)
    pc = tr['pmax'] if np.isfinite(tr['pmax']) else max(x.max(), 1)
    st = s[v & (s > 0)]
    inter = on & np.r_[False, on[:-1]] & np.r_[on[1:], False]
    y = x[inter] / pc
    d = {k: tr.get(k, np.nan) for k in STATS[:11]}
    d['first_h'] = float(np.median(x[st] / pc)) if len(st) else np.nan
    st2 = st[st + 1 < len(x)]
    d['second_h'] = float(np.median(x[st2 + 1] / pc)) if len(st2) else np.nan
    if len(y) > 20:
        d['load_p10'], d['load_p50'], d['load_p90'] = np.percentile(y, [10, 50, 90])
        d['at_max'] = float(np.mean(y > 0.95))
        dy = np.diff(x)[(inter[1:] & inter[:-1])] / pc
        d['flat'] = float(np.mean(np.abs(dy) < 0.01)) if len(dy) else np.nan
    else:
        d.update(load_p10=np.nan, load_p50=np.nan, load_p90=np.nan, at_max=np.nan, flat=np.nan)
    return d


def twin_params(uid, rec):
    tr = truth(rec['op'], rec['P'], rec['HI'], rec['meas'])
    U = realdata.units().set_index('unit_id').loc[uid]
    tech = realdata.truth_tech(U)
    pmax = tr['pmax']; pf = tr['pmin_frac'] if np.isfinite(tr['pmin_frac']) else 0.5
    hf = tr['hr_full'] if np.isfinite(tr['hr_full']) else 10.0
    hm = tr['hr_min'] if np.isfinite(tr['hr_min']) else hf
    fp = tr.get('f_part', 0.5) if np.isfinite(tr.get('f_part', np.nan)) else 0.5
    beta = float(np.clip((hm / hf - 1) / max(1 / fp - 1, 1e-3), 0.0, 0.6)) if fp < 0.98 else 0.1
    on = (rec['op'] > 0) & (rec['P'] > 0)
    s_, L_, v_ = runs(on)
    up = L_[1:-1][v_[1:-1]]; dn = L_[1:-1][~v_[1:-1]]
    p10up = np.percentile(up, 10) if len(up) >= 3 else 1.0       # 10th percentile: robust to trips and tests
    p10dn = np.percentile(dn, 10) if len(dn) >= 3 else 1.0
    ctx = realdata.context(uid)
    cogen = bool(ctx.get('chp')) and (tr['starts_yr'] or 0) < 15 and (tr['cf'] or 0) > 0.4
    return dict(uid=uid, tech=tech, pmax=pmax, pmin=pf * pmax, hr_full=hf, beta=beta, curv=0.0, vom=3.0,
                start_heat={'coal_steam': 18, 'gas_cc': 6, 'gas_ct': 1, 'gas_steam': 15}[tech],
                min_up=float(max(1, round(p10up))), min_dn=float(max(1, round(p10dn))),
                ramp=float(max(tr['ramp_up'] if np.isfinite(tr['ramp_up']) else 0.3, 0.02) * pmax * 1.2),
                cogen=cogen, cf=tr['cf'], chp=bool(ctx.get('chp')))


def simulate_twin(tp, cal, seed, years=DEV_YEARS):
    from bank_dp import simulate_unit
    pr = realdata.prices()
    rng = np.random.default_rng(seed)
    recs = []
    u = dict(tp)
    u['start_mult'] = float(np.exp(rng.normal(*cal['start_mult'][tp['tech']])))
    u['fc_noise'] = float(rng.uniform(*cal['fc_noise']))
    u['efor'], u['planned'] = 0.04, 0.04
    u['kappa'] = float(rng.uniform(*cal['kappa'][tp['tech']]))
    rep = dict(chp=bool(tp.get('chp')), ct_only=False, n_share=1, s0=1.0, s1=0.0, phi1=0.0, phi2=0.0, aux=0.03)
    for y in years:
        p = pr[pr.index.year == y]
        bg = dict(price=p.price.ffill().bfill().values, gas=p.gas.values)
        on, Pd = simulate_unit(u, bg, rng, rep, cal, 'cogen' if tp['cogen'] else 'merchant')
        recs.append(record(u, on, Pd, p.index.dayofyear.values, rng, rep, cal['emu']))
    return {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}


def run(version, tag='', seeds=(0,), cal=None, verbose=True):
    cal = calib.get(version) if cal is None else cal
    U = realdata.units(); U = U[U.split == 'dev']
    rows = []
    for uid in U.unit_id:
        rec, t = real_unit('dev', uid)
        tt = pd.to_datetime(t, unit='s')
        m = np.isin(tt.year, DEV_YEARS)
        rec = {k: v[m] for k, v in rec.items()}
        if (rec['op'] > 0).sum() < 100 or not np.isfinite(truth(rec['op'], rec['P'])['pmax']):
            continue
        tp = twin_params(uid, rec)
        real = pattern(rec)
        for sd in seeds:
            tw = pattern(simulate_twin(tp, cal, 1000 * sd + zlib.crc32(uid.encode()) % 997))
            rows.append(dict(unit=uid, tech=tp['tech'], cogen=tp['cogen'], seed=sd,
                             **{f'real_{k}': real[k] for k in STATS}, **{f'twin_{k}': tw[k] for k in STATS}))
        if verbose: print(uid, tp['tech'], 'starts real/twin %.0f/%.0f' % (real['starts_yr'], tw['starts_yr']),
              'med_up %.0f/%.0f' % (real['med_up'], tw['med_up']), 'cf %.2f/%.2f' % (real['cf'], tw['cf']), flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(f'{ROOT}/out', exist_ok=True)
    df.to_csv(f'{ROOT}/out/twin_{version}{tag}.csv', index=False)
    return df


def summary(df):
    """Median log-ratio (twin / real) per technology for positive stats; median difference for fractions."""
    out = {}
    for tech, g in df.groupby('tech'):
        d = {}
        for k in STATS:
            a, b = g[f'real_{k}'].astype(float), g[f'twin_{k}'].astype(float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() == 0:
                continue
            if k in ('starts_yr', 'med_up', 'min_up', 'min_dn', 'ramp_up', 'ramp_dn'):
                d[k] = round(float(np.median(np.log((b[m] + 0.5) / (a[m] + 0.5)))), 2)
            else:
                d[k] = round(float(np.median(b[m] - a[m])), 3)
        out[tech] = d
    return out


def mismatch(df):
    """One number: mean absolute log-ratio / difference over the pattern statistics (lower = twins look real)."""
    s = summary(df)
    return float(np.mean([abs(v) for d in s.values() for v in d.values()]))


if __name__ == '__main__':
    df = run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '')
    s = summary(df)
    print(json.dumps(s, indent=1)); print('mismatch', mismatch(df))
    json.dump(dict(summary=s, mismatch=mismatch(df)), open(f'{ROOT}/out/twin_{sys.argv[1]}{sys.argv[2] if len(sys.argv) > 2 else ""}.json', 'w'), indent=1)
