"""The trip-energy law's REAL exam. Run ONCE, after results/road_ratchet.json is final. Chooses nothing.

Exam: RoadData's frozen exam splits, scored on exactly the units EnergyNet v2.1's exam scored (so its row is paired):
  - VED exam trips of petrol cars, hybrids and the electric cars (EnergyNet's exam2_ved_v2 frame: label_ok, above its
    per-trip floors; plug-ins left out), truth = RoadData's per-second label summed over valid seconds, per km;
  - EPA exam test cars (gasoline petrol, hybrids, battery-electric; EnergyNet's exam2_epa_v2 frame), one row per vehicle
    and test (FTP, HWY, US06, 20 F cold FTP, SC03; BEV city and highway), truth = the published test value as tank or DC
    energy per km.
Metric: median absolute % error (MdAPE), per stratum.
Rows: the released law; every rung (after the fact); RoadSim's own law (the knee of the last frozen set's front);
road-load energy / eta with one eta per powertrain fitted on dev (textbook, tuned on dev) and fitted on RoadSim;
EnergyNet v2.1 (the collection's network, its exam predictions); RoadData's classical VED baselines (window label x
distance, MOVES-like VSP) and its class median; dumb: the dev median energy per km of the powertrain group.
"""
import os, sys, json, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import lawratchet as lr
import law_road as L
from _paths import ROOT, other_work

OUT = f'{ROOT}/results/exam_road.json'


def mets(y, p, sig=None):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0)
    if ok.sum() < 3:
        return {'n': int(ok.sum())}
    y, p = y[ok], p[ok]
    m = {'n': int(ok.sum()), 'mdape': float(np.median(np.abs(p / y - 1)) * 100), 'bias_median_pct': float(np.median(p / y - 1) * 100),
         'mae': float(np.mean(np.abs(p - y))), 'rho': float(spearmanr(y, p)[0]), 'r2': lr.r2(y, p)}
    if sig is not None:
        m['cov68'] = float(np.mean(np.abs(p - y) <= (sig[ok] if np.ndim(sig) else sig)))
    return m


def exam_frames(dry=False):
    W = other_work('energynet')
    if dry:
        v = L.ved_rows('dev'); e = L.epa_rows('dev')
        for df in (v, e):
            df['en'] = df['b_label'] = df['b_vsp'] = df['b_class'] = np.nan
        return v, e, {}
    en_v = pd.read_parquet(f'{W}/data/exam2_ved_v2.parquet')
    en_v = en_v[en_v.group.isin(L.GROUPS)].set_index('trip_id')
    v = L.ved_rows('exam', trip_ids=set(en_v.index))
    j = en_v.loc[v.id]
    # EnergyNet's frame: y and pred per trip in its unit; conv turns the unit into MJ. Check the truth agrees.
    v['E_en'] = (j.y.values * j.conv.values) / v.dist_km.values
    chk = {'ved_truth_max_rel_diff': float(np.nanmax(np.abs(v.E_en / v.E - 1))), 'ved_trips_en': int(len(en_v)), 'ved_trips_law': int(len(v))}
    v['en'] = j.pred.values * j.conv.values / v.dist_km.values
    for b in ('b_label', 'b_vsp', 'b_class'):
        v[b] = j[b].values / v.dist_km.values          # already in MJ in EnergyNet's frame (x conv)
    en_e = pd.read_parquet(f'{W}/data/exam2_epa_v2.parquet').set_index(['vehicle_id', 'test'])
    e = L.epa_rows('exam')
    key = pd.MultiIndex.from_arrays([e.vehicle_id, e.test])
    have = key.isin(en_e.index)
    e = e[have].reset_index(drop=True); key = key[have]
    jj = en_e.loc[key]
    e['en'] = e.E.values * jj.pred.values / jj.y.values          # EnergyNet's ratio to its own truth, on our scale
    e['b_label'] = e['b_vsp'] = e['b_class'] = np.nan
    chk.update(epa_rows_en=int(len(en_e)), epa_rows_law=int(len(e)))
    return v, e, chk


def main(dry=False):
    global OUT
    if dry:
        OUT = OUT.replace('exam_', 'dryrun_')   # code check on DEV units only; never the exam
    if os.path.exists(OUT) and not dry:
        sys.exit(f'{OUT} exists: the exam runs once.')
    dry or open(f'{ROOT}/logs/exam_road.started', 'w').write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    R = json.load(open(f'{ROOT}/results/road_ratchet.json'))
    rel = next(r for r in R['rungs'] if r['id'] == R['released'])
    kn = lr.knee(R)
    dev = L.load_dev()
    eta_dev = L.fit_eta(dev)
    tr = L.load_bank(L.VERSIONS[-1], 'train')
    eta_sim = L.fit_eta(tr)
    dev_med = {g: float(np.median(dev.E[dev.group == g])) for g in L.GROUPS}
    v, e, chk = exam_frames(dry)
    res = {'released': rel['id'], 'expr': rel['expr'], 'knee': kn['id'], 'sigma': R['bars']['sigma'], 'eta_dev': eta_dev,
           'eta_sim': eta_sim, 'dev_median': dev_med, 'checks': chk, 'strata': {}, 'strata_rungs': {}, 'by_test': {}}
    names = {'law': f'LawFinder law ({rel["id"]})', 'sim_law': f"RoadSim's law ({kn['id']})",
             'roadload_dev': 'Road-load energy / eta, eta per powertrain fitted on dev',
             'roadload_sim': 'Road-load energy / eta, eta per powertrain fitted on RoadSim',
             'energynet': 'EnergyNet v2.1 (network; its exam predictions)', 'b_label': 'Window label x distance (RoadData)',
             'b_vsp': 'MOVES-like VSP bins (RoadData)', 'b_class': 'Class median (RoadData)', 'dumb': 'Dev median per powertrain'}
    pts = []
    for src, df in (('VED', v), ('EPA', e)):
        X = L.X_of(df); y = df.E.values
        P = {'law': lr.law_eval(rel, X), 'sim_law': lr.law_eval(kn, X), 'roadload_dev': L.road_load(df, eta_dev),
             'roadload_sim': L.road_load(df, eta_sim), 'energynet': df.en.values, 'b_label': df.b_label.values,
             'b_vsp': df.b_vsp.values, 'b_class': df.b_class.values, 'dumb': df.group.map(dev_med).values}
        for g in L.GROUPS + ('all',):
            m = (df.group == g).values if g != 'all' else np.ones(len(df), bool)
            if m.sum() < 3:
                continue
            k = f'{src}_{g}'
            res['strata'][k] = {n: {'name': names[n], **mets(y[m], p[m], R['bars']['sigma'] if n == 'law' else None)} for n, p in P.items()}
            res['strata_rungs'][k] = {r['id']: mets(y[m], lr.law_eval(r, X[m])) for r in R['rungs']}
        if src == 'EPA':
            for (g, t), d in df.groupby(['group', 'test']):
                m = df.index.isin(d.index)
                res['by_test'][f'{g}_{t}'] = {n: mets(y[m], p[m]) for n, p in P.items() if n in ('law', 'roadload_dev', 'energynet', 'dumb')}
        out = df.copy()
        for n, p in P.items():
            out[f'pred_{n}'] = p
        pts.append(out)
    # the headline block (what the page and the release read): every VED exam trip, the three powertrains pooled
    res['rows'] = {n: v for n, v in res['strata']['VED_all'].items() if v.get('n', 0) >= 3}
    res['rungs'] = res['strata_rungs']['VED_all']
    dry or pd.concat(pts, ignore_index=True).to_parquet(f'{ROOT}/results/exam_road_points.parquet')
    json.dump(res, open(OUT, 'w'), indent=1, default=str)
    for k, rows in res['strata'].items():
        print(f'== {k}')
        for n, m in rows.items():
            if m['n'] >= 3:
                print(f"  {m['name']:60s} n={m['n']:6d} MdAPE={m['mdape']:5.1f}% bias={m['bias_median_pct']:+5.1f}%"
                      + (f" cov68={m['cov68']:.2f}" if 'cov68' in m else ''))


if __name__ == '__main__':
    main(dry='--dry-dev' in sys.argv)
