"""BuildingDetective exam 2 (HEAPO): score v1.0 once, as split_exam2_heapo.json says; r10 hourly for information. 17 Nov 2026.

    python exam2_heapo.py build --check      # first: the processed homes must reproduce the frozen hashes
    python exam2_heapo_score.py              # -> training/exam2_heapo_results.json

v1.0 = the released weights/ (hourly, monthly, annual). r10 = the calibrated r10 hourly network in
work/models/r10_rel/ (kept by the ladder rule, not released): information only, hourly rows only.
Truth is re-read from the protocols (exam2_heapo.truth_table). Baselines are fitted on the 25 dev homes only.
Interval: 95 % percentile bootstrap over exam homes, paired, 2,000 resamples, seed 20261117, on the model's statistic
minus the best baseline's (best = better point value on the full exam row).
"""
from __future__ import annotations

import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths as _P  # noqa: E402
import exam2_heapo as H  # noqa: E402

sys.path.insert(0, _P.MODEL)
from buildingdetective import Detective  # noqa: E402
import features as F  # noqa: E402

OUT = os.path.join(HERE, 'exam2_heapo_results.json')
R10 = os.path.join(_P.WORK, 'models', 'r10_rel')
NB, SEED = 2000, H.SEED
MODES = ['hourly', 'monthly', 'annual']


def read_home(hid):
    d = pd.read_parquet(os.path.join(H.PROC, f'{hid}.parquet'))
    idx = pd.DatetimeIndex(pd.to_datetime(d.time_utc, utc=True))
    return idx, d.elec_kW.to_numpy(float), d.T_out.to_numpy(float), d.GHI.to_numpy(float)


def interval(diffs):
    ok = np.isfinite(diffs)
    return [float(x) for x in np.percentile(diffs[ok], [2.5, 97.5])], int(ok.sum())


def verdict(lo, hi, better):
    if better == 'lower':
        return 'win' if hi < 0 else 'loss' if lo > 0 else 'tie'
    return 'win' if lo > 0 else 'loss' if hi < 0 else 'tie'


def main():
    S = json.load(open(H.SPLIT, encoding='utf-8'))
    U = S['units']
    T = H.truth_table(zipfile.ZipFile(os.path.join(H.RAW, 'heapo_data.zip')))
    v10, r10 = Detective(), Detective(R10)
    assert set(r10.nets) == {'hourly'}
    rows = []
    for hid, u in sorted(U.items(), key=lambda kv: int(kv[0])):
        idx, elec, Tout, ghi = read_home(hid)
        h = H.hash_series(idx, [elec, Tout, ghi])
        if h != u['sha256']:
            raise SystemExit(f'home {hid}: processed hash {h} != frozen {u["sha256"]}')
        t = T.loc[int(hid)]
        rec = dict(home=int(hid), role=u['role'], area_row=u['area_row'], year_row=u['year_row'],
                   area_true=float(t.area_m2), year_true=pd.to_numeric(t.year_built, errors='coerce'),
                   building_type=t.building_type if isinstance(t.building_type, str) else None,
                   use_true=t.use if isinstance(t.use, str) else None,
                   kwh=float(np.nanmean(elec) * 8760))
        if u['role'] == 'exam':
            loc = idx.tz_convert(H.TZ)
            meters = {'elec': elec}
            d = np.asarray(loc.normalize().asi8)
            _, inv = np.unique(d, return_inverse=True)
            Td = np.bincount(inv, weights=np.nan_to_num(Tout, nan=np.nanmean(Tout))) / np.bincount(inv)
            hdd, cdd = float(np.maximum(F.HB - Td, 0).sum()), float(np.maximum(Td - F.HB, 0).sum())
            out = {'v10_hourly': v10.read_hourly(meters, Tout, ghi, loc, H.LAT),
                   'v10_monthly': v10.read_monthly_from_hourly(meters, Tout, loc, H.LAT),
                   'v10_annual': v10.read_annual({'elec': rec['kwh']}, hdd, cdd),
                   'r10_hourly': r10.read_hourly(meters, Tout, ghi, loc, H.LAT)}
            for k, r in out.items():
                rec[f'{k}_area'] = r['floor_area_m2'][0]
                rec[f'{k}_year'] = r['year_built'][0]
                rec[f'{k}_psingle'] = r['use']['residential_single']
                rec[f'{k}_pmulti'] = r['use']['residential_multi']
                rec[f'{k}_php'] = r['heat']['heat_pump']
                rec[f'{k}_hptop'] = max(r['heat'], key=r['heat'].get) == 'heat_pump'
        rows.append(rec)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(_P.WORK, 'exam2', 'heapo_scored_homes.csv'), index=False)
    dev, ex = D[D.role == 'dev'], D[D.role == 'exam'].reset_index(drop=True)

    # ---- baselines fitted on dev only
    da, dy = dev[dev.area_row], dev[dev.year_row]
    eui_all = float(np.median(da.kwh / da.area_true))
    eui_type = {k: float(np.median(g.kwh / g.area_true)) for k, g in da.groupby('building_type')}
    bA = np.polyfit(np.log10(da.kwh), np.log10(da.area_true), 1)
    bY = np.polyfit(np.log10(dy.kwh), dy.year_true.astype(float), 1)
    fitted = dict(dev_n=int(len(dev)), dev_area_n=int(len(da)), dev_year_n=int(len(dy)),
                  dev_median_kwh_per_m2=eui_all, dev_median_kwh_per_m2_by_type=eui_type,
                  area_line=dict(b=float(bA[0]), a=float(bA[1])), year_line=dict(b=float(bY[0]), a=float(bY[1])),
                  dev_median_area=float(np.median(da.area_true)), dev_median_year=float(np.median(dy.year_true)),
                  dev_majority_type=dev.use_true.mode().iloc[0])
    ex['b_area_eui_type'] = [k / eui_type[bt] if bt in eui_type else np.nan for k, bt in zip(ex.kwh, ex.building_type)]
    ex['b_area_eui_all'] = ex.kwh / eui_all
    ex['b_area_line'] = 10 ** np.polyval(bA, np.log10(ex.kwh))
    ex['b_area_median'] = fitted['dev_median_area']
    ex['b_year_median'] = fitted['dev_median_year']
    ex['b_year_line'] = np.polyval(bY, np.log10(ex.kwh))
    ex.to_csv(os.path.join(_P.WORK, 'exam2', 'heapo_scored_exam.csv'), index=False)

    rng = np.random.default_rng(SEED)
    res = dict(exam='BuildingDetective exam 2, HEAPO (Zurich heat-pump homes), scored once on 17 Nov 2026',
               spec=os.path.basename(H.SPLIT), split_sha256=S['split_sha256'], n_exam=int(len(ex)),
               models=dict(v10='BuildingDetective v1.0 (released weights/)',
                           r10='r10 hourly, calibrated (work/models/r10_rel): kept by the ladder rule but not released; '
                               'information only'),
               baselines_fitted_on_dev=fitted, interval=dict(resamples=NB, seed=SEED, unit='home', level=0.95),
               not_run_as_specified=[], rows={})
    n_other = int((~ex.building_type.isin(eui_type)).sum())
    res['not_run_as_specified'].append(
        f'Area baseline "annual elec / dev median kWh per m2 of the TRUE building type": all 25 dev homes are single-'
        f'family, so no dev kWh/m2 exists for the {n_other} exam homes of other types (multi-family, other, unknown). '
        'Where it is defined it equals the all-dev baseline. It is listed for the single-family homes only and not '
        'used to choose the best baseline.')

    def run_row(rid, sel, truth, preds, bases, metric, better, info=False):
        e = ex[sel].reset_index(drop=True)
        n = len(e)
        y = e[truth].to_numpy(float) if truth else None
        draws = [rng.integers(0, n, n) for _ in range(NB)]

        def m(col, ix):
            return metric(e, col, ix, y)
        allix = np.arange(n)
        bv = {b: m(b, allix) for b in bases if b != 'b_area_eui_type'}
        best = (min if better == 'lower' else max)(bv, key=bv.get)
        row = dict(n=n, better=better, baselines=bv, best_baseline=best)
        if 'b_area_eui_type' in bases:
            st = e.building_type.isin(eui_type).to_numpy()
            row['baseline_true_type_single_family_only'] = dict(n=int(st.sum()), value=m('b_area_eui_type', np.where(st)[0]))
        for p in preds:
            v = m(p, allix)
            diffs = np.array([m(p, ix) - m(best, ix) for ix in draws])
            (lo, hi), used = interval(diffs)
            row[p] = dict(value=v, minus_best=v - bv[best], interval=[lo, hi], resamples_used=used,
                          verdict='info only' if info else verdict(lo, hi, better))
        res['rows'][rid] = row

    def mdape(e, col, ix, y):
        return float(np.median(np.abs(e[col].to_numpy(float)[ix] - y[ix]) / y[ix]) * 100)

    def mae(e, col, ix, y):
        return float(np.mean(np.abs(e[col].to_numpy(float)[ix] - y[ix])))

    def acc(e, col, ix, y):
        tr = e.use_true.to_numpy()[ix]
        if col == 'b_type_majority':
            pr = np.full(len(ix), fitted['dev_majority_type'])
        else:
            pr = np.where(e[f'{col}_psingle'].to_numpy()[ix] >= e[f'{col}_pmulti'].to_numpy()[ix],
                          'residential_single', 'residential_multi')
        return float((pr == tr).mean())

    for mode in MODES:
        preds = [f'v10_{mode}'] + (['r10_hourly'] if mode == 'hourly' else [])
        run_row(f'area_mdape_{mode}', ex.area_row.to_numpy(bool), 'area_true', [f'{p}_area' for p in preds],
                ['b_area_eui_type', 'b_area_eui_all', 'b_area_line', 'b_area_median'], mdape, 'lower')
        run_row(f'year_mae_{mode}', ex.year_row.to_numpy(bool), 'year_true', [f'{p}_year' for p in preds],
                ['b_year_median', 'b_year_line'], mae, 'lower')
        run_row(f'type_acc_{mode}', ex.use_true.isin(['residential_single', 'residential_multi']).to_numpy(), None,
                preds, ['b_type_majority'], acc, 'higher', info=True)
        res['rows'][f'heatpump_info_{mode}'] = {p: dict(share_top=float(ex[f'{p}_hptop'].mean()),
                                                         median_p=float(ex[f'{p}_php'].median()), n=int(len(ex)))
                                                for p in preds}
    with open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(res, fh, indent=1, default=float)
        fh.write('\n')
    for k, r in res['rows'].items():
        if 'best_baseline' in r:
            print(k, 'n', r['n'], 'best', r['best_baseline'], round(r['baselines'][r['best_baseline']], 2),
                  {p: (round(v['value'], 2), [round(x, 2) for x in v['interval']], v['verdict'])
                   for p, v in r.items() if isinstance(v, dict) and 'verdict' in v})
        else:
            print(k, r)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
