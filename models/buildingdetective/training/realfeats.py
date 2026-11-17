"""Feature matrices for the real sets (cached in feats/real/<split>_<set>.npz + .json labels).

Sets: bdg2 (hourly 2017, plus monthly and annual made from the same meters, plus an electricity-only hourly copy),
recs, cbecs (annual), hue (hourly, exam only). Labels are kept beside the features and never enter them.
usage: python3 realfeats.py dev|exam
"""
import json
import os
import sys
import time

import numpy as np

import common as C
import features as F
import realdata as R

OD = os.path.join(C.ROOT, 'feats', 'real')


def _save(name, mats, labels):
    os.makedirs(OD, exist_ok=True)
    np.savez(os.path.join(OD, name + '.npz'), **{k: np.array(v, np.float32) for k, v in mats.items()})
    json.dump(labels, open(os.path.join(OD, name + '.json'), 'w'), default=str)


def bdg2(split):
    meta = R.bdg2_buildings(split)
    mats = dict(hourly=[], monthly=[], annual=[], hourly_elec=[], hourly_1m=[], hourly_3m=[], hourly_6m=[], hourly_2016=[])
    labels = []
    t0 = time.time()
    for _, row in meta.iterrows():
        c = R.bdg2_case(row['id'], 2017, row)
        if c is None:
            continue
        m, T, G, loc, lat = c['meters'], c['T'], c['GHI'], c['loc'], c['lat']
        mats['hourly'].append(F.feat_hourly(m, T, G, loc, lat))
        mm, Tm, Hm, Cm, nd = F.monthly_from_hourly(m, T, loc)
        mats['monthly'].append(F.feat_monthly(mm, Tm, Hm, Cm, nd, lat))
        n = len(T)
        kwh = {}
        for ch, x in m.items():
            v = np.isfinite(x)
            kwh[ch] = float(np.nanmean(x) * 8760) if v.mean() > 0.2 else 0.0
        d = np.asarray(loc.normalize().asi8)
        ud, inv = np.unique(d, return_inverse=True)
        Td = np.bincount(inv, weights=np.nan_to_num(T, nan=np.nanmean(T))) / np.bincount(inv)
        hdd, cdd = float(np.maximum(F.HB - Td, 0).sum()), float(np.maximum(Td - F.HB, 0).sum())
        mats['annual'].append(F.feat_annual(kwh, hdd, cdd))
        if 'elec' in m:
            mats['hourly_elec'].append(F.feat_hourly({'elec': m['elec']}, T, G, loc, lat))
        else:
            mats['hourly_elec'].append(np.full(F.N_W_H + 4 * F.N_C_H, np.nan, np.float32))
        mon = np.asarray(loc.month)
        for k in (1, 3, 6):
            mk = {ch: np.where(mon <= k, x, np.nan) for ch, x in m.items()}
            mats[f'hourly_{k}m'].append(F.feat_hourly(mk, T, G, loc, lat)
                                        if all(np.isfinite(v).sum() > 24 * 10 for v in mk.values())
                                        else np.full(F.N_W_H + 4 * F.N_C_H, np.nan, np.float32))
        c16 = R.bdg2_case(row['id'], 2016, row)
        mats['hourly_2016'].append(F.feat_hourly(c16['meters'], c16['T'], c16['GHI'], c16['loc'], c16['lat'])
                                   if c16 is not None else np.full(F.N_W_H + 4 * F.N_C_H, np.nan, np.float32))
        lab = dict(c['labels'])
        lab.update(id=c['id'], kwh=kwh, hdd=hdd, cdd=cdd, channels=sorted(m))
        labels.append(lab)
        if len(labels) % 100 == 0:
            print(split, len(labels), '%.0f s' % (time.time() - t0), flush=True)
    _save(f'{split}_bdg2', mats, labels)


def surveys(split):
    for name, fn in (('recs', R.recs_rows), ('cbecs', R.cbecs_rows)):
        df, rows = fn(split)
        X, labels = [], []
        for r in rows:
            X.append(F.feat_annual(r['kwh'], r['hdd'], r['cdd']))
            x = r['row']
            lab = dict(id=r['id'], A=x.floor_area_m2, year_lo=x.year_built_lo, year_hi=x.year_built_hi,
                       use=x.primary_use, heating_fuel=x.heating_fuel, kwh=r['kwh'], hdd=r['hdd'], cdd=r['cdd'])
            if name == 'recs':
                lab.update(heat=R.recs_heat_label(x), heating_equipment=x.heating_equipment,
                           insulation=x.insulation_reported, drafty=x.drafty, housing_type=x.housing_type,
                           T_heat_home=x.T_heat_home_C, T_heat_night=x.T_heat_night_C, T_cool_home=x.T_cool_home_C,
                           cooling_equipment=x.cooling_equipment, climate_zone=x.climate_zone)
            else:
                lab.update(heat=R.cbecs_heat_label(x), cool=R.cbecs_cool_label(x), chiller=x.chillr, hp=x.htpmph,
                           boiler=x.boiler, system=x.system, hours_per_week=x.hours_per_week, open24=x.open24,
                           climate_class=x.climate_class)
            labels.append(lab)
        _save(f'{split}_{name}', dict(annual=X), labels)
        print(split, name, len(X), flush=True)


def hue():
    cs = R.hue_cases()
    X, Xm, labels = [], [], []
    for c in cs:
        X.append(F.feat_hourly(c['meters'], c['T'], c['GHI'], c['loc'], c['lat']))
        mm, Tm, Hm, Cm, nd = F.monthly_from_hourly(c['meters'], c['T'], c['loc'])
        Xm.append(F.feat_monthly(mm, Tm, Hm, Cm, nd, c['lat']))
        lab = dict(c['labels']); lab['id'] = c['id']
        labels.append(lab)
    _save('exam_hue', dict(hourly=X, monthly=Xm), labels)
    print('hue', len(X))


if __name__ == '__main__':
    sp = sys.argv[1]
    what = sys.argv[2:] or ['surveys', 'bdg2'] + (['hue'] if sp == 'exam' else [])
    if 'surveys' in what:
        surveys(sp)
    if 'bdg2' in what:
        bdg2(sp)
    if 'hue' in what and sp == 'exam':
        hue()
