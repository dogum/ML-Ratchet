"""Assemble web/buildingdetective.html from page.html + detective.js + explorer.json + lab grid + exam/ratchet summary.
usage: python3 buildpage.py"""
import base64
import json
import os

import numpy as np
import pandas as pd

import common as C
from webdata import tz_changes

WEB = os.path.join(C.ROOT, 'web')


def f(v, d=0, pct=False, pm=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '–'
    s = f'{v:,.{d}f}'
    return s + ('%' if pct else '')


def lab_weather():
    out = {}
    for site in ('Eagle', 'Rat'):
        z = np.load(os.path.join(C.ROOT, 'bank', 'weather', f'site_{site}_2017.npz'))
        idx = pd.DatetimeIndex(z['utc'].astype('datetime64[us]'), tz='UTC')
        T = np.round(np.nan_to_num(z['T'].astype(float), nan=-3276.8) * 10).astype(np.int16)
        G = np.round(np.clip(z['GHI'].astype(float), 0, 6000)).astype(np.uint16)
        out[site] = dict(T=base64.b64encode(T.tobytes()).decode(), GHI=base64.b64encode(G.tobytes()).decode(),
                         t0=int(idx[0].value // 10 ** 6), n=len(idx), tzc=tz_changes(idx, str(z['tz'])), lat=float(z['lat']))
    return out


def summary():
    X = json.load(open(os.path.join(C.ROOT, 'exam.json')))
    R = json.load(open(os.path.join(C.ROOT, 'ratchet.json')))
    TW = json.load(open(os.path.join(C.ROOT, 'models', 'twintest.json')))
    B = X['bdg2_baselines_B'].get('detective', X['bdg2_baselines_B'])
    ex = []
    b = X['bdg2']
    bb = X['bdg2_baselines']
    rows = []
    best = min(b[m]['area']['MdAPE'] for m in ('hourly', 'monthly', 'annual'))
    for m in ('hourly', 'monthly', 'annual'):
        rows.append(dict(task=f'Floor area, median abs. % error', input=f'{m} input, 2017', truth='real', n=b[m]['area']['n'],
                         det=f(b[m]['area']['MdAPE'], 1, True) + f" · {f(100 * b[m]['area']['within_x2'])}% within ×2",
                         win=b[m]['area']['MdAPE'] < min(bb['area_eui_by_true_use']['MdAPE'], B['bdg2_floor_area']['eui_by_use_MdAPE']),
                         base=[dict(name=f"electricity ÷ dev median EUI of the TRUE use, same buildings (n={bb['area_eui_by_true_use']['n']})", v=f(bb['area_eui_by_true_use']['MdAPE'], 1, True)),
                               dict(name='same rule as quoted by BuildingData (1,133 quality-checked meters)', v=f(B['bdg2_floor_area']['eui_by_use_MdAPE'], 1, True)),
                               dict(name='dev median area', v=f(bb['area_dumb']['MdAPE'], 1, True))]))
    pu = B['bdg2_primary_use']
    rows.append(dict(task='Primary use, accuracy · macro-F1', input='hourly input', truth='real', n=b['hourly']['use']['n'],
                     det=f"{f(100 * b['hourly']['use']['accuracy'])}% · {b['hourly']['use']['macroF1']:.2f}",
                     win=b['hourly']['use']['macroF1'] > max(pu['weekly_shape_centroid_macroF1'], bb['use_majority']['macroF1']),
                     base=[dict(name='majority class (education)', v=f"{f(100 * bb['use_majority']['accuracy'])}% · {bb['use_majority']['macroF1']:.2f}"),
                           dict(name='weekly-shape nearest centroid (B)', v=f"{f(100 * pu['weekly_shape_centroid_accuracy'])}% · {pu['weekly_shape_centroid_macroF1']:.2f}")]))
    rows.append(dict(task='Year built, mean abs. error', input='hourly input', truth='real', n=b['hourly']['year']['n'],
                     det=f(b['hourly']['year']['MAE'], 1) + ' yr', win=b['hourly']['year']['MAE'] < bb['year_dev_median']['MAE'],
                     base=[dict(name='dev median', v=f(bb['year_dev_median']['MAE'], 1) + ' yr')]))
    eo = b['elec_only']
    rows.append(dict(task='Has a chilled-water meter? (from electricity alone), ROC AUC', input='hourly electricity only',
                     truth='real', n=eo['n'], det=f(eo['auc_district_cooling'], 2), win=eo['auc_district_cooling'] > 0.5,
                     base=[dict(name='chance', v='0.50')]))
    rows.append(dict(task='Has a steam / hot-water meter? (from electricity alone), ROC AUC', input='hourly electricity only',
                     truth='real', n=eo['n'], det=f(eo['auc_district_heat'], 2), win=eo['auc_district_heat'] > 0.5,
                     base=[dict(name='chance', v='0.50')]))
    ex.append(dict(group='BDG2: 1,584 real non-residential buildings (exam split), meters for 2017', rows=rows))
    # RECS
    r, rr = X['recs'], X['recs_rules']
    rows = [dict(task='Floor area, median abs. % error', input='annual kWh by fuel + degree days', truth='real', n=r['area']['n'],
                 det=f(r['area']['MdAPE'], 1, True), win=r['area']['MdAPE'] < B['recs_floor_area']['dumb_dev_median_by_type_MdAPE'],
                 base=[dict(name='energy ÷ dev EUI by TRUE type × climate (B)', v=f(B['recs_floor_area']['eui_by_type_climate_MdAPE'], 1, True)),
                       dict(name='dev median by TRUE housing type (B)', v=f(B['recs_floor_area']['dumb_dev_median_by_type_MdAPE'], 1, True))]),
            dict(task='Main heating: fuel vs electric vs none, accuracy', input='annual', truth='real', n=r['heat_fuel_vs_electric']['n'],
                 det=f(100 * r['heat_fuel_vs_electric']['accuracy'], 1, True), win=r['heat_fuel_vs_electric']['accuracy'] > rr['fuel_rule_accuracy'],
                 base=[dict(name='rule: any fuel used → fuel', v=f(100 * rr['fuel_rule_accuracy'], 1, True)),
                       dict(name='majority (fuel)', v=f(100 * rr['majority_accuracy'], 1, True))]),
            dict(task='Heat pump vs electric resistance, balanced accuracy · AUC', input='annual', truth='real', n=r['hp_vs_resistance']['n'],
                 det=f"{r['hp_vs_resistance']['balanced_accuracy']:.2f} · {r['hp_vs_resistance']['auc']:.2f}",
                 win=r['hp_vs_resistance']['balanced_accuracy'] > 0.52,
                 base=[dict(name='kWh/m²/HDD rule, uses TRUE area (B)', v=f"{B['recs_heat_pump_vs_resistance']['rule_balanced_accuracy']:.2f}"),
                       dict(name='majority', v='0.50')]),
            dict(task='Vintage, mean abs. error to the survey range midpoint', input='annual', truth='real', n=r['year']['n'],
                 det=f(r['year']['MAE'], 1) + ' yr', win=r['year']['MAE'] < rr['year_dev_median_MAE'],
                 base=[dict(name=f"dev median {rr['year_dev_median']:.0f}", v=f(rr['year_dev_median_MAE'], 1) + ' yr')])]
    ex.append(dict(group='RECS 2020: 7,401 real homes (exam split), annual energy only', rows=rows))
    c, cr, ch = X['cbecs'], X['cbecs_rules'], X['cbecs_hours']
    rows = [dict(task='Floor area, median abs. % error', input='annual', truth='real', n=c['area']['n'], det=f(c['area']['MdAPE'], 1, True),
                 win=c['area']['MdAPE'] < B['cbecs_floor_area']['eui_by_type_climate_MdAPE'],
                 base=[dict(name='energy ÷ dev EUI by TRUE type × climate (B)', v=f(B['cbecs_floor_area']['eui_by_type_climate_MdAPE'], 1, True)),
                       dict(name='dev median by TRUE type (B)', v=f(B['cbecs_floor_area']['dumb_dev_median_by_type_MdAPE'], 1, True))]),
            dict(task='Primary use (10 commercial classes), accuracy · macro-F1', input='annual', truth='real', n=c['use']['n'],
                 det=f"{f(100 * c['use']['accuracy'])}% · {c['use']['macroF1']:.2f}", win=False,
                 base=[dict(name='majority class', v=f"{f(100 * X['cbecs_use_majority']['accuracy'])}% · {X['cbecs_use_majority']['macroF1']:.2f}")]),
            dict(task='Heating: fuel / electric / district / none, accuracy', input='annual', truth='real', n=c['heat_class']['n'],
                 det=f(100 * c['heat_class']['accuracy'], 1, True), win=c['heat_class']['accuracy'] > cr['fuel_rule_accuracy'],
                 base=[dict(name='rule: fuel used → fuel, district → district', v=f(100 * cr['fuel_rule_accuracy'], 1, True)),
                       dict(name='majority (fuel)', v=f(100 * cr['majority_accuracy'], 1, True))]),
            dict(task='Has a chiller, ROC AUC', input='annual', truth='real', n=c['chiller']['n'], det=f(c['chiller']['auc'], 2),
                 win=c['chiller']['auc'] > 0.5, base=[dict(name='chance', v='0.50')]),
            dict(task='Weekly operating hours, mean abs. error', input='annual', truth='real', n=ch['n'], det=f(ch['detective_MAE_h'], 1) + ' h',
                 win=ch['detective_MAE_h'] < ch['dev_median_MAE_h'],
                 base=[dict(name='dev median', v=f(ch['dev_median_MAE_h'], 1) + ' h'),
                       dict(name='dev median of the TRUE use', v=f(ch['median_by_true_use_MAE_h'], 1) + ' h')]),
            dict(task='Vintage, mean abs. error', input='annual', truth='real', n=c['year']['n'], det=f(c['year']['MAE'], 1) + ' yr',
                 win=c['year']['MAE'] < cr['year_dev_median_MAE'],
                 base=[dict(name=f"dev median {cr['year_dev_median']:.0f}", v=f(cr['year_dev_median_MAE'], 1) + ' yr')])]
    ex.append(dict(group='CBECS 2018: 2,558 real commercial buildings (exam split), annual energy only', rows=rows))
    if 'hue' in X:
        h = X['hue']['hourly']
        maj = 1 - h['n_electric'] / h['n'] if h['n_electric'] < h['n'] / 2 else h['n_electric'] / h['n']
        rows = [dict(task='Electric main heat (baseboard or heat pump) vs not, accuracy · AUC', input='hourly electricity only', truth='real',
                     n=h['n'], det=f"{f(100 * h['accuracy_electric_vs_not'])}% · {h['auc_electric_heat']:.2f}",
                     win=h['accuracy_electric_vs_not'] > maj, base=[dict(name='majority', v=f(100 * maj) + '%')])]
        ex.append(dict(group=f"HUE: {h['n']} real homes in British Columbia, hourly electricity", rows=rows))
    # simulated
    sv, sb = X['sim_val']['hourly'], X['sim_baselines']['hourly']
    rows = []
    for k, name, unit in (('logUA', 'Envelope UA, MAE in log10', ''), ('logtau', 'Time constant, MAE in log10', ''),
                          ('Tb_heat', 'Heating balance point, MAE', ' K'), ('Tb_cool', 'Cooling balance point, MAE', ' K'),
                          ('logS_heat', 'Heating energy per K, MAE in log10', ''), ('op_hours', 'Weekday operating hours, MAE', ' h')):
        bl = [dict(name='median of the TRUE use type', v=f(sb['median_by_use_' + k]['MAE'], 2) + unit)]
        if 'prism_' + k in sb:
            bl.insert(0, dict(name=f"PRISM change-point (fits {f(100 * sb['prism_' + k]['share_fitted'])}% of units)", v=f(sb['prism_' + k]['MAE'], 2) + unit))
        win = sv[k]['MAE'] < min(float(x['v'].split()[0].replace(',', '')) for x in bl)
        rows.append(dict(task=name, input='hourly, quirked meters', truth='sim', n=sv[k]['n'], det=f(sv[k]['MAE'], 2) + unit, win=win, base=bl))
    rows.append(dict(task='Heating system (5 classes), balanced accuracy', input='hourly', truth='sim', n=None,
                     det=f(sv['heat']['balanced_accuracy'], 2), win=sv['heat']['balanced_accuracy'] > sb['heat_fuel_rule']['balanced_accuracy'],
                     base=[dict(name='rule from which meters exist', v=f(sb['heat_fuel_rule']['balanced_accuracy'], 2)),
                           dict(name='majority', v='0.20')]))
    rows.append(dict(task='Heat pump vs resistance, ROC AUC', input='hourly', truth='sim', n=sv['hp_vs_resistance']['n'],
                     det=f(sv['hp_vs_resistance']['auc'], 2), win=True, base=[dict(name='chance', v='0.50')]))
    ex.append(dict(group='Simulated: 1,000 held-out BuildingSim buildings (frozen validation set), exact labels', rows=rows))
    st = X['stock']
    rows = []
    for kind, label in (('resstock', 'ResStock homes'), ('comstock', 'ComStock buildings')):
        if kind not in st:
            continue
        s = st[kind]['hourly']
        rows.append(dict(task=f'{label}: envelope UA, MAE in log10', input='EnergyPlus hourly meters, AMY2018', truth='simsim', n=s['logUA']['n'],
                         det=f(s['detective_on_prism_units_logUA']['MAE'], 2) + f" ({f(s['logUA']['MAE'], 2)} on all)",
                         win=s['detective_on_prism_units_logUA']['MAE'] < min(s['prism_logUA']['MAE'], s['logUA']['MAE_median_predictor']),
                         base=[dict(name=f"PRISM × nominal efficiency (fits {s['prism_logUA']['n']})", v=f(s['prism_logUA']['MAE'], 2)),
                               dict(name='sample median', v=f(s['logUA']['MAE_median_predictor'], 2))]))
        rows.append(dict(task=f'{label}: heating balance point, MAE', input='EnergyPlus hourly', truth='simsim', n=s['Tb_heat']['n'],
                         det=f(s['detective_on_prism_units_Tb_heat']['MAE'], 1) + ' K', win=s['detective_on_prism_units_Tb_heat']['MAE'] < s['prism_Tb_heat']['MAE'],
                         base=[dict(name='PRISM', v=f(s['prism_Tb_heat']['MAE'], 1) + ' K')]))
        rows.append(dict(task=f'{label}: heating system accuracy (stock label)', input='EnergyPlus hourly', truth='real-label-sim-meter', n=s['heat']['n'],
                         det=f(100 * s['heat']['accuracy'], 0, True), win=False, base=[]))
    ex.append(dict(group='NREL ResStock / ComStock models: EnergyPlus meters, truth from a BuildingSim twin of the same characteristics', rows=rows))
    # coverage
    cov = []
    for m in ('hourly', 'annual'):
        cov.append(f"BDG2 floor area ({m}) {f(100 * b[m]['area']['cov1'])}%")
    cov.append(f"RECS area {f(100 * r['area']['cov1'])}%, RECS vintage {f(100 * r['year']['cov1'])}%")
    cov.append(f"simulated UA {f(100 * sv['logUA']['cov1'])}%, tau {f(100 * sv['logtau']['cov1'])}%")
    cov_note = ('Error-bar coverage (share of truths inside ±1σ, 68% is ideal), after one widening factor per output fitted on '
                'dev only: ' + '; '.join(cov) + '.')
    rungs = []
    for x in R['rungs']:
        if x.get('kind') == 'rebaseline':
            rungs.append(dict(name=x['name'], desc=x['reason'], val=f"{x['old_val_skill']:.3f} → {x['val_skill']:.3f}", gate='', kept=True,
                              decision='re-scored'))
            continue
        rungs.append(dict(name=x['name'], desc=x['description'], val=f"{x['val_skill']:.3f}", gate=f"{x['dev_gate']:.3f}",
                          kept=x['kept'], decision=('kept' if x['kept'] else 'rejected') + (': ' + x['reason'].split('|')[-1].strip()
                                                                                         if 'WITHDRAWN' in x['reason'] else '')))
    twin = []
    a, b2 = TW['devtwin_v1'], TW['devtwin_v2']
    for k in ('elec|night/day', 'elec|cooling slope', 'dheat|heating slope', 'dheat|night/day', 'dheat|p99/3',
              'dcool|night/day', 'dcool|cooling slope'):
        ch, fe = k.split('|')
        twin.append(dict(f={'elec': 'Electricity', 'dheat': 'District heat', 'dcool': 'Chilled water'}[ch] + ' · ' + fe,
                         real=f(a[k]['real'], 2), v1=f(a[k]['twin'], 2), v2=f(b2[k]['twin'], 2)))
    lim = X.get('limits', [])
    return dict(exam=ex, coverage_note=cov_note, rungs=rungs, twin=twin, limits=lim, n_train_buildings=X['n_train_buildings'],
                kpis=X['kpis'])


def nonan(o):
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: nonan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [nonan(v) for v in o]
    if isinstance(o, (np.floating,)):
        return nonan(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main():
    E = json.load(open(os.path.join(WEB, 'explorer.json')))
    lab = json.load(open(os.path.join(WEB, 'lab_grid.json')))
    lab['weather'] = lab_weather()
    if os.path.exists(os.path.join(C.ROOT, 'exam.json')):
        S = summary()
        hero = json.load(open(os.path.join(C.ROOT, 'exam.json'))).get('hero', E['units'][0]['id'])
    else:                                    # layout preview before the exam exists
        S = dict(exam=[], coverage_note='(exam not run yet)', rungs=[], twin=[], limits=['(pending)'], n_train_buildings=0,
                 kpis=[dict(v='–', l='pending')] * 4)
        hero = E['units'][0]['id']
    D = dict(units=E['units'], weather=E['weather'], nets=E['nets'], lab=lab, summary=S, hero=hero)
    page = open(os.path.join(WEB, 'page.html')).read()
    js = open(os.path.join(WEB, 'detective.js')).read()
    data = json.dumps(nonan(D), separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    out = page.replace('/*PAGEDATA*/', data).replace('/*DETECTIVEJS*/', js)
    fn = os.path.join(WEB, 'buildingdetective.html')
    open(fn, 'w').write(out)
    print(fn, '%.2f MB' % (len(out) / 1e6))


if __name__ == '__main__':
    main()
