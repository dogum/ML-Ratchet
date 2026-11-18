"""Export FlowCast's released weights (ungauged and nowcast), the three real exam basins and Python reference cases
for the browser instrument (site/public/instruments/flowcast/).

Packs:
  weights.{json,bin}  the ungauged weights (rung r1b), float16; k_widen exact in meta
  nowcast.{json,bin}  the gauged nowcast weights (n1), float16
  basins.{json,bin}   per basin i: F<i> (1,097 days x 8 forcing channels, float32, WY1994 warm-up + WY1995-96 + one
                      day), doy<i> (day of year of each row), obs<i> (731 days of observed USGS flow, mm/day)

Reference cases run flowcast.py itself (FlowCast.predict, or predict_arrays for the frozen-statics case) on the
float32 forcing the browser sees, with the float16-roundtripped weights, recorded and with what-ifs on the input:
precipitation, temperature and PET scaled or shifted, a storm inserted or removed, attributes changed, the climate
statics frozen at their recorded values, and the nowcast at several leads with a gauge outage. Each case keeps the
18 static features and the mean, q05 and q95 on every 5th predicted day.

Run from the repository root: python models/flowcast/web/export_instrument.py
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = Path(__file__).resolve().parents[1]
REPO = MODEL.parents[1]
sys.path.insert(0, str(MODEL))
sys.path.insert(0, str(REPO))
import flowcast as fc  # noqa: E402
from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

STRATUM = {'camels_01137500': 'snow', 'camels_07315700': 'arid', 'camels_12095000': 'humid'}   # webdata.py's draw
STEP = 5          # keep every 5th predicted day in the reference
TEMP = ['T_mean', 'T_min', 'T_max']


def released(path):
    """A FlowCast instance on float16-roundtripped weights (what the browser sees); widening factors exact."""
    m = fc.FlowCast(str(path))
    with np.load(path) as z:
        W = {k: z[k] for k in z.files if k != 'k_widen'}
        k = [float(x) for x in z['k_widen']]
    m.W = {kk: v.astype(np.float64) for kk, v in roundtrip(W, 'f16').items()}
    return m, W, k


def perturb(F, attrs, ops):
    """Apply the instrument's what-ifs to a forcing frame (float64 copy of the float32 values) and attributes."""
    F = F.copy()
    a = copy.deepcopy(attrs)
    if 'pmul' in ops:
        F['P'] = F['P'] * ops['pmul']
    if 'tshift' in ops:
        for c in TEMP:
            F[c] = F[c] + ops['tshift']
    if 'petmul' in ops:
        F['PET'] = F['PET'] * ops['petmul']
    if 'drop' in ops:                       # remove a storm: P = 0 on forcing rows r0..r1 (inclusive)
        r0, r1 = ops['drop']
        F.iloc[r0:r1 + 1, F.columns.get_loc('P')] = 0.0
    if 'storm' in ops:                      # insert a two-day storm: 60 % on row r, 40 % on row r + 1
        r, mm = ops['storm']
        j = F.columns.get_loc('P')
        F.iloc[r, j] += 0.6 * mm
        F.iloc[r + 1, j] += 0.4 * mm
    if 'elev' in ops:
        for kk in ('elev_mean_m', 'elev_min_m', 'elev_max_m'):
            a[kk] = a[kk] + ops['elev']
    if 'area' in ops:
        a['area_km2'] = a['area_km2'] * ops['area']
    if 'forest' in ops:
        a['forest_frac'] = ops['forest']
    # float32 again, as the browser stores it after a what-if (Math.fround)
    return F.astype(np.float32).astype(np.float64), a


def run(m, k, F, attrs, start, end, ops, obs=None, lead=None, outage=None):
    Fp, a = perturb(F, attrs, ops)
    q = None
    if obs is not None:
        q = obs.copy()
        if outage:
            q.iloc[outage[0]:outage[1] + 1] = np.nan
    if ops.get('freeze'):                   # network reads the perturbed weather, but the climate statics and the
        idx = pd.date_range(start, end)     # output scale stay at the recorded forcing's values
        F0 = F.values.astype(np.float32)
        clim = fc.climate_stats(F0[:, 0], F0[:, 1], F0[:, 2])
        st = fc.static_vector(fc.spec_attrs(a), clim)
        pm = clim['p_mean']
        mean, qq = m.predict_arrays(Fp.values.astype(np.float32), Fp.index.day_of_year.values, st, len(idx), None, 1)
        qq = m.widen(qq) * pm
        out = pd.DataFrame({'Q': mean * pm, 'q05': qq[:, 0], 'q95': qq[:, 4]}, index=idx)
    else:
        out = m.predict(Fp, a, start, end, qobs=q, lead=lead or 1)
        Fa = Fp.values.astype(np.float32)
        st = fc.static_vector(fc.spec_attrs(a), fc.climate_stats(Fa[:, 0], Fa[:, 1], Fa[:, 2]))
    sel = slice(None, None, STEP)
    return [float(x) for x in st] + [float(x) for x in out.Q.values[sel]] + \
        [float(x) for x in out.q05.values[sel]] + [float(x) for x in out.q95.values[sel]]


def main():
    out = instrument_dir('flowcast')
    m, W, k = released(MODEL / 'flowcast_weights.npz')
    n, Wn, kn = released(MODEL / 'flowcast_nowcast_weights.npz')
    m.k, n.k = tuple(k), tuple(kn)
    write_pack(out / 'weights', W, dtype='f16', meta={'k_widen': k, 'rung': 'r1b', 'statics': fc.STATIC_NAMES})
    write_pack(out / 'nowcast', Wn, dtype='f16', meta={'k_widen': kn, 'rung': 'n1', 'statics': fc.STATIC_NAMES})

    ex = json.loads((MODEL / 'examples/examples.json').read_text(encoding='utf-8'))
    specs_path = REPO / 'shared/water/data/proc/specs/camels_specs.json'
    specs = json.loads(specs_path.read_text(encoding='utf-8')) if specs_path.exists() else {}
    scores_path = REPO / 'shared/water/data/proc/baselines/flow_scores.parquet'
    scores = pd.read_parquet(scores_path) if scores_path.exists() else None
    arrays, info, cases = {}, [], []
    for i, e in enumerate(ex):
        F = pd.read_csv(MODEL / 'examples' / e['forcing_csv'], index_col=0, parse_dates=True)[fc.DYN]
        F = F.astype(np.float32).astype(np.float64)          # exactly the float32 values the browser holds
        obs = pd.read_csv(MODEL / 'examples' / e['obs_csv'], index_col=0, parse_dates=True)['Q_mm']
        obs = obs.astype(np.float32).astype(np.float64)
        idx = pd.date_range(e['start'], e['end'])
        assert F.index[0] == idx[0] - pd.Timedelta(days=365) and len(F) == 365 + len(idx) + 1
        assert obs.index.equals(idx)
        arrays[f'F{i}'] = F.values.astype(np.float32)
        arrays[f'doy{i}'] = F.index.day_of_year.values.astype(np.int32)
        arrays[f'obs{i}'] = obs.values.astype(np.float32)
        sp = specs.get(e['id'], {})
        decade = {}
        if scores is not None:
            s = scores[(scores.basin == e['id']) & (scores.period == 'val')].set_index('method').nse
            decade = {kk: round(float(v), 3) for kk, v in s.items()}
        rescore = MODEL / 'training/results/rescore_main_2026-11-18.json'   # the released model, re-scored per basin
        if rescore.exists():
            rs = json.loads(rescore.read_text(encoding='utf-8'))
            row = {r[0]: r for r in rs['rows']}.get(e['id'])
            if row is not None:
                decade['flowcast'] = round(float(row[rs['columns'].index('nse')]), 3)
        info.append({'id': e['id'], 'name': e['name'], 'stratum': STRATUM.get(e['id'], ''), 'attrs': e['attrs'],
                     'start': e['start'], 'end': e['end'], 'forcing_start': str(F.index[0].date()),
                     'n_target': len(idx), 'lat': sp.get('lat'), 'lon': sp.get('lon'),
                     'decade_nse': decade})
        # the largest storm of WY1996 (target rows 365..730): its day and the day either side, as the UI's "remove"
        t = 365 + 365 + int(np.argmax(F['P'].values[365 + 365:365 + 731]))
        variants = [
            ('recorded', {}), ('P x0.7', {'pmul': 0.7}), ('P x1.4', {'pmul': 1.4}),
            ('P x1.4, statics frozen', {'pmul': 1.4, 'freeze': True}), ('T +3 C', {'tshift': 3.0}),
            ('T -3 C', {'tshift': -3.0}), ('PET x1.2', {'petmul': 1.2}),
            ('storm 80 mm on target day 400', {'storm': (365 + 400, 80.0)}),
            (f'drop storm rows {t - 1}-{t + 1}', {'drop': (t - 1, t + 1)}),
            ('elevation +800 m', {'elev': 800.0}), ('area x10', {'area': 10.0}), ('forest 0.1', {'forest': 0.1}),
            ('P x0.8, T +2 C, area x0.5', {'pmul': 0.8, 'tshift': 2.0, 'area': 0.5}),
        ]
        for name, ops in variants:
            cases.append(dict(example=i, variant=name, basin=i, weights='weights', ops=ops,
                              outputs=run(m, k, F, e['attrs'], e['start'], e['end'], ops)))
        for name, ops, lead, outage in [('nowcast lead 1', {}, 1, None), ('nowcast lead 3', {}, 3, None),
                                        ('nowcast lead 7', {}, 7, None),
                                        ('nowcast lead 3, gauge out target days 400-429', {}, 3, (400, 429)),
                                        ('nowcast lead 2, P x1.2', {'pmul': 1.2}, 2, None)]:
            cases.append(dict(example=i, variant=name, basin=i, weights='nowcast', ops=ops, lead=lead,
                              outage=list(outage) if outage else None,
                              outputs=run(n, kn, F, e['attrs'], e['start'], e['end'], ops, obs=obs, lead=lead,
                                          outage=outage)))
    write_pack(out / 'basins', arrays, dtype='f32', keep={f'doy{i}': 'i32' for i in range(len(ex))},
               meta={'basins': info, 'step': STEP})
    for c in cases:
        c['ops'] = {kk: (list(v) if isinstance(v, tuple) else v) for kk, v in c['ops'].items()}
    write_json(out / 'reference.json', dict(cases=cases, tolerance=1e-5, step=STEP))
    size = sum(p.stat().st_size for p in out.iterdir())
    print(f'{len(ex)} basins, {len(cases)} parity cases, {size / 1e6:.2f} MB in {out}')


if __name__ == '__main__':
    main()
