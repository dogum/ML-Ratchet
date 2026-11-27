"""Export BasinDetective's live instrument: weights, the four real exam basins and Python reference cases.

    python web/export_instrument.py          (from models/basindetective/, or anywhere; writes site/public/instruments/basindetective/)

- weights.{json,bin}: the 5-member ensemble. The six weight matrices and biases of each member are float16; the
  input normalisation (x_med, x_iqr), the target scales (y_mu, y_sd) and the widen-only sigma factors stay float32.
- examples.{json,bin}: the shipped examples/ exactly as in the CSV files (they hold two decimals): calibration decade
  P, T_mean, PET and Q_mm and target decade P, T_mean, PET, stored as int32 hundredths, so int / 100 gives the very
  same doubles Python parses. meta: names, dates, basin area (for the cfs rounding), truth and GR4J from
  <id>_truth.json, expected.json.
- reference.json: every basin under several what-ifs. Each case runs the model's own featurize() on the perturbed
  input, then its own Detective forward pass with the float16-roundtripped weights. outputs = 11 means, 11 sigmas,
  p_regulated and the 150 features (NaN written as -999), so parity covers the feature port and the network.

The what-ifs (mirrored exactly in site/src/instruments/basindetective/whatif.ts):
  years   keep only the last N water years of the gauged decade (forcing and flow)
  res_k   route the gauged flow through a linear reservoir with time constant k days (storage = k x mean flow, so
          storage / annual flow = k / 365): Q_out[t] = a Q_out[t-1] + (1 - a) Q_in[t], a = exp(-1/k), started at
          the first year's mean inflow
  cv      multiplicative AR(1) gauge jitter (rho 0.9), Q x exp(cv e - cv^2 / 2), e ~ N(0, 1) from mulberry32(seed)
  sig     round the flow to `sig` significant figures in cfs (0.01 cfs floor), as calib_gauge.round_sig
  pmul    target-decade precipitation x pmul
  dT      target-decade temperature + dT, PET x (1 + 0.025 dT) (the training bank's own perturbation)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = Path(__file__).resolve().parents[1]
REPO = MODEL.parents[1]
sys.path.insert(0, str(MODEL))
sys.path.insert(0, str(REPO))
from mlratchet.webexport import instrument_dir, write_json, write_pack  # noqa: E402
import basindetective as bd  # noqa: E402
from basindetective.features import featurize  # noqa: E402
from basindetective.model import Detective, UNITS  # noqa: E402

# Basin areas (km2) used to convert mm/day to cfs: WaterData's CAMELS-US specs (area_km2), copied here so the
# exporter needs no unpacked platform data.
AREA = {'camels_04185000': 1061.3076171875, 'camels_08165300': 436.9837036132813,
        'camels_12388400': 60.54656600952149, 'camels_14020000': 339.1293029785156}
CFS = 0.028316846592
BIG = ('w0', 'b0', 'w1', 'b1', 'wo', 'bo')
BASE = dict(years=9, res_k=0.0, cv=0.0, sig=0, seed=1, pmul=1.0, dT=0.0)
VARIANTS = {
    'recorded': {},
    'drier, warmer target decade': dict(pmul=0.85, dT=2.0),
    'wetter target decade': dict(pmul=1.2),
    'cooler target decade': dict(dT=-1.0),
    'reservoir k 60 d': dict(res_k=60.0),
    'gauge jitter 0.10, 2 figures': dict(cv=0.10, sig=2, seed=7),
    'rounded to 1 figure': dict(sig=1),
    'last 4 gauged years': dict(years=4),
    'combined': dict(years=6, res_k=30.0, cv=0.05, sig=3, seed=3, pmul=0.9, dT=1.0),
}


# ------------------------------------------------------------------------------------ what-ifs (mirrored in TS)
def mulberry32(seed):
    a = seed & 0xFFFFFFFF

    def u():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (1 | a)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    def n():
        x = 0.0
        while x == 0.0:
            x = u()
        y = u()
        return math.sqrt(-2 * math.log(x)) * math.cos(2 * math.pi * y)
    return n


def round_half_even(x):
    return np.round(x)


def apply(cal, tgt, area, v):
    """cal: DataFrame P, T_mean, PET, Q_mm (daily); tgt: P, T_mean, PET. Returns (cal forcing, q, tgt forcing)."""
    v = {**BASE, **v}
    wy = cal.index.year + (cal.index.month >= 10)
    cal = cal[wy > wy.max() - v['years']]
    q = cal['Q_mm'].to_numpy(float).copy()
    if v['res_k'] > 0:
        a = math.exp(-1.0 / v['res_k'])
        out = np.empty_like(q)
        s = float(np.mean(q[:365]))
        for t in range(q.size):
            s = a * s + (1 - a) * q[t] if t > 0 else s
            out[t] = s
        q = out
    if v['cv'] > 0:
        n = mulberry32(int(v['seed']))
        rho, cv = 0.9, v['cv']
        e = n()
        for t in range(q.size):
            if t > 0:
                e = rho * e + math.sqrt(1 - rho * rho) * n()
            q[t] = q[t] * math.exp(cv * e - 0.5 * cv * cv)
    if v['sig'] > 0:
        k = area / 86.4 / CFS
        c = q * k
        r = np.zeros_like(c)
        pos = c > 0
        e10 = np.floor(np.log10(c[pos]))
        step = np.maximum(10.0 ** (e10 - v['sig'] + 1), 0.01)
        r[pos] = round_half_even(c[pos] / step) * step
        q = r / k
    t2 = tgt.copy()
    t2['P'] = t2['P'] * v['pmul']
    t2['T_mean'] = t2['T_mean'] + v['dT']
    t2['PET'] = t2['PET'] * (1 + 0.025 * v['dT'])
    return cal[['P', 'T_mean', 'PET']], q, t2


def main():
    out = instrument_dir('basindetective')
    det = bd.load()
    W = det.W
    n_members = int(W['n_members'])
    names = det.names
    # ---- weights pack: f16 matrices, f32 normalisation / scales
    tensors, keep = {}, {}
    for i in range(n_members):
        for k in BIG:
            tensors[f'm{i}_{k}'] = W[f'm{i}_{k}']
        for k in ('y_mu', 'y_sd'):
            tensors[f'm{i}_{k}'] = W[f'm{i}_{k}']
            keep[f'm{i}_{k}'] = 'f32'
        assert np.array_equal(W[f'm{i}_x_med'], W['m0_x_med']) and np.array_equal(W[f'm{i}_x_iqr'], W['m0_x_iqr'])
    for k in ('x_med', 'x_iqr'):
        tensors[k] = W[f'm0_{k}']
        keep[k] = 'f32'
    tensors['sig_scale'] = W['sig_scale']
    keep['sig_scale'] = 'f32'
    meta = {'names': names, 'targets': det.targets, 'n_members': n_members, 'n_layers': int(W['m0_n_layers']),
            'cls_temp': float(W['cls_temp']), 'units': UNITS, 'version': '1.1',
            'note': 'BasinDetective v1.1 (symmetric precipitation-change prior), 5-member ensemble; matrices float16, normalisation float32'}
    write_pack(out / 'weights', tensors, dtype='f16', meta=meta, keep=keep)
    # the weights the browser sees, back in the model's own class
    Wr = {k: v for k, v in W.items()}
    for k, v in tensors.items():
        dt = np.float32 if keep.get(k) == 'f32' else np.float16
        a = np.asarray(v).astype(dt).astype(np.float64)
        if k in ('x_med', 'x_iqr'):
            for i in range(n_members):
                Wr[f'm{i}_{k}'] = a
        else:
            Wr[k] = a
    Wr['sig_scale'] = np.asarray(W['sig_scale']).astype(np.float32).astype(np.float64)
    rt = Detective(None, Wr)

    # ---- examples pack
    exp = json.loads((MODEL / 'examples' / 'expected.json').read_text(encoding='utf-8'))
    arrays, basins, cases = {}, [], []
    worst16 = 0.0
    for i, f in enumerate(sorted((MODEL / 'examples').glob('*_cal.csv'))):
        b = f.name[:-8]
        cal = pd.read_csv(f, index_col=0, parse_dates=True)
        tgt = pd.read_csv(f.with_name(f'{b}_tgt.csv'), index_col=0, parse_dates=True)[['P', 'T_mean', 'PET']]
        tr = json.loads(f.with_name(f'{b}_truth.json').read_text(encoding='utf-8'))
        assert cal.notna().all().all() and tgt.notna().all().all()
        assert (np.diff(cal.index.values).astype('timedelta64[D]').astype(int) == 1).all()
        assert (np.diff(tgt.index.values).astype('timedelta64[D]').astype(int) == 1).all()
        ci = np.rint(cal[['P', 'T_mean', 'PET', 'Q_mm']].to_numpy() * 100).astype(np.int32)
        ti = np.rint(tgt.to_numpy() * 100).astype(np.int32)
        # int / 100 must give exactly the doubles pandas parsed, so Python and the browser start from one input
        assert np.array_equal(ci / 100.0, cal[['P', 'T_mean', 'PET', 'Q_mm']].to_numpy())
        assert np.array_equal(ti / 100.0, tgt.to_numpy())
        arrays[f'cal{i}'], arrays[f'tgt{i}'] = ci, ti
        x0, s_cal, _ = featurize(cal[['P', 'T_mean', 'PET']], cal['Q_mm'].values, tgt)
        fs, ar = float(x0[names.index('c_fsnow')]), float(x0[names.index('c_arid')])
        stratum = ('regulated' if tr['reservoir_storage_over_annual_flow'] > 0.1 else
                   ('snow' if fs >= 0.15 else 'rain') + ', ' + ('arid' if ar > 1 else 'humid'))
        full = det.predict(x0[None])[0]
        basins.append({'id': b, 'name': tr['name'], 'stratum': stratum, 'area_km2': AREA[b],
                       'cal_start': str(cal.index[0].date()), 'tgt_start': str(tgt.index[0].date()),
                       'n_cal': int(len(cal)), 'n_tgt': int(len(tgt)),
                       'observed_change': tr['observed_change'], 'gr4j_change': tr['gr4j_change'],
                       'dor': tr['reservoir_storage_over_annual_flow'], 'expected': exp[b],
                       'cal_signatures': {k: s_cal[k] for k in ('q_mean', 'runoff_ratio', 'bfi_lh', 'q5', 'q95',
                                                                 'hfd_mean')},
                       'release': {'mu': [full[t][0] for t in det.targets], 'sd': [full[t][1] for t in det.targets],
                                   'p': full['p_regulated']}})
        for vname, v in VARIANTS.items():
            cf, q, tf = apply(cal, tgt, AREA[b], v)
            x, _, nm = featurize(cf, q, tf)
            assert nm == names
            mu, sd, p = rt.forward(x[None].astype(np.float64))
            if vname == 'recorded':
                mu0, sd0, _ = det.forward(x[None])
                worst16 = max(worst16, float(np.max(np.abs(mu - mu0) / np.maximum(1, np.abs(mu0)))))
            feats = [(-999.0 if not np.isfinite(val) else float(val)) for val in x]
            cases.append({'example': i, 'variant': vname, **{**BASE, **v},
                          'outputs': [*map(float, mu[0]), *map(float, sd[0]), float(p[0]), *feats]})
        print(b, stratum)
    write_pack(out / 'examples', arrays, dtype='i32', meta={'basins': basins})
    write_json(out / 'reference.json', {'cases': cases, 'tolerance': 1e-6,
                                        'note': 'outputs = 11 means, 11 sigmas, p_regulated, 150 features (NaN = -999); '
                                                'weights float16-roundtripped'})
    size = sum(p.stat().st_size for p in out.iterdir())
    print(f'{len(cases)} reference cases; float16 vs float64 release on the recorded inputs: worst relative '
          f'mean difference {worst16:.1e}; {size / 1e6:.2f} MB in {out}')


if __name__ == '__main__':
    main()
