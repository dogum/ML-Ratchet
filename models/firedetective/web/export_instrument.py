"""Export FireDetective's live instrument: weights, the four real exam fires and Python reference cases.

    python web/export_instrument.py          (from anywhere; writes site/public/instruments/firedetective/)

- weights.{json,bin}: the 3-member ensemble. Every member matrix and bias is float16; the input normalisation
  (seq_mu, seq_sd, st_mu, st_sd) stays float32. meta: the dev calibration (weights/calib.json), feature names.
- examples.{json,bin}: the four exam fires in examples/ as raw observations, exactly as demo.py loads them:
  cumulative FEDS burned-area masks (u8, [steps, ny, nx], 180 m), LANDFIRE layers (fuel u8; cc, ch, cbh, cbd, slope,
  elev float32, as stored). meta: per fire the VIIRS detections (row, col, FRP), day/night flags, gridMET rows, day
  of year, grid convergence, name, stratum, truth (MTBS high-severity share, gridMET wind, Globe-LFMC) and the
  classical values (crown-fire potential, ERC max, raw Rothermel back-calculated wind).
- reference.json: every fire under several what-ifs. Each case runs the model's own feats.py on the changed raw
  input (pipeline.features), then FireDetective.predict with the float16-roundtripped weights. outputs = the 13
  readable outputs, the 12 x 25 sequence features, the wind-blind copy's 3 wind columns and flag, and the 17
  static features, so parity covers the feature port and the network.

The what-ifs change the INPUT (mirrored exactly in site/src/instruments/firedetective/whatif.ts):
  drop    drop each VIIRS detection with probability `drop` (u < drop, u from mulberry32(seed), all steps in order):
          cloud or omission. The FEDS perimeters stay as observed.
  frp     every detection's fire radiative power x frp
  dry     gridMET 100-h and 1000-h dead fuel moisture minus `dry` percentage points (floor 1 %); ERC and BI unchanged
  wind    gridMET wind speed x wind (direction unchanged). The wind reading is made with the wind inputs removed, so
          only the other readings can move.
  cbh     LANDFIRE canopy base height x cbh (float32, as the layers are stored)
  steps   keep the first `steps` half-day steps (the record the network reads; it was trained on up to 12)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[1]
REPO = MODEL.parents[1]
sys.path.insert(0, str(MODEL))
sys.path.insert(0, str(REPO))
from mlratchet.webexport import instrument_dir, write_json, write_pack  # noqa: E402
import demo  # noqa: E402
import feats  # noqa: E402
import firedetective as fdet  # noqa: E402
import pipeline  # noqa: E402

LAYERS = ('fuel', 'cc', 'ch', 'cbh', 'cbd', 'slope', 'elev')
OUTS = ('hs', 'hs_sd', 'crown', 'wind', 'wind_lo', 'wind_hi', 'lfmc', 'lfmc_sd', 'ros_adj', 'm1', 'wind_regime',
        'p_spot', 'p_ia')
BASE = dict(drop=0.0, seed=1, frp=1.0, dry=0.0, wind=1.0, cbh=1.0, steps=12)
VARIANTS = {
    'recorded': {},
    'half the VIIRS pixels lost': dict(drop=0.5, seed=3),
    'FRP x 4': dict(frp=4.0),
    'FRP x 0.25': dict(frp=0.25),
    'fuels 4 points drier': dict(dry=4.0),
    'fuels 4 points wetter': dict(dry=-4.0),
    'wind x 2': dict(wind=2.0),
    'canopy base height x 0.4': dict(cbh=0.4),
    'first 3 steps': dict(steps=3),
    'combined': dict(drop=0.3, seed=11, frp=2.0, dry=2.0, wind=1.5, cbh=0.7, steps=5),
}
SHORT = {'NM3624410503720220417': 'Cooks Peak, NM', 'OK3496309538920170202': 'Powder Mill, OK',
         'UT4071511223920180915': 'Black, UT', 'WA4757812313020180804': 'Maple, WA'}


def mulberry32(seed):
    """The site kit's rng(seed).u (site/src/instruments/kit.ts), bit for bit."""
    a = seed & 0xFFFFFFFF

    def u():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (1 | a)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return u


def apply(fire, v):
    """A what-if on one fire's raw observations (dict as demo.load_fire returns). Returns a new fire dict."""
    v = {**BASE, **v}
    u = mulberry32(int(v['seed']))
    dets = []
    for d in fire['dets']:
        keep = np.array([u() >= v['drop'] for _ in range(len(d['frp']))], bool) if v['drop'] > 0 else np.ones(len(d['frp']), bool)
        dets.append(dict(row=np.asarray(d['row'])[keep], col=np.asarray(d['col'])[keep],
                         frp=np.asarray(d['frp'])[keep] * v['frp']))
    wx = []
    for w in fire['wx']:
        w = dict(w)
        w['fm100'] = max(w['fm100'] - v['dry'], 1.0)
        w['fm1000'] = max(w['fm1000'] - v['dry'], 1.0)
        w['vs'] = w['vs'] * v['wind']
        wx.append(w)
    layers = dict(fire['layers'])
    if v['cbh'] != 1.0:
        layers['cbh'] = (layers['cbh'] * np.float32(v['cbh'])).astype(np.float32)
    n = min(int(v['steps']), len(fire['masks']))
    return dict(masks=fire['masks'][:n], dets=dets[:n], day=fire['day'][:n], wx=wx[:n], layers=layers,
                doy=fire['doy'], gamma=fire['gamma'])


def main():
    out = instrument_dir('firedetective')
    model = fdet.load()
    W = model.W
    cal = model.cal
    # ---- weights pack: float16 member tensors, float32 normalisation
    norm = ('seq_mu', 'seq_sd', 'st_mu', 'st_sd')
    tensors = {k: W[k] for k in sorted(W)}
    keep = {k: 'f32' for k in norm}
    members = sorted({k.split('.')[0] for k in W if '.' in k})
    meta = {'members': members, 'calib': {k: cal[k] for k in ('hs_map', 'hs_widen', 'wind_widen', 'lfmc_widen')},
            'seq_names': feats.SEQ_NAMES, 'stat_names': feats.STAT_NAMES, 'wind_cols': feats.WIND_COLS,
            'params': int(sum(v.size for k, v in W.items() if '.' in k)), 'version': '1.0',
            'note': 'FireDetective v1.0, 3-seed ensemble of 1-D CNNs; member tensors float16, normalisation float32'}
    write_pack(out / 'weights', tensors, dtype='f16', meta=meta, keep=keep)
    Wr = {k: (np.asarray(v).astype(np.float32) if k in norm else np.asarray(v).astype(np.float16)).astype(np.float64)
          for k, v in tensors.items()}
    rt = fdet.FireDetective(Wr, cal)

    # ---- examples pack (the raw observations of examples/*.npz) and reference cases
    arrays, fires, cases = {}, [], []
    worst16 = 0.0
    for i, f in enumerate(sorted((MODEL / 'examples').glob('*.npz'))):
        fire, m = demo.load_fire(str(f))
        arrays[f'masks{i}'] = fire['masks'].astype(np.uint8)
        for k in LAYERS:
            a = fire['layers'][k]
            if k == 'fuel':
                assert a.min() >= 0 and a.max() <= 255
                arrays[f'{k}{i}'] = a.astype(np.uint8)
            else:
                assert a.dtype == np.float32
                arrays[f'{k}{i}'] = a
        rel = pipeline.run([fire], model)
        fires.append({
            'fid': m['fid'], 'name': m['name'], 'short': SHORT.get(m['fid'], m['name']), 'year': m['year'],
            'period': m['period'], 'region': m['region'], 'stratum': m['stratum'], 'steps': m['steps'],
            'day': [bool(x) for x in m['day']], 'dets': m['dets'], 'wx': m['wx'], 'doy': m['doy'], 'gamma': m['gamma'],
            'shape': list(fire['masks'].shape),
            'truth': {k: (None if v is None or not np.isfinite(v) else float(v)) for k, v in m['truth'].items()},
            'classical': {k: (None if v is None or not np.isfinite(v) else float(v)) for k, v in m['classical'].items()},
            'release': {k: float(rel[k][0]) for k in OUTS}})
        for vname, v in VARIANTS.items():
            g = apply(fire, v)
            seq, seq_h, st = pipeline.features(g['masks'], g['dets'], g['day'], g['wx'], g['layers'], g['doy'], g['gamma'])
            o = rt.predict(seq[None], seq_h[None], st[None])
            if vname == 'recorded':
                worst16 = max(worst16, max(abs(float(o[k][0]) - rel[k][0]) / max(1.0, abs(float(rel[k][0]))) for k in OUTS))
            hid = [float(x) for x in seq_h[:, feats.WIND_COLS + [feats.SEQ_NAMES.index('wind_hidden')]].ravel()]
            cases.append({'example': i, 'variant': vname, **{**BASE, **v},
                          'outputs': [float(o[k][0]) for k in OUTS] + [float(x) for x in seq.ravel()] + hid
                          + [float(x) for x in st]})
        print(m['fid'], m['name'], m['stratum'], fire['masks'].shape)
    write_pack(out / 'examples', arrays, dtype='f32', meta={'fires': fires, 'layers': list(LAYERS)},
               keep={k: 'u8' for k in arrays if k.startswith(('masks', 'fuel'))})
    write_json(out / 'reference.json', {
        'cases': cases, 'tolerance': 1e-6,
        'note': 'outputs = 13 readable outputs (' + ', '.join(OUTS) + '), 12 x 25 sequence features, the wind-blind '
                'copy\'s wind columns and flag (12 x 4), 17 static features; weights float16-roundtripped'})
    size = sum(p.stat().st_size for p in out.iterdir())
    print(f'{len(cases)} reference cases; float16 vs released weights on the recorded inputs: worst relative output '
          f'difference {worst16:.1e}; {size / 1e6:.2f} MB in {out}')


if __name__ == '__main__':
    main()
