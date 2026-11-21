"""Export FireSize's live instrument: weights, the eight real exam ignitions and Python reference cases.

    python web/export_instrument.py      (from models/firesize/, or anywhere; writes site/public/instruments/firesize/)

- weights.{json,bin}: the 3-member MLP ensemble. Weight matrices and biases are float16; the input normalisation
  (mu, sd, fill) and the Rothermel lookup-table rows of the fuel codes the examples use (ros_lut.npz is float32
  already) stay float32. meta: the 75 feature names, the temperature, class edges, the dev-fitted ERC coupling.
- examples.{json,bin}: codes (8 x 169 int32): the 13 x 13 LANDFIRE FBFM40 patches of examples/patches.json. meta: the
  records and the 75 network inputs of examples/demo_examples.csv (JSON numbers, so the browser starts from the very
  doubles Python parses), the real size, and the release's float64 reading on the recorded inputs.
- reference.json: every ignition under 13 what-ifs. Each case runs the what-if below, recomputes the Rothermel inputs
  with firesize.py's own simard_emc() and ros_mmin(), then FireSize.predict() with the float16-roundtripped weights.
  outputs = 12 widened class probabilities, 12 raw (T = 1) ones, P(escape), P(large), median acres, the 80 % range
  (2), then the 75 network inputs, so parity covers the feature port and the network.

The what-ifs (mirrored exactly in site/src/instruments/firesize/net.ts):
  dWind   m/s added to the daily mean wind of both days (vs, vs_m1), floored at 0
  dRH     points added to RHmin of both days, clipped to [1, 100]
  dT      deg C added to Tmax of both days (Tmin, RHmax and VPD stay recorded)
  dERC    added to ERC of both days (floored at 0); BI, FM100 and FM1000 of both days move along the dev-fitted
          slopes (per ERC unit: BI +0.821, FM100 -0.175, FM1000 -0.197; least squares over the dev ignitions, from the
          shipped page), BI floored at 0 and the fuel moistures at 1 %
  hour    local discovery hour (sets hr_sin, hr_cos, hr_miss = 0); null keeps the record
  cause   0 human, 1 natural, 2 missing/undetermined (one-hot c_hum, c_nat, c_miss); null keeps the record
  fed     0 state/local, 1 federal reporting agency; null keeps the record
  region  index into REGIONS (one-hot r_*); null keeps the record
If any of the four weather shifts is non-zero, ros_pt, ros5, ros13 and ros_pt_m1 are recomputed from the fuel patch
(Simard EMC at Tmax and RHmin, 10 m wind = 1.45 x the daily mean), as firesize.make_features does; otherwise the
exam's own columns are kept.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[1]
REPO = MODEL.parents[1]
sys.path.insert(0, str(MODEL))
sys.path.insert(0, str(REPO))
from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402
import firesize as fz  # noqa: E402

# least-squares slopes against ERC over the dev ignitions (build_release.py's erc_coupling, from the shipped page)
ERC_COUPLING = {'bi': 0.8210495640156434, 'fm100': -0.17485057933038436, 'fm1000': -0.19738818683196405}
REGION_NAMES = {'CA': 'California', 'NW': 'Northwest', 'NR': 'Northern Rockies', 'GB': 'Great Basin',
                'SW': 'Southwest', 'RM': 'Rocky Mountain', 'SP': 'Southern Plains', 'SE': 'Southeast',
                'NE_MW': 'Northeast / Midwest'}
CAUSE_KEYS = ['c_hum', 'c_nat', 'c_miss']
BASE = dict(dWind=0.0, dRH=0.0, dT=0.0, dERC=0.0, hour=None, cause=None, fed=None, region=None)
VARIANTS = {
    'recorded': {},
    'windier +4 m/s': dict(dWind=4.0),
    'calmer -2 m/s': dict(dWind=-2.0),
    'hotter, drier: Tmax +5, RHmin -10': dict(dT=5.0, dRH=-10.0),
    'humid: RHmin +20': dict(dRH=20.0),
    'drier fuels: ERC +25': dict(dERC=25.0),
    'wetter fuels: ERC -20': dict(dERC=-20.0),
    'found at 15:00': dict(hour=15.0),
    'lightning': dict(cause=1),
    'federal agency': dict(fed=1),
    'moved to California': dict(region=0),
    'moved to the Southeast': dict(region=7),
    'combined': dict(dWind=3.0, dRH=-8.0, dT=3.0, dERC=15.0, hour=16.5, cause=0, fed=1, region=4),
}


def apply(x, codes, w):
    """x: dict of the 75 recorded inputs; codes: 13 x 13 FBFM40. Returns the what-if inputs (a new dict)."""
    w = {**BASE, **w}
    x = dict(x)
    if w['dWind'] != 0:
        for k in ('vs', 'vs_m1'):
            x[k] = float(np.maximum(0.0, x[k] + w['dWind']))
    if w['dRH'] != 0:
        for k in ('rmin', 'rmin_m1'):
            x[k] = float(np.minimum(100.0, np.maximum(1.0, x[k] + w['dRH'])))
    if w['dT'] != 0:
        for k in ('tmmx', 'tmmx_m1'):
            x[k] = x[k] + w['dT']
    if w['dERC'] != 0:
        d = w['dERC']
        for s in ('', '_m1'):
            x['erc' + s] = float(np.maximum(0.0, x['erc' + s] + d))
            x['bi' + s] = float(np.maximum(0.0, x['bi' + s] + ERC_COUPLING['bi'] * d))
            for k in ('fm100', 'fm1000'):
                x[k + s] = float(np.maximum(1.0, x[k + s] + ERC_COUPLING[k] * d))
    if any(w[k] != 0 for k in ('dWind', 'dRH', 'dT', 'dERC')):
        m1 = np.clip(fz.simard_emc(x['tmmx'], x['rmin']), 1, 35)
        u = x['vs'] * 1.45
        R = np.log1p(fz.ros_mmin(codes, np.full((13, 13), m1), np.full((13, 13), u)))
        x['ros_pt'] = float(R[6, 6])
        x['ros5'] = float(R[4:9, 4:9].mean())
        x['ros13'] = float(R.mean())
        m1b = np.clip(fz.simard_emc(x['tmmx_m1'], x['rmin_m1']), 1, 35)
        x['ros_pt_m1'] = float(np.log1p(fz.ros_mmin(int(codes[6, 6]), m1b, x['vs_m1'] * 1.45)))
    if w['hour'] is not None:
        h = float(w['hour'])
        x['hr_sin'], x['hr_cos'], x['hr_miss'] = math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24), 0.0
    if w['cause'] is not None:
        for i, k in enumerate(CAUSE_KEYS):
            x[k] = float(i == w['cause'])
    if w['fed'] is not None:
        x['fed'] = float(w['fed'])
    if w['region'] is not None:
        for i, r in enumerate(fz.REGIONS):
            x[f'r_{r}'] = float(i == w['region'])
    return x


def reading(m, x):
    out = m.predict({k: np.array([x[k]]) for k in m.features})
    p1 = m.probs({k: np.array([x[k]]) for k in m.features}, temp=1.0)
    return out, p1


def main():
    out_dir = instrument_dir('firesize')
    m = fz.FireSize()
    W = m.W
    n_ens = int(W['n_ens'])
    n_layers = 0
    while f'e0_L{n_layers}_w' in W:
        n_layers += 1
    with open(MODEL / 'examples' / 'demo_examples.csv', newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    patches = json.loads((MODEL / 'examples' / 'patches.json').read_text(encoding='utf-8'))
    assert [int(r['FOD_ID']) for r in rows] == [p['FOD_ID'] for p in patches]
    codes = np.array([np.array(p['fuel13'], int) for p in patches])          # 8 x 13 x 13
    lut_codes = sorted({int(c) for c in np.unique(codes)})

    # ---- weights pack: f16 layers, f32 normalisation and LUT rows
    full_lut = np.load(MODEL / 'ros_lut.npz')['lut']
    assert full_lut.dtype == np.float32
    tensors, keep = {}, {}
    for e in range(n_ens):
        for k in range(n_layers):
            tensors[f'e{e}_L{k}_w'] = W[f'e{e}_L{k}_w']
            tensors[f'e{e}_L{k}_b'] = W[f'e{e}_L{k}_b']
    for k in ('mu', 'sd', 'fill'):
        tensors[k] = W[k]
        keep[k] = 'f32'
    tensors['lut'] = full_lut[lut_codes]
    keep['lut'] = 'f32'
    E = [float(v) for v in fz._E]
    meta = {'feats': m.features, 'temp': float(W['temp']), 'n_ens': n_ens, 'n_layers': n_layers,
            'edges_ac': fz.EDGES_AC.tolist(), 'E': E, 'class_labels': fz.CLASS_LABELS, 'esc_bin': fz.ESC_BIN,
            'large_bin': fz.LARGE_BIN, 'regions': fz.REGIONS, 'region_names': REGION_NAMES, 'causes': fz.CAUSES,
            'fcls': fz.FCLS, 'wxf': fz.WXF, 'erc_coupling': ERC_COUPLING, 'lut_codes': lut_codes,
            'lut_m1': fz.M1.tolist(), 'lut_u10': fz.U10.tolist(), 'version': '1.0',
            'note': 'FireSize v1.0, 3-member MLP ensemble; layers float16, normalisation and ROS table float32'}
    write_pack(out_dir / 'weights', tensors, dtype='f16', meta=meta, keep=keep)

    # the weights and table the browser sees, back in the model's own class and functions
    rt = fz.FireSize()
    Wr = dict(W)
    for k, v in roundtrip({k: v for k, v in tensors.items() if k not in keep}, 'f16').items():
        Wr[k] = v.astype(np.float64)
    for k in ('mu', 'sd', 'fill'):
        Wr[k] = np.asarray(W[k]).astype(np.float32).astype(np.float64)
    rt.W = Wr
    lut = np.zeros_like(full_lut)
    lut[lut_codes] = full_lut[lut_codes]
    fz._LUT = lut                                    # rows the examples never touch are zero in the browser too

    # ---- examples pack and reference cases
    ign, cases = [], []
    worst16 = 0.0
    for i, (r, p) in enumerate(zip(rows, patches)):
        x0 = {k: float(r[k]) for k in m.features}
        assert all(math.isfinite(v) for v in x0.values())
        full, _ = reading(m, x0)                      # float64 release
        ign.append({'fod_id': int(r['FOD_ID']), 'fire_name': r['fire_name'], 'name': p['name'], 'year': int(r['year']),
                    'state': r['state'], 'date': p['discovery_date'], 'cause': p['nwcg_general_cause'],
                    'fuel': p['fuel_at_point'], 'size_class': p['fire_size_class'], 'size_bin': p['size_bin'],
                    'real_acres': float(r['real_acres']), 'exam_p_escape': float(r['exam_p_escape']),
                    'x': [x0[k] for k in m.features],
                    'release': {'p_escape': float(full['p_escape'][0]), 'p_large': float(full['p_large'][0]),
                                'median': float(full['median_acres'][0]),
                                'lo': float(full['interval80_acres'][0, 0]), 'hi': float(full['interval80_acres'][0, 1])}})
        for vname, v in VARIANTS.items():
            x = apply(x0, codes[i], v)
            o, p1 = reading(rt, x)
            if vname == 'recorded':
                worst16 = max(worst16, abs(float(o['p_escape'][0]) - float(full['p_escape'][0])))
            cases.append({'example': i, 'variant': vname, **{**BASE, **v},
                          'outputs': [*map(float, o['probs'][0]), *map(float, p1[0]), float(o['p_escape'][0]),
                                      float(o['p_large'][0]), float(o['median_acres'][0]),
                                      float(o['interval80_acres'][0, 0]), float(o['interval80_acres'][0, 1]),
                                      *[float(x[k]) for k in m.features]]})
        print(f"{p['name']:20s} P(escape) {full['p_escape'][0]:.4f} (exam {float(r['exam_p_escape']):.4f})")
    write_pack(out_dir / 'examples', {'codes': codes.reshape(len(codes), -1)}, dtype='i32', meta={'ignitions': ign})
    write_json(out_dir / 'reference.json', {
        'cases': cases, 'tolerance': 1e-9,
        'note': 'outputs = 12 widened probabilities, 12 raw (T = 1), P(escape), P(large), median acres, 80 % range lo, '
                'hi, then the 75 inputs; weights float16-roundtripped'})
    size = sum(q.stat().st_size for q in out_dir.iterdir())
    print(f'{len(cases)} reference cases; float16 vs float64 release on the recorded inputs: worst |dP(escape)| '
          f'{worst16:.1e}; {size / 1e3:.0f} kB in {out_dir}')


if __name__ == '__main__':
    main()
