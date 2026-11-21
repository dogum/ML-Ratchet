"""Export FireCast's released weights, the three real exam fire-days of examples/ and Python reference cases for the
browser instrument (site/public/instruments/firecast/).

Packs:
  weights.{json,bin}  the released U-Net (rung r2, BatchNorm folded), float16; the dev calibration T, b and the
                      yes/no threshold exact in meta
  days.{json,bin}     per fire-day i, on the FULL fire grid: now<i> (burned at issue, packed bits), pm<i> / am<i>
                      (VIIRS detections at the issue and the night overpass: row, col, ps, pt, frp; float32).
                      On the map WINDOW (exam.py's rule: <= 240 x 240 cells, multiples of 16, centred on the burned
                      area; the published page used the same windows): grew<i> (burned now, not 24 h earlier), fl<i>
                      (FEDS active fire line), nxt<i> (truth: burned by the next afternoon overpass), all packed bits,
                      and off<i> (2 x H x W uint8: row and column offset from each cell to its nearest fire-line
                      cell, as scipy's feature transform picked it, clipped to +-112 cells and stored +112).
                      meta: names, dates, window, grid convergence, the 16 gridMET features, the fire scalars, the
                      exam-run scores.

The browser rebuilds the 22 input maps from these: VIIRS FRP rasters and the distances to the burned area and to the
nearest hot spot by an exact Euclidean distance transform on the full grid, the fire-line distance and the
along/across-wind offsets from off<i> and tomorrow's wind. Then the network runs on the window, as the published
page's lab did. Reference cases run firecast.py itself (make_inputs with crop=window, FireCastNP.forward) on the
float32 detections the browser holds, with the float16-roundtripped weights: recorded and with what-ifs on the input
(tomorrow's wind speed and direction, minimum humidity, 100-h fuel moisture, VIIRS hot spots removed, yesterday's
perimeter withheld). Each case keeps the 22 scalar inputs and the logits on every 5th cell of the window.

Run from the repository root: python models/firecast/web/export_instrument.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

MODEL = Path(__file__).resolve().parents[1]
REPO = MODEL.parents[1]
sys.path.insert(0, str(MODEL))
sys.path.insert(0, str(REPO))
import firecast as fc  # noqa: E402
from demo import load_days  # noqa: E402
from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

# Grid convergence (grid north = true north + gamma, degrees) of each fire, from exam.py examples() (firesim
# realdata.grid_convergence_deg at the fire centre), as stored in the published page's data. Only the instrument's
# "wind from" slider needs it (to turn a true bearing into the grid features); the recorded features are exact.
GAMMA = {'LA3098109281120230824': -1.9090198985181657, 'CA4176412395720150801': 16.85538278554988,
         'UT3980611166020180906': 9.363181051274903}
PAGE_WINDOW = {'LA3098109281120230824': (3, 1, 160, 144), 'CA4176412395720150801': (0, 0, 128, 144),
               'UT3980611166020180906': (186, 146, 240, 240)}
WIN = 240
CLIP = 112          # cells; 112 x 90 m = 10.08 km > the 10 km clip of every offset feature
STEP, START = 5, 2  # logits kept on rows/cols START::STEP of the window


def window(now, win=WIN):
    """exam.py examples(): a <= win x win window (multiples of 16) centred on the burned area at issue."""
    H, W = now.shape
    rr, cc = np.nonzero(now)
    hh, ww = min(win, H // 16 * 16), min(win, W // 16 * 16)
    r0 = int(np.clip(rr.mean() - hh / 2, 0, H - hh))
    c0 = int(np.clip(cc.mean() - ww / 2, 0, W - ww))
    return r0, c0, hh, ww


def fl_offsets(now, fl):
    """The nearest fire-line cell of every cell, as full_fields() finds it (fire line, else the burned edge)."""
    flb = fl.astype(bool) & now if fl is not None else np.zeros(now.shape, bool)
    src = flb if flb.any() else (now & ~ndimage.binary_erosion(now, border_value=0))
    _, (ir, ic) = ndimage.distance_transform_edt(~src, sampling=fc.RES, return_indices=True)
    rr, cc = np.indices(now.shape)
    return rr - ir, cc - ic          # off_e = dc * 90 m, off_n = -dr * 90 m


def f32_pix(p):
    return {k: np.asarray(v, np.float32).astype(np.float64) for k, v in p.items()}


def wind_wx(wx, d1, gamma):
    """Tomorrow's gridMET values changed: recompute only those entries of the 16 features, with firecast's own
    feature_vector (indices 8-15 are tomorrow: toward-east, toward-north, speed, rmin, tmmx, vpd, pr, fm100)."""
    wx = np.array(wx, np.float32)
    keys = ('vs', 'th', 'rmin', 'tmmx', 'vpd', 'pr', 'fm100')
    f = fc.feature_vector({k: d1.get(k, np.nan) for k in keys}, {k: d1.get(k, np.nan) for k in keys}, gamma)
    if 'vs' in d1:
        wx[8:11] = f[0:3]
    if 'rmin' in d1:
        wx[11] = f[3]
    if 'fm100' in d1:
        wx[15] = f[7]
    return wx


def recorded_d1(wx, gamma):
    """Tomorrow's gridMET values recovered from the stored features (for slider start positions)."""
    vs = float(wx[10]) * 5.0
    th = (np.degrees(np.arctan2(float(wx[8]), float(wx[9]))) - 180.0 - gamma) % 360.0
    return dict(vs=vs, th=th, rmin=float(wx[11]) * 15.0 + 20.0, fm100=float(wx[15]) * 4.0 + 9.0)


def case(net, d, win, ops):
    pm, am = d['pix_pm'], d['pix_am']
    if ops.get('viirs_off'):
        pm = am = {k: np.zeros(0) for k in ('row', 'col', 'ps', 'pt', 'frp')}
    has_prev = bool(d['has_prev']) and not ops.get('prev_off')
    wx = d['wx'] if 'd1' not in ops else wind_wx(d['wx'], ops['d1'], ops['gamma'])
    r0, c0, hh, ww = win
    X, s = fc.make_inputs(d['now'], d['prev'], d['fl'], pm, am, None, wx, has_prev, bool(d['first']),
                          crop=(r0, r0 + hh, c0, c0 + ww))
    lg = net.forward(X, s)
    keep = lg[START::STEP, START::STEP]
    return [float(v) for v in s] + [round(float(v), 6) for v in keep.ravel()], X, s, lg


def window_scores(d, win, prob, thr):
    r0, c0, hh, ww = win
    sl = (slice(r0, r0 + hh), slice(c0, c0 + ww))
    now = d['now'][sl]
    dist = ndimage.distance_transform_edt(~d['now'], sampling=fc.RES)[sl]
    return fc.nextday(now, d['nxt'][sl], prob, fc.CELL_KM2, thr=thr, window=dist <= 10000.0)


def main():
    out = instrument_dir('firecast')
    base = fc.FireCastNP()
    with np.load(MODEL / 'firecast_weights.npz') as z:
        W = {k: z[k] for k in z.files if k not in ('cal_T', 'cal_b', 'thr')}
        cal = {k: float(z[k]) for k in ('cal_T', 'cal_b', 'thr')}
    net = fc.FireCastNP()
    net.p.update(roundtrip(W, 'f16'))                       # what the browser sees; calibration exact
    write_pack(out / 'weights', W, dtype='f16', meta=dict(cal, rung='r2', channels=fc.SPATIAL))

    days = load_days(MODEL / 'examples' / 'demo_days.npz')
    arrays, info, cases, keep_dt = {}, [], [], {}
    for i, d in enumerate(days):
        m = d['meta']
        d['pix_pm'], d['pix_am'] = f32_pix(d['pix_pm']), f32_pix(d['pix_am'])   # float32, as the browser holds them
        win = window(d['now'])
        assert win == PAGE_WINDOW[m['fire_id']], (win, m['fire_id'])
        r0, c0, hh, ww = win
        sl = (slice(r0, r0 + hh), slice(c0, c0 + ww))
        now = d['now']
        prev = d['prev'] if d['prev'] is not None else np.zeros_like(now)
        fl = d['fl'] if d['fl'] is not None else np.zeros_like(now)
        dr, dc = fl_offsets(now, d['fl'])
        arrays[f'now{i}'] = np.packbits(now)
        for k, p in (('pm', d['pix_pm']), ('am', d['pix_am'])):
            arrays[f'{k}{i}'] = np.stack([p[c] for c in ('row', 'col', 'ps', 'pt', 'frp')]).astype(np.float32).reshape(5, -1)
        arrays[f'grew{i}'] = np.packbits(now[sl] & ~prev[sl])
        arrays[f'fl{i}'] = np.packbits(fl[sl].astype(bool) & now[sl])
        arrays[f'nxt{i}'] = np.packbits(d['nxt'][sl] | now[sl])
        arrays[f'off{i}'] = (np.stack([np.clip(dr[sl], -CLIP, CLIP), np.clip(dc[sl], -CLIP, CLIP)]) + CLIP).astype(np.uint8)
        for k in arrays:
            keep_dt[k] = 'f32' if k[:2] in ('pm', 'am') else 'u8'
        # fire scalars as make_inputs computes them (recorded inputs)
        area = float(now.sum() * fc.RES * fc.RES / 1e6)
        area_prev = float(prev.sum() * fc.RES * fc.RES / 1e6)
        bnd = now & ~ndimage.binary_erosion(now, border_value=0)
        fl_frac = min(float((fl.astype(bool) & now).sum()) / max(int(bnd.sum()), 1), 1.0)
        gamma = GAMMA[m['fire_id']]
        rec, X, s, lg = case(net, d, win, {})
        prob = 1.0 / (1.0 + np.exp(-(lg / net.cal_T + net.cal_b)))
        sc = window_scores(d, win, prob, net.thr)
        # the released float32 weights on the same window, for the f16 note
        X0, s0 = fc.make_inputs(now, d['prev'], d['fl'], d['pix_pm'], d['pix_am'], None, d['wx'], bool(d['has_prev']),
                                bool(d['first']), crop=(r0, r0 + hh, c0, c0 + ww))
        p0 = base.prob(X0, s0)
        info.append(dict(fire_id=m['fire_id'], name=m['name'], date=m['date'], k=m['k'], period=m['period'],
                         stratum=m['stratum'], shape=list(now.shape), window=list(win), gamma=gamma,
                         wx=[float(v) for v in d['wx']], has_prev=bool(d['has_prev']), first=bool(d['first']),
                         area_km2=area, area_prev_km2=area_prev, fl_frac=fl_frac,
                         d1=recorded_d1(d['wx'], gamma),
                         exam=dict(ap=m['exam_ap'], f1=m['exam_f1'], fl_prev_ap=m['fl_prev_ap'],
                                   fl_prev_f1=m['fl_prev_f1'], new_obs_km2=m['new_obs_km2']),
                         window_scores=dict(ap=sc['ap'], f1=sc['f1'], new_obs_km2=sc['new_obs_km2'],
                                            new_pred_km2=sc['new_pred_km2']),
                         f16_max_dprob=float(np.abs(prob - p0).max())))
        d1 = info[-1]['d1']
        th = round(d1['th'] / 5) * 5
        variants = [
            ('recorded', {}),
            ('wind 2 m/s', {'d1': {'vs': 2.0, 'th': th}}),
            ('wind 12 m/s', {'d1': {'vs': 12.0, 'th': th}}),
            ('wind turned 90 degrees', {'d1': {'vs': round(d1['vs'] * 2) / 2, 'th': (th + 90) % 360}}),
            ('wind reversed', {'d1': {'vs': round(d1['vs'] * 2) / 2, 'th': (th + 180) % 360}}),
            ('min humidity 8 %', {'d1': {'rmin': 8.0}}),
            ('min humidity 60 %', {'d1': {'rmin': 60.0}}),
            ('100-h fuel moisture 4 %', {'d1': {'fm100': 4.0}}),
            ('100-h fuel moisture 25 %', {'d1': {'fm100': 25.0}}),
            ('VIIRS hot spots removed', {'viirs_off': True}),
            ('hot, dry, windy: 10 m/s, rmin 6 %, fm100 4 %, no hot spots',
             {'d1': {'vs': 10.0, 'th': th, 'rmin': 6.0, 'fm100': 4.0}, 'viirs_off': True}),
        ]
        if d['has_prev']:
            variants.append(("yesterday's perimeter withheld", {'prev_off': True}))
        for name, ops in variants:
            if 'd1' in ops:
                ops = dict(ops, gamma=gamma)
            outputs = case(net, d, win, ops)[0]
            cases.append(dict(example=i, variant=name, day=i, ops={k: v for k, v in ops.items() if k != 'gamma'},
                              outputs=outputs))
        print(f"{m['name']}: window {win} of {now.shape}; window AP {sc['ap']:.3f} F1 {sc['f1']:.3f}; "
              f"f16 vs f32 max |dp| {info[-1]['f16_max_dprob']:.2e}")
    write_pack(out / 'days', arrays, dtype='u8', keep=keep_dt,
               meta={'days': info, 'clip': CLIP, 'step': STEP, 'start': START, 'res_m': fc.RES})
    write_json(out / 'reference.json', dict(cases=cases, tolerance=1e-4, step=STEP, start=START))
    size = sum(p.stat().st_size for p in out.iterdir())
    print(f'{len(days)} fire-days, {len(cases)} parity cases, {size / 1e6:.2f} MB in {out}')


if __name__ == '__main__':
    main()
