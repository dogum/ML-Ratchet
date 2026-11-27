"""Solar reporting offset read at issue time (rung r9_align, 15 Nov 2026). Numpy, from the shipped weights.

v1.0 shifts CISO's and PJM's EIA-930 solar by one fixed hour learned on the dev years (real.py, renew_dev.json `lag`).
EIA-930 solar timestamps are not stable from year to year, so r9 reads the offset of the 168 h history before issue
(demandcast.solar_offset: the reported-minus-physics error regressed on the physics diurnal profile and its slope),
puts the history back on the physics clock, runs the same network and moves the forecast by the offset. Only inputs
available at issue are used; the exam day's own output never enters. The pipeline, the truth and the baselines are
unchanged (the fixed dev shift stays in real.py), so v1.0 and r9 are scored on the same rows.

    python solar_align.py dev        # offsets read on the dev bank; dev gate errors v1.0 vs r9
    python solar_align.py stress     # dev gate days with the reported solar moved by -1 / 0 / +1 h
    python solar_align.py y2022      # CISO and PJM 2022 (training-input year, outside dev and exam): diagnostic only
    python solar_align.py val <val>  # frozen simulated set (val_v4): objective v1.0 vs r9, offsets read vs true
    python solar_align.py add <val>  # val + dev gate -> decision (ratchet rule) appended to training/ratchet.json
"""
import os, sys, json
import numpy as np
from common import *
import real
from train import masks, XK, PK
import evaluate as E
from ratchet import gate_rows

MODEL = os.path.dirname(SRC)
sys.path.insert(0, MODEL)
import demandcast as DC  # noqa: E402

M = DC.DemandCast(DC.WEIGHTS_V1_1, DC.META_V1_1)     # the r8 network these rungs were scored with
LOG = os.path.join(SRC, 'ratchet.json')


_REF = None


def dev_ref():
    """Per grid: median offset estimate r_hat over every dev day (2019, 2021), the clock the fixed dev shift left the
    reported solar on. A dev calibration, like the shift itself."""
    global _REF
    if _REF is None:
        Bd = dict(np.load(f'{DATA}/banks/real_dev.npz'))
        _, rh = DC.solar_offset(np.nan_to_num(Bd['xs'].astype(np.float64)))
        _REF = {ba: float(np.median(rh[(Bd['meta_ba'] == ba) & Bd['hass']])) for ba in BAS
                if ((Bd['meta_ba'] == ba) & Bd['hass']).any()}
    return _REF


def ref_rows(B, align):
    """'abs' (r9a): offsets counted from the physics clock; 'ref' (r9b): from each grid's dev reference. Simulated
    regions report on the physics clock apart from their drawn offset, so their reference is 0."""
    if align != 'ref' or 'meta_region' in B or 'meta_ba' not in B:
        return 0.0
    R = dev_ref()
    return np.array([R.get(b, 0.0) for b in B['meta_ba']])


def predict(B, align, k_only=None):
    """evaluate.predict with the numpy forward (raw sigmas); align 'abs' / 'ref' are rungs r9a / r9b for solar."""
    out = {}
    for k in ('d', 'w', 's') if k_only is None else (k_only,):
        X = np.nan_to_num(B[XK[k]].astype(np.float32)).astype(np.float64)
        r = np.zeros(len(X), int)
        if k == 's' and align:
            r, _ = DC.solar_offset(X, ref_rows(B, align)); X = DC.align_solar_x(X, r)
        P = B[PK[k]].astype(np.float64) if k != 'd' else None
        mu, sig = DC._forward(M.W[k], X, k, P)
        if k != 'd':
            mu = np.clip(mu, 0, 1.2)
        if k == 's':
            night = B['night'] > 0.5
            mu = np.where(night, 0.0, mu); sig = np.where(night, DC.FLOOR['s'], sig)
            if r.any():
                mu = DC._shift(mu, r, 0.0); sig = DC._shift(sig, r, DC.FLOOR['s'])
        out[k] = (mu.astype(np.float32), sig.astype(np.float32))
    return out


def solar_err(B, pred, rows=None):
    """Sum |err| / sum output over all hours with a truth (capacity weighted), as exam.block: model, persistence."""
    y = B['ys']; mu = pred['s'][0]
    m = np.isfinite(y) & B['hass'][:, None] if 'hass' in B else np.isfinite(y)
    if 's_pers' in B: m &= np.isfinite(B['s_pers'])
    if rows is not None: m &= rows[:, None]
    w = np.broadcast_to(B['caps'][:, None], y.shape)
    f = lambda p: float(np.sum((np.abs(p - y) * w)[m]) / max(np.sum((np.nan_to_num(y) * w)[m]), 1e-9) * 100)
    o = dict(model=f(mu), hours=int(m.sum()))
    if 's_pers' in B: o['persistence'] = f(B['s_pers'])
    o['physics'] = f(B['ps'])
    return o


def dev():
    Bd = dict(np.load(f'{DATA}/banks/real_dev.npz')); rows = gate_rows(Bd)
    r, rh = DC.solar_offset(np.nan_to_num(Bd['xs'].astype(np.float64)))
    hs = Bd['hass']
    print('dev bank: solar rows', int(hs.sum()), '| offset read -1/0/+1:', [int(((r == k) & hs).sum()) for k in (-1, 0, 1)])
    for ba in BAS:
        s = (Bd['meta_ba'] == ba) & hs
        if s.any():
            print(f'  {ba}: {[int(((r == k) & s).sum()) for k in (-1, 0, 1)]}  r_hat median {np.median(rh[s]):+.2f} '
                  f'p5 {np.percentile(rh[s], 5):+.2f} p95 {np.percentile(rh[s], 95):+.2f}')
    res = {}
    print('dev reference r_hat per grid:', {k: round(v, 3) for k, v in dev_ref().items()})
    for name, al in (('v1.0', False), ('r9a', 'abs'), ('r9b', 'ref')):
        p = predict(Bd, al)
        res[name] = {k: E.errors(Bd, p, k, rows=rows) for k in ('d', 'w', 's')}
        res[name]['s_all_dev'] = E.errors(Bd, p, 's')
        res[name]['sigma_scale_s'] = E.sigma_scale(Bd, p, 's')
        print(name, 'gate: d %.3f w %.2f s %.2f | solar all dev days %.2f | solar sigma scale %.4f' % (
            res[name]['d']['err'], res[name]['w']['err'], res[name]['s']['err'], res[name]['s_all_dev']['err'],
            res[name]['sigma_scale_s']))
    rel = {v: float(np.mean([res[v][k]['err'] / res['v1.0'][k]['err'] for k in ('d', 'w', 's')])) for v in ('r9a', 'r9b')}
    print('dev gate: mean relative error vs v1.0 (gate <= 1.02):', rel)
    return res, rel


def build_rows(ba, days, calib, shift, fcs=None):
    """Real rows for (ba, days) with the reported solar moved by `shift` hours (reported[t] = original[t - shift])."""
    real._CACHE.clear()
    g = real.real_region(ba, calib)
    ys0 = g.ys.copy(); g.ys = np.roll(ys0, shift)
    acc = {k: [] for k in ('xs', 'ys', 'ps', 'night', 'caps', 's_pers')}
    for day in days:
        i0 = day_start_index(ba, day)
        s = g.sample(i0, None)
        if 'xs' not in s: continue
        for k in ('xs', 'ys', 'ps', 'night'): acc[k].append(s[k])
        acc['caps'].append(s['caps']); acc['s_pers'].append(g.ys[i0 - 24:i0] / max(s['caps'], 1))
    g.ys = ys0
    B = {k: np.array(v, np.float32) for k, v in acc.items()}
    B['hass'] = np.ones(len(B['ys']), bool); B['meta_ba'] = np.array([ba] * len(B['ys']))
    return B


def stress():
    """Dev gate days (seed 7, 30 per grid) with each grid's reported solar moved by -1, 0, +1 h."""
    Bd = dict(np.load(f'{DATA}/banks/real_dev.npz')); rows = gate_rows(Bd)
    out = {}
    for shift in (-1, 0, 1):
        parts = []
        for ba in BAS:
            ix = np.where(rows & (Bd['meta_ba'] == ba) & Bd['hass'])[0]
            for y, calib in ((2019, 'dev2021'), (2021, 'dev2019')):      # cross-year fits, as realbank.py dev
                days = [pd.Timestamp(d) for d in Bd['meta_day'][ix] if int(d[:4]) == y]
                if days: parts.append(build_rows(ba, days, calib, shift))
        B = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        r, _ = DC.solar_offset(np.nan_to_num(B['xs'].astype(np.float64)), ref_rows(B, 'ref'))
        e0 = solar_err(B, predict(B, False, 's')); e1 = solar_err(B, predict(B, 'ref', 's'))
        out[shift] = dict(rows=len(B['ys']), offset_read={str(k): int((r == k).sum()) for k in (-1, 0, 1)},
                          v10=e0['model'], r9=e1['model'], persistence=e0['persistence'], physics=e0['physics'])
        print(f'shift {shift:+d} h: {len(B["ys"])} days | offset read -1/0/+1 {[int((r == k).sum()) for k in (-1, 0, 1)]} | '
              f'solar v1.0 {e0["model"]:.1f}% r9 {e1["model"]:.1f}% persistence {e0["persistence"]:.1f}% physics {e0["physics"]:.1f}%')
    return out


def y2022():
    """Diagnostic on a training-input year (its solar output was never a training sample and chose nothing)."""
    out = {}
    for ba in ('CISO', 'PJM'):
        days = days_of_year(ba, 2022)
        B = build_rows(ba, days, 'dev', 0)
        r, _ = DC.solar_offset(np.nan_to_num(B['xs'].astype(np.float64)), ref_rows(B, 'ref'))
        e0 = solar_err(B, predict(B, False, 's')); e1 = solar_err(B, predict(B, 'ref', 's'))
        out[ba] = dict(days=len(B['ys']), offset_read={str(k): int((r == k).sum()) for k in (-1, 0, 1)},
                       v10=e0['model'], r9=e1['model'], persistence=e0['persistence'], physics=e0['physics'])
        print(f'{ba} 2022: {len(B["ys"])} days | offset read -1/0/+1 {[int((r == k).sum()) for k in (-1, 0, 1)]} | '
              f'solar v1.0 {e0["model"]:.1f}% r9 {e1["model"]:.1f}% persistence {e0["persistence"]:.1f}% physics {e0["physics"]:.1f}%')
    return out


def val(name):
    from train import load_banks
    Bv = load_banks([name])
    res = {}
    for nm, al in (('v1.0', False), ('r9', 'ref')):
        res[nm] = E.sim_score(Bv, predict(Bv, al))
        print(f"{nm} on {name}: objective {res[nm]['objective']:.4f} | skill d {res[nm]['d']['skill']:.4f} "
              f"w {res[nm]['w']['skill']:.4f} s {res[nm]['s']['skill']:.4f} | solar err {res[nm]['s']['err']:.2f}% "
              f"(physics {res[nm]['s']['prior_err']:.2f}%)")
    if 'meta_soff' in Bv:
        r, _ = DC.solar_offset(np.nan_to_num(Bv['xs'].astype(np.float64)))
        t = Bv['meta_soff']
        conf = {f'{a:+d}': [int(((t == a) & (r == b)).sum()) for b in (-1, 0, 1)] for a in (-1, 0, 1)}
        print('true offset -> read -1/0/+1:', conf)
        res['confusion'] = conf
        for a in (-1, 0, 1):
            s = t == a
            for nm, al in (('v1.0', False), ('r9', 'ref')):
                Bs = {k: v[s] for k, v in Bv.items() if len(v) == len(t)}
                res.setdefault('by_offset', {}).setdefault(str(a), {})[nm] = E.errors(Bs, predict(Bs, al, 's'), 's')['err']
        print('solar error by true offset:', res['by_offset'])
    return res


def add(name, rung='r9b'):
    v = val(name); d, rels = dev(); rel = rels[rung]; d['r9'] = d[rung]
    s_ok = v['r9']['objective'] > v['v1.0']['objective'] + 0.01; d_ok = rel <= 1.02
    why = (f"objective {v['r9']['objective']:.4f} vs {v['v1.0']['objective']:.4f} on {name}" + ('' if s_ok else ' (needs +0.01)')
           + f"; dev errors {100 * (rel - 1):+.1f}% vs incumbent" + ('' if d_ok else ' (gate: <= +2%)'))
    rec = dict(ckpt=rung, date='2026-11-15', weights='demandcast_weights.npz (r8, unchanged)',
               note='r8 network + solar reporting offset read at issue from the 168 h history (demandcast.solar_offset)',
               val_set=name, val=v['r9'], objective=v['r9']['objective'], dev=d['r9'],
               incumbent_rescored=dict(ckpt='r8_wide', val=v['v1.0'], objective=v['v1.0']['objective'], dev=d['v1.0']),
               confusion=v.get('confusion'), by_offset=v.get('by_offset'), dev_rel=float(rel), vs='r8_wide',
               accepted=bool(s_ok and d_ok), why=why)
    log = json.load(open(LOG)); log.append(rec); jdump(log, LOG)
    print('ACCEPTED' if rec['accepted'] else 'rejected', '-', why)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'dev': dev()
    elif cmd == 'stress': stress()
    elif cmd == 'y2022': y2022()
    elif cmd == 'val': val(sys.argv[2])
    elif cmd == 'add': add(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'r9b')
