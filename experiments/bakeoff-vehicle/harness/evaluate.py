"""Score an arm on the frozen sim val set and on the dev gate (the two things a rung decision may look at).

usage: python3 evaluate.py <arm_dir> [--track table|trace] [--sets simval,dev] [--nmax N] [--tag TAG] [--out DIR]
       python3 evaluate.py --refs          (harness only: dev reference rows -> results/dev/)
Writes <out>/<tag>_<track>_summary.json, <tag>_<track>_dev_preds.parquet (per dev record: pred, q05..q95, MJ for the
whole trip; the conformal layer fits on these) and <tag>_<track>_simval_preds.parquet (per scored val trip).
Default out = <arm_dir>/work/eval, tag = the arm's released rung, track = the arm's first track.
Sim val: the 3,985 scored trips of the frozen val bank with the frozen tiers (EnergyNet's protocol): MdAPE, median
|log error|, skill (1 - median |log error| / the dumb baseline's), bias, cov50 / cov90, crps_log, per architecture.
Dev gate: VED 16 cars, EPA dev families, Mersin dev week, ICCT dev truck, 5 ZTBus legs: every set's MdAPE, bias,
coverage, crps_log, car totals; D16 (mean of the 16 EnergyNet D2 terms available here); paired bootstrap against
EnergyNet v2.1 and the references on every set.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import bake  # noqa: E402
import score as S  # noqa: E402
import refs as R  # noqa: E402
from armlib import Arm, strip_batch  # noqa: E402

DEV_DIR = os.path.join(C.RESULTS, 'dev')
ANCHOR_ARM = 'enet_v21'


def run_simval(arm, track, nmax=None, log=print):
    ys, P, Q = [], [], []
    t0 = time.time()
    for b, y in bake.simval_batches(track, nmax=nmax):
        o = arm.run([strip_batch(b)])[0]
        p, q, nc, _ = S.check_output(o, b['n'])
        ys.append(y); P.append(p); Q.append(q)
    y, p, q = np.concatenate(ys), np.concatenate(P), np.concatenate(Q)
    meta = bake.simval_meta()
    if nmax:
        meta = meta.iloc[:len(y)]
    s = S.simval_summary(y, p, q, meta, bake.dumb_rates())
    s['wall_s'] = round(time.time() - t0, 1)
    df = meta.assign(y=y, pred=p, **{k: q[:, j] for j, k in enumerate(C.QKEYS)})
    return s, df


def run_dev(arm, track):
    """-> per-record frame (dev meta + pred, q05..q95, n_cross)."""
    Fr = bake.dev_frame()
    P = np.zeros(len(Fr)); Q = np.zeros((len(Fr), 5)); NC = 0; COMP = np.full((len(Fr), 5), np.nan)
    off = {name: int(Fr.index[Fr.recset == name][0]) for name in bake.SETS}
    for name, s, b in bake._real_batches('dev', track):
        o = arm.run([strip_batch(b)])[0]
        p, q, nc, comp = S.check_output(o, b['n'])
        i0 = off[name] + s.start
        P[i0:i0 + b['n']] = p; Q[i0:i0 + b['n']] = q; NC += nc
        if comp is not None:
            COMP[i0:i0 + b['n']] = comp
    Fr = Fr.assign(pred=P, **{k: Q[:, j] for j, k in enumerate(C.QKEYS)})
    for j, k in enumerate(['c_aero', 'c_rolling', 'c_brake', 'c_aux', 'c_losses']):
        Fr[k] = COMP[:, j]
    return Fr, NC


def ref_frames(Fr):
    """dev reference rows as per-record frames (point predictions)."""
    out = {}
    for name, p in R.reference_preds(Fr).items():
        d = Fr.assign(pred=p, **{k: p for k in C.QKEYS})
        out[name] = d[np.isfinite(d.pred)]
    return out


def dev_summary(Fr):
    U = S.real_units(Fr)
    s = S.real_summary(U)
    return U, s


def build_dev_refs():
    os.makedirs(DEV_DIR, exist_ok=True)
    Fr = bake.dev_frame()
    out = {}
    for name, d in ref_frames(Fr).items():
        d.to_parquet(os.path.join(DEV_DIR, f'ref_{name}.parquet'))
        U, s = dev_summary(d)
        out[name] = {'D16': S.d16(s), 'sets': s}
    json.dump(out, open(os.path.join(DEV_DIR, 'refs_summary.json'), 'w'), indent=1, default=float)
    print({k: {kk: round(vv['MdAPE'], 4) for kk, vv in v['sets'].items() if vv.get('n')} for k, v in out.items()})
    return out


def compare_dev(U, rng):
    """paired bootstrap of the arm against EnergyNet v2.1 (if its dev preds exist) and the reference rows, per set."""
    res = {}
    others = {}
    p = os.path.join(DEV_DIR, f'{ANCHOR_ARM}_table_dev_preds.parquet')
    if os.path.exists(p):
        others[ANCHOR_ARM] = S.real_units(pd.read_parquet(p))
    for name in R.REF_ROWS:
        q = os.path.join(DEV_DIR, f'ref_{name}.parquet')
        if os.path.exists(q):
            others[name] = S.real_units(pd.read_parquet(q))
    for k, u in U.items():
        if k.endswith('_cars') or k.startswith('VED_arch'):
            continue
        res[k] = {}
        for o, Uo in others.items():
            if k in Uo and len(Uo[k]):
                res[k][o] = S.compare(u, Uo[k], rng=rng, nboot=2000)
    return res


def evaluate(arm_dir, track=None, sets=('simval', 'dev'), nmax=None, tag=None, out=None, compare=True):
    arm = Arm(arm_dir)
    track = track or arm.meta['tracks'][0]
    if track not in arm.meta['tracks']:
        raise SystemExit(f'{arm.meta["arm_id"]} does not declare track {track}')
    tag = tag or arm.meta.get('released', 'r0')
    out = out or os.path.join(arm.dir, 'work', 'eval')
    os.makedirs(out, exist_ok=True)
    summ = {'arm_id': arm.meta['arm_id'], 'tag': tag, 'track': track, 'when': time.strftime('%Y-%m-%dT%H:%M:%S')}
    if 'simval' in sets:
        c0 = arm.seconds
        s, df = run_simval(arm, track, nmax)
        s['predict_cpu_s'] = round(arm.seconds - c0, 1)
        summ['simval'] = s
        df.to_parquet(os.path.join(out, f'{tag}_{track}_simval_preds.parquet'))
        print(f"simval: MdAPE {s['MdAPE']:.4f} skill {s.get('skill', float('nan')):.4f} bias {s['bias']:+.4f} "
              f"cov50/90 {s['cov50']:.2f}/{s['cov90']:.2f} crps_log {s['crps_log']:.4f}", flush=True)
    if 'dev' in sets:
        c0 = arm.seconds
        Fr, nc = run_dev(arm, track)
        Fr.to_parquet(os.path.join(out, f'{tag}_{track}_dev_preds.parquet'))
        U, s = dev_summary(Fr)
        summ['dev'] = {'D16': S.d16(s), 'n_cross': nc, 'predict_cpu_s': round(arm.seconds - c0, 1), 'sets': s}
        if compare:
            summ['dev']['paired'] = compare_dev(U, np.random.default_rng(C.SEED))
        print(f"dev: D16 {summ['dev']['D16']:.4f}  " + '  '.join(f"{k} {v['MdAPE']:.3f}" for k, v in s.items()
                                                               if k in S.D16), flush=True)
    p = os.path.join(out, f'{tag}_{track}_summary.json')
    json.dump(summ, open(p, 'w'), indent=1, default=float)
    print('wrote', p)
    return summ


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('arm', nargs='?')
    ap.add_argument('--track', default=None)
    ap.add_argument('--sets', default='simval,dev')
    ap.add_argument('--nmax', type=int, default=None)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--refs', action='store_true')
    ap.add_argument('--no-compare', action='store_true')
    a = ap.parse_args()
    if a.refs:
        build_dev_refs()
    else:
        evaluate(a.arm, a.track, tuple(a.sets.split(',')), a.nmax, a.tag, a.out, not a.no_compare)
