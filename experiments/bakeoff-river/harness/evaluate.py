"""Score an arm on the frozen sim val set and on the dev gate (the two things a rung decision may look at).

usage: python3 evaluate.py <arm_dir> [--sets simval,dev] [--protocols ungauged,gauged] [--nmax N] [--tag TAG]
                           [--out DIR] [--save-preds]
Writes <out>/<tag>_summary.json and per-unit parquet tables (default out = <arm_dir>/work/eval, tag = released rung).
Gauged protocol is scored at leads 1, 3 and 7 days. Dev references (U-dev, U-k12, GR4J, NWM, climatology,
persistence) are scored with the same functions (results/dev_refs.parquet, built once by `python3 evaluate.py --refs`).
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

DEV_REFS = os.path.join(C.RESULTS, 'dev_refs.parquet')


def jobs_for(protocols):
    J = []
    if 'ungauged' in protocols:
        J.append(('ungauged', None))
    if 'gauged' in protocols:
        J += [('gauged', L) for L in C.LEADS]
    return J


def tagname(protocol, lead):
    return protocol if lead is None else f'{protocol}_L{lead}'


def eval_simval(arm, protocol, lead, nmax=None, chunk=50, log=print):
    B = bake.load_bank('val', ['F', 'qo', 'attrs', 'st', 'lat', 'lon', 'start'])
    n = len(B['qo']) if nmax is None else min(nmax, len(B['qo']))
    rows, buf = [], []
    t0 = time.time()

    def flush(buf):
        outs = arm.run([strip_batch(b) for b, _, _ in buf])
        per = {}
        for (b, y, i), o in zip(buf, outs):
            m, Q, nc = S.check_output(o, b['n_target'])
            per.setdefault(i, []).append((y, m, Q, nc))
        for i, parts in per.items():
            y = np.concatenate([p[0] for p in parts]); m = np.concatenate([p[1] for p in parts])
            Q = np.concatenate([p[2] for p in parts]); nc = sum(p[3] for p in parts)
            r = S.simval_scores(y, m, Q)
            r.update({'run': i, 'n_cross': nc})
            rows.append(r)

    for i in range(n):
        for a in bake.SIMVAL_OFFSETS:
            b = bake.sim_window(B, i, a, 365, protocol, lead)
            buf.append((b, bake.sim_target(B, i, b), i))
        if len(buf) >= chunk:
            flush(buf); buf = []
            if len(rows) % 200 == 0:
                log(f'  simval {tagname(protocol, lead)} {len(rows)}/{n} {time.time() - t0:.0f}s')
    if buf:
        flush(buf)
    return pd.DataFrame(rows).set_index('run')


def eval_dev(arm, protocol, lead, chunk=4, basins=None, preds=None):
    rows = []
    ids = basins or bake.dev_basins()
    for k in range(0, len(ids), chunk):
        bs = [bake.dev_batch(b, protocol, lead) for b in ids[k:k + chunk]]
        outs = arm.run([strip_batch(b) for b in bs])
        for bid, b, o in zip(ids[k:k + chunk], bs, outs):
            m, Q, nc = S.check_output(o, b['n_target'])
            obs = bake._obs(bid, 'dev')
            r = S.score_real(obs, m, Q, n_cross=nc)
            r['basin'] = bid
            rows.append(r)
            if preds is not None:
                preds[bid] = np.concatenate([obs.values[:, None], m[:, None], Q], 1).astype(np.float32)
    return pd.DataFrame(rows).set_index('basin')


def build_dev_refs():
    """Reference rows on the dev gate (52 dev basins, WY2001-2014), all scored by score.score_real."""
    rows = []
    for bid in bake.dev_basins():
        obs = bake._obs(bid, 'dev')
        import waterdata as wd
        full = wd.streamflow(bid, 'caravan').Q_mm
        for name in ('udev', 'uk12', 'gr4j', 'nwm', 'clim'):
            s = R.reference(name, bid, 'dev')
            if s is None:
                continue
            r = S.score_real(obs, s.values)
            r.update({'basin': bid, 'method': name, 'lead': 0})
            rows.append(r)
        for L in C.LEADS:
            s = R.reference('persistence', bid, 'dev', obs_full=full, lead=L)
            r = S.score_real(obs, s.values)
            r.update({'basin': bid, 'method': 'persistence', 'lead': L})
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_parquet(DEV_REFS)
    return df


def dev_refs():
    if not os.path.exists(DEV_REFS):
        return build_dev_refs()
    return pd.read_parquet(DEV_REFS)


def dev_comparisons(dfa, protocol, lead, classes):
    refs = dev_refs()
    out = {}
    names = ['udev', 'uk12', 'gr4j', 'nwm', 'clim'] + (['persistence'] if protocol == 'gauged' else [])
    for name in names:
        g = refs[(refs.method == name) & (refs.lead == (lead if name == 'persistence' else 0))].set_index('basin')
        out[name] = {'ref_summary': S.summarize(g), 'paired': S.compare(dfa, g, classes=classes)}
    fc = os.path.join(C.RESULTS, 'dev', f'flowcast_v1_{tagname(protocol, lead)}_dev.parquet')
    if os.path.exists(fc):
        g = pd.read_parquet(fc)
        out['flowcast_v1'] = {'ref_summary': S.summarize(g), 'paired': S.compare(dfa, g, classes=classes)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm', nargs='?')
    ap.add_argument('--sets', default='simval,dev')
    ap.add_argument('--protocols', default=None)
    ap.add_argument('--nmax', type=int, default=None, help='sim val runs (default all 800; use all for decisions)')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--save-preds', action='store_true', help='also save daily dev predictions: '
                    '<tag>_dev_<job>_preds.npz, per basin (days, 7) = obs, mean, q05..q95 (for conformal fitting)')
    ap.add_argument('--refs', action='store_true', help='(re)build results/dev_refs.parquet and exit')
    ap.add_argument('--publish-dev', action='store_true', help=argparse.SUPPRESS)  # harness: copy dev tables
    a = ap.parse_args()
    if a.refs:
        df = build_dev_refs()
        print(df.groupby(['method', 'lead'])[['nse', 'kge']].median().round(3))
        return
    arm = Arm(a.arm)
    meta = arm.meta
    protocols = a.protocols.split(',') if a.protocols else meta['protocols']
    tag = a.tag or meta.get('released', 'eval')
    out = a.out or os.path.join(arm.dir, 'work', 'eval')
    os.makedirs(out, exist_ok=True)
    classes = {b: bake.basin_classes(b) for b in bake.dev_basins()}
    summ = {'arm_id': meta['arm_id'], 'tag': tag, 'when': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'harness': 'river-bakeoff harness v1', 'simval': {}, 'dev': {}}
    for protocol, lead in jobs_for(protocols):
        tn = tagname(protocol, lead)
        if 'simval' in a.sets:
            t0 = time.time(); c0 = arm.seconds
            df = eval_simval(arm, protocol, lead, a.nmax)
            df.to_parquet(os.path.join(out, f'{tag}_simval_{tn}.parquet'))
            s = S.summarize(df, ('nse', 'kge', 'fhv', 'cov50', 'cov90', 'crps_rel', 'width90_rel'))
            s['predict_cpu_s'] = round(arm.seconds - c0, 1); s['wall_s'] = round(time.time() - t0, 1)
            summ['simval'][tn] = s
            print(f'{meta["arm_id"]} simval {tn}: NSE {s["nse"]:.3f} KGE {s["kge"]:.3f} cov90 {s.get("cov90")} '
                  f'n {s["n"]} ({s["wall_s"]:.0f}s)', flush=True)
        if 'dev' in a.sets:
            t0 = time.time(); c0 = arm.seconds
            preds = {} if a.save_preds else None
            df = eval_dev(arm, protocol, lead, preds=preds)
            if preds:
                np.savez_compressed(os.path.join(out, f'{tag}_dev_{tn}_preds.npz'), **preds)
            df.to_parquet(os.path.join(out, f'{tag}_dev_{tn}.parquet'))
            if a.publish_dev:
                os.makedirs(os.path.join(C.RESULTS, 'dev'), exist_ok=True)
                df.to_parquet(os.path.join(C.RESULTS, 'dev', f'{meta["arm_id"]}_{tn}_dev.parquet'))
            s = S.summarize(df)
            s['by_class_nse'] = S.by_class(df, classes)
            s['predict_cpu_s'] = round(arm.seconds - c0, 1); s['wall_s'] = round(time.time() - t0, 1)
            s['vs'] = dev_comparisons(df, protocol, lead, classes)
            summ['dev'][tn] = s
            print(f'{meta["arm_id"]} dev {tn}: NSE {s["nse"]:.3f} KGE {s["kge"]:.3f} FHV {s["fhv"]:.1f} '
                  f'cov50/90 {s.get("cov50")}/{s.get("cov90")} CRPSrel {s.get("crps_rel")} ({s["wall_s"]:.0f}s)', flush=True)
    fn = os.path.join(out, f'{tag}_summary.json')
    with open(fn, 'w') as f:
        json.dump(summ, f, indent=1, default=float)
    print('wrote', fn)


if __name__ == '__main__':
    main()
