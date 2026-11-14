"""Ratchet: score a rung on the frozen simulated set and the real dev gate; log to ratchet.json.

    python3 ratchet.py score <rung> <frozen_bank> [--decide "reason"]   # rung = ckpt name, or 'rolling' (classical)
    python3 ratchet.py rescore <frozen_bank>                              # re-score every logged rung on a new set

Frozen score = capture = net revenue / perfect-foresight net revenue, summed over the 3 battery durations (1, 2, 4 h)
of each path-year and averaged over the path-years of the frozen simulated set (each path-year counts the same; a
dollar-weighted sum would be dominated by a few scarcity years).  Dev capture is averaged the same way over the 10
real dev node-years.  Rule: keep a rung if it beats the incumbent's frozen
score by > 0.01 AND passes the dev gate: its aggregate capture on the real dev node-years (ERCOT 2019 and 2021 at
4 hubs, CAISO Aug-Dec 2023 at NP15/SP15; all 3 durations) is not more than 0.01 below the incumbent's.
"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd
import _paths  # noqa: E402  (this folder, GridSim and GridData on sys.path; outputs to WORK)
from evaluate import evaluate_path
from model import PolicyNet
from realdata import node_frame, fill

ROOT = _paths.WORK   # was the sandbox workspace grid/batterypilot
RJ = f'{ROOT}/ratchet.json'
DEV_SETS = [('ERCOT', n, y) for y in (2019, 2021) for n in ('HB_HOUSTON', 'HB_NORTH', 'HB_SOUTH', 'HB_WEST')] + \
           [('CAISO', n, 2023) for n in ('TH_NP15_GEN-APND', 'TH_SP15_GEN-APND')]


def load_frozen(bank):
    out = []
    for fn in sorted(glob.glob(f'{ROOT}/data/banks/{bank}/*.npz')):
        z = np.load(fn, allow_pickle=True)
        out.append({k: z[k] for k in z.files})
    return out


def dev_paths():
    cache = f'{ROOT}/data/dev_paths.pkl'
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    out = {}
    for m, n, y in DEV_SETS:
        s, e = (f'{y}-01-01', f'{y + 1}-01-01') if m == 'ERCOT' else ('2023-08-03', '2024-01-01')
        f, miss = fill(node_frame(m, n, s, e))
        f = f.dropna(subset=['rt', 'da'])
        out[(m, n, y)] = f
    pd.to_pickle(out, cache)
    return out


def path_dict(f):
    return {c: f[c].values for c in ['rt', 'da', 'df', 'T', 'hour', 'day', 'dow', 'doy']}


def score_strategy(paths, strategy, net=None, durations=(1, 2, 4)):
    rows = []
    for i, p in enumerate(paths):
        r, _ = evaluate_path(p, durations, (strategy,), net=net)
        r['path'] = i
        rows.append(r)
    return pd.concat(rows)


def baseline_table(paths, key, durations=(1, 2, 4)):
    fn = f'{ROOT}/data/baselines_{key}.parquet'
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    rows = []
    for i, p in enumerate(paths):
        r, _ = evaluate_path(p, durations, ('pf', 'rolling', 'spread'))
        r['path'] = i; rows.append(r)
    t = pd.concat(rows); t.to_parquet(fn)
    return t


def capture(t, base):
    m = t.merge(base[base.strategy == 'pf'][['path', 'duration', 'net']].rename(columns={'net': 'pf'}),
                on=['path', 'duration'])
    per = m.groupby('path').apply(lambda g: g.net.sum() / g.pf.sum(), include_groups=False)
    return float(per.mean()), m


def score_rung(rung, frozen):
    fz = load_frozen(frozen)
    base = baseline_table(fz, frozen)
    dev = dev_paths()
    dps = [path_dict(f) for f in dev.values()]
    dbase = baseline_table(dps, 'dev')
    if rung in ('rolling', 'spread'):
        tf = base[base.strategy == rung]; td = dbase[dbase.strategy == rung]
    else:
        net = PolicyNet(f'{ROOT}/ckpt/{rung}.npz')
        tf = score_strategy(fz, 'bp', net); td = score_strategy(dps, 'bp', net)
    cf, _ = capture(tf, base)
    cd, md = capture(td, dbase)
    agg = float(md.net.sum() / md.pf.sum())
    by = {}
    keys = list(dev.keys())
    for (m_, n_, y_), g in md.assign(key=[keys[i] for i in md.path]).groupby('key'):
        by[f'{m_}:{n_}:{y_}'] = round(float(g.net.sum() / g.pf.sum()), 4)
    byd = {int(d): round(float(g.net.sum() / g.pf.sum()), 4) for d, g in md.groupby('duration')}
    return dict(rung=rung, frozen=frozen, frozen_capture=round(cf, 4), dev_capture=round(cd, 4), dev_capture_dollar_weighted=round(agg, 4), dev_by_set=by,
                dev_by_duration=byd,
                frozen_cycles=round(float(tf.cycles.mean()), 1), dev_cycles=round(float(td.cycles.mean()), 1))


def log(entry):
    R = json.load(open(RJ)) if os.path.exists(RJ) else {'rule': __doc__, 'rungs': []}
    R['rungs'].append(entry)
    json.dump(R, open(RJ, 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('rung'); ap.add_argument('frozen', nargs='?')
    ap.add_argument('--note', default='')
    a = ap.parse_args()
    if a.cmd == 'score':
        t0 = time.time()
        r = score_rung(a.rung, a.frozen)
        r['note'] = a.note; r['secs'] = round(time.time() - t0)
        print(json.dumps(r, indent=1))
        os.makedirs(f'{ROOT}/data/scores', exist_ok=True)
        json.dump(r, open(f'{ROOT}/data/scores/{a.rung}__{a.frozen}.json', 'w'), indent=1)
