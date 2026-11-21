"""Run a FlowCast checkpoint on REAL basins (forcing + attributes only: ungauged) and score it.

usage: python3 evalreal.py <ckpt name> dev            -> dev gate (52 dev basins, WY2000-2014, Caravan Q)
       python3 evalreal.py <ckpt name> exam [factors]  -> exam (619 basins; val 1989-10..1999-09 Caravan Q;
                                                          recent WY2016-2023 NWIS Q). ONLY for the final report.
Writes results/<ckpt>_<set>.parquet (per basin x period) and preds/<ckpt>_<set>_<period>.npz.
"""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import torch
torch.set_num_threads(1)
import waterdata as wd
from waterdata import metrics as M
from model import FlowNet, QS
import bankdata as bd
from train import build_net


def load_net(name):
    ck = torch.load(f'{ROOT}/ckpt/{name}.pt', weights_only=False)
    cfg = ck['cfg']
    net = build_net(cfg)
    net.load_state_dict(ck['state'])
    net.eval()
    return net, cfg


def predict_series(net, nextday, F, doy, st, n_target):
    """F (T, 8) covering 365 warm-up days + n_target days + 1 extra day. Blocks of 365 target days, each with its
    own 365-day warm-up (as in training). Returns mean (n_target,), q (n_target, 5) in units of p_mean."""
    W = bd.WARM
    blocks = []
    a = 0
    while a < n_target:
        b = min(a + 365, n_target)
        blocks.append((a, b))
        a = b
    L = W + 365
    X = []
    for a, b in blocks:
        Fw = F[a:a + W + (b - a) + 1]
        if len(Fw) < L + 1:                                  # pad the last block at its end (causal: no effect)
            Fw = np.concatenate([Fw, np.repeat(Fw[-1:], L + 1 - len(Fw), 0)])
        X.append(bd.inputs(Fw, doy[a] - 1, nextday))
    X = torch.from_numpy(np.stack(X)); S = torch.from_numpy(np.repeat(st[None], len(blocks), 0))
    with torch.no_grad():
        m, q = net(X, S)
    m = m.numpy(); q = q.numpy()
    mean = np.concatenate([m[k, W:W + b - a] for k, (a, b) in enumerate(blocks)])
    qq = np.concatenate([q[k, W:W + b - a] for k, (a, b) in enumerate(blocks)])
    return mean, qq


def widen(q, k):
    """Widen-only interval calibration: k = (k50, k90) scale distances from the median."""
    if k is None:
        return q
    q = q.copy(); med = q[:, 2:3]
    k50, k90 = k
    q[:, 1] = med[:, 0] - k50 * (med[:, 0] - q[:, 1]); q[:, 3] = med[:, 0] + k50 * (q[:, 3] - med[:, 0])
    q[:, 0] = med[:, 0] - k90 * (med[:, 0] - q[:, 0]); q[:, 4] = med[:, 0] + k90 * (q[:, 4] - med[:, 0])
    return np.maximum(q, 0)


def run_basin(net, nextday, bid, start, end):
    F, doy, st, idx = real_inputs(bid, start, (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
    n_target = len(pd.date_range(start, end))
    mean, q = predict_series(net, nextday, F, doy, st, n_target)
    pm = math.exp(st[11])
    return pd.date_range(start, end), mean * pm, q * pm


def score(obs, idx, mean, q, k=None):
    o = obs.reindex(idx)
    s = pd.Series(mean, index=idx)
    r = M.summary(o, s)
    qq = widen(q, k)
    v = o.notna().values
    ov = o.values[v]
    r['cov90'] = float(np.mean((ov >= qq[v, 0]) & (ov <= qq[v, 4]))) if v.any() else np.nan
    r['cov50'] = float(np.mean((ov >= qq[v, 1]) & (ov <= qq[v, 3]))) if v.any() else np.nan
    r['width90_rel'] = float(np.mean(qq[v, 4] - qq[v, 0]) / max(np.mean(ov), 1e-6)) if v.any() else np.nan
    return r


def main(name, which, k=None, save_preds=True):
    net, cfg = load_net(name)
    nextday = cfg.get('nextday', False)
    os.makedirs(f'{ROOT}/results', exist_ok=True); os.makedirs(f'{ROOT}/preds', exist_ok=True)
    rows = []
    if which == 'dev':
        jobs = [('dev', DEVWIN, 'caravan')]
        ids = wd.basins('camels', split='dev')
    else:
        jobs = [('val', VAL, 'caravan'), ('recent', REC, 'nwis')]
        ids = wd.basins('camels', split='exam')
    store = {p: {} for p, _, _ in jobs}
    t0 = time.time()
    for n, b in enumerate(ids):
        for per, (s0, s1), src in jobs:
            try:
                obs = wd.streamflow(b, src).Q_mm
            except FileNotFoundError:
                obs = pd.Series(dtype=float)
            o = obs.loc[s0:s1]
            if o.notna().sum() < 365:
                continue
            idx, mean, q = run_basin(net, nextday, b, s0, s1)
            r = score(obs, idx, mean, q, k)
            r.update({'basin': b, 'period': per})
            rows.append(r)
            if save_preds:
                store[per][b] = np.concatenate([mean[:, None], q], 1).astype(np.float32)
        if n % 50 == 0:
            print(n, b, f'{time.time()-t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    tag = f'{name}_{which}' + ('' if k is None else '_cal')
    df.to_parquet(f'{ROOT}/results/{tag}.parquet')
    if save_preds:
        for per in store:
            if store[per]:
                np.savez_compressed(f'{ROOT}/preds/{tag}_{per}.npz', **store[per])
    for per, g in df.groupby('period'):
        print(per, len(g), 'NSE', round(g.nse.median(), 3), 'KGE', round(g.kge.median(), 3), 'FHV', round(g.fhv.median(), 1),
              'peak', round(g.peak_abs_err_pct_median.median(), 1), 'cov90', round(g.cov90.median(), 3),
              'cov50', round(g.cov50.median(), 3), 'NSE<0', round((g.nse < 0).mean(), 3), flush=True)
    return df


if __name__ == '__main__':
    k = None
    if len(sys.argv) > 3:
        k = tuple(float(x) for x in sys.argv[3].split(','))
    main(sys.argv[1], sys.argv[2], k)
