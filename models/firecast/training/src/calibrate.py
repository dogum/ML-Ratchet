"""Dev calibration of FireCast probabilities (2 numbers + a threshold, dev only):
    p = sigmoid(logit / T + b), T >= 1 (widen-only: may only flatten the network's confidence), b free (level shift
    for the sim-to-real growth bias); then the F1 threshold that maximises mean dev F1 (B's protocol allows a
    threshold fixed on dev). Leave-one-fire-out estimates are reported too.
Usage: python3 calibrate.py <rung> [F|L]  -> out/cal_<rung>_<mode>.json"""
from common import *  # noqa
import pickle
import evaluate as EV
from scipy.optimize import minimize


def nll(params, L, Y, W):
    T, b = max(params[0], 1.0), params[1]
    z = L / T + b
    return float(np.sum(W * (np.logaddexp(0, z) - Y * z)) / np.sum(W))


def fit_Tb(L, Y, W):
    best = None
    for T0 in (1.0, 1.5, 2.5):
        for b0 in (-2.0, 0.0, 2.0):
            r = minimize(nll, [T0, b0], args=(L, Y, W), method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-6))
            if best is None or r.fun < best.fun:
                best = r
    return max(best.x[0], 1.0), best.x[1]


def main(rung, mode='F'):
    net = EV.load_net(rung)
    land = mode == 'L'
    rows, store = EV.dev_gate(net, land, keep_logits=True)
    fires = np.array([r['fire_id'] for r in rows])
    # per fire-day equal weight (each day's cells weighted 1/n_cells, matching the per-day mean metrics)
    Ls, Ys, Ws, F = [], [], [], []
    rng = np.random.default_rng(0)
    for (lg, y), fid in zip(store, fires):
        n = len(lg)
        k = rng.choice(n, size=min(n, 4000), replace=False)
        Ls.append(lg[k].astype(np.float32))
        Ys.append(y[k].astype(np.float32))
        Ws.append(np.full(len(k), 1.0 / len(k), np.float32))
        F.append(np.full(len(k), fid))
    L, Y, W, F = map(np.concatenate, (Ls, Ys, Ws, F))
    T, b = fit_Tb(L, Y, W)
    # threshold: maximise mean F1 over dev days at calibrated probabilities
    thrs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]

    def f1s(Tc, bc, thr):
        out = []
        for lg, y in store:
            p = EV.sig(lg.astype(np.float32) / Tc + bc) >= thr
            tp = (p & y).sum()
            d = p.sum() + y.sum()
            out.append(2 * tp / d if d else np.nan)
        return float(np.nanmean(out))
    f1_by_thr = {t: f1s(T, b, t) for t in thrs}
    thr = max(f1_by_thr, key=f1_by_thr.get)
    # leave-one-fire-out: refit T, b, thr without each fire, score that fire
    lofo = []
    ufires = sorted(set(fires))
    for fid in ufires:
        m = F != fid
        Tf, bf = fit_Tb(L[m], Y[m], W[m])
        idx = [i for i, f in enumerate(fires) if f != fid]
        best_t, best_v = 0.5, -1
        for t in thrs:
            v = []
            for i in idx:
                lg, y = store[i]
                p = EV.sig(lg.astype(np.float32) / Tf + bf) >= t
                d = p.sum() + y.sum()
                v.append(2 * (p & y).sum() / d if d else np.nan)
            v = np.nanmean(v)
            if v > best_v:
                best_t, best_v = t, v
        for i, f in enumerate(fires):
            if f != fid:
                continue
            lg, y = store[i]
            p = EV.sig(lg.astype(np.float32) / Tf + bf) >= best_t
            d = p.sum() + y.sum()
            lofo.append(2 * (p & y).sum() / d if d else np.nan)
    raw_f1 = f1s(1.0, 0.0, 0.5)
    out = dict(rung=rung, mode=mode, T=float(T), b=float(b), thr=float(thr), f1_by_thr=f1_by_thr,
               dev_f1_raw_at_0p5=raw_f1, dev_f1_cal_at_thr=f1_by_thr[thr], dev_f1_cal_at_0p5=f1_by_thr[0.5],
               dev_f1_lofo=float(np.nanmean(lofo)), nll_raw=nll([1.0, 0.0], L, Y, W), nll_cal=nll([T, b], L, Y, W),
               n_days=len(store), note='fit on DEV fire-days only (pooled candidate cells, 4000 per day, equal day weights)')
    # reliability (pooled cells, calibrated)
    P = EV.sig(L / T + b)
    bins = np.array([0, .02, .05, .1, .2, .3, .5, .7, 1.0001])
    rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (P >= lo) & (P < hi)
        if m.sum():
            rel.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()), p_mean=float(np.average(P[m], weights=W[m])),
                            obs=float(np.average(Y[m], weights=W[m]))))
    out['reliability_dev'] = rel
    jdump(out, f'{OUT}/cal_{rung}_{mode}.json')
    print(json.dumps({k: v for k, v in out.items() if k != 'reliability_dev'}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'F')
