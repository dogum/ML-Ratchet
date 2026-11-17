"""Error-bar calibration on DEV only (widen-only): one factor per output and mode, plus class temperatures >= 1.

Regression widen factor s_j = max(1, q68(|err| / sigma)) using
  - floor area and year built: real dev (BDG2 dev for hourly/monthly/annual; + RECS/CBECS dev for annual);
  - every other output (UA, tau, balance points, sensitivities, base share, schedule): the simulated twins of the
    BDG2 dev buildings (feats/real/devtwin_*), because no real truth exists for them.
Class temperature (>= 1) minimises the NLL on real dev labels (use: BDG2 dev + CBECS dev; heat: RECS + CBECS dev).
Writes the factors into the incumbent npz files (models/final/<mode>.npz) and models/final/calibration.json.
"""
import json
import os
import shutil

import numpy as np

import common as C
import evaluate as E
from detective import Net

J = {n: j for j, n in enumerate(C.REG)}
NU, NH = len(C.USES), len(C.HEAT)


def q68(z):
    z = np.abs(np.asarray(z, float))
    z = z[np.isfinite(z)]
    return float(max(1.0, np.quantile(z, 0.6827))) if len(z) >= 10 else 1.0


def nll_temp(lg, y, Ts=np.arange(1.0, 10.01, 0.05)):
    best = (np.inf, 1.0)
    for T in Ts:
        z = lg / T
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        v = -np.mean(lp[np.arange(len(y)), y])
        best = min(best, (v, T))
    return float(best[1])


def main(src='incumbent', dst='final', twin_tag='devtwin_v2'):
    sd = os.path.join(C.ROOT, 'models', src)
    od = os.path.join(C.ROOT, 'models', dst)
    os.makedirs(od, exist_ok=True)
    rep = {}
    mats, labs = E.load('dev', 'bdg2')
    rmats, rlabs = E.load('dev', 'recs')
    cmats, clabs = E.load('dev', 'cbecs')
    for mode in ('hourly', 'monthly', 'annual'):
        fn = os.path.join(sd, f'{mode}.npz')
        if not os.path.exists(fn):
            continue
        net = Net(fn)
        net.widen = np.ones(len(C.REG))
        tw = np.load(os.path.join(C.ROOT, 'feats', 'real', f'{twin_tag}_{mode}.npz'))
        mu, sg, lg = net.raw(tw['X'])
        widen = np.ones(len(C.REG))
        Ytw = C.clean_labels(tw['Y'])
        for j, n in enumerate(C.REG):
            m = np.isfinite(Ytw[:, j])
            widen[j] = q68((mu[m, j] - Ytw[m, j]) / sg[m, j])
        # real dev for area and year
        mu, sg, lg = E.run_net(net, mats[mode])
        A = np.array([l['A'] for l in labs], float)
        za = list((mu[:, J['logA']] - np.log10(A)) / sg[:, J['logA']])
        yb = np.array([l['year'] for l in labs], float)
        zy = list((mu[:, J['year']] - yb) / sg[:, J['year']])
        use_lg, use_y = [lg[:, :NU]], [np.array([C.USES.index(l['use']) for l in labs])]
        heat_lg, heat_y = [], []
        if mode == 'annual':
            for M, L in ((rmats, rlabs), (cmats, clabs)):
                mu2, sg2, lg2 = E.run_net(net, M['annual'])
                A2 = np.array([l['A'] for l in L], float)
                za += list((mu2[:, J['logA']] - np.log10(A2)) / sg2[:, J['logA']])
                lo = np.array([l['year_lo'] for l in L], float); hi = np.array([l['year_hi'] for l in L], float)
                mid = np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2, np.nan)
                zy += list((mu2[:, J['year']] - mid) / np.sqrt(sg2[:, J['year']] ** 2 + 25))
                h = np.array([C.HEAT.index(l['heat']) if l.get('heat') in C.HEAT else -1 for l in L])
                heat_lg.append(lg2[h >= 0, NU:NU + NH]); heat_y.append(h[h >= 0])
                if L is clabs:
                    use_lg.append(lg2[:, :NU]); use_y.append(np.array([C.USES.index(l['use']) for l in L]))
        widen[J['logA']] = q68(za)
        widen[J['year']] = q68(zy)
        temp = np.ones(3)
        ul, uy = np.concatenate(use_lg), np.concatenate(use_y)
        ok = np.isfinite(ul).all(1)
        temp[0] = nll_temp(ul[ok], uy[ok])
        if heat_lg:
            hl, hy = np.concatenate(heat_lg), np.concatenate(heat_y)
            temp[1] = nll_temp(hl, hy)
        z = dict(np.load(fn))
        z['widen'], z['temp'] = widen, temp
        np.savez(os.path.join(od, f'{mode}.npz'), **z)
        rep[mode] = dict(widen=dict(zip(C.REG, widen.round(3).tolist())), temperature=temp.round(2).tolist())
        print(mode, rep[mode])
    json.dump(rep, open(os.path.join(od, 'calibration.json'), 'w'), indent=1)
    return rep


if __name__ == '__main__':
    main()
