"""Gauged/nowcast mode on REAL basins: the network also reads the basin's own observed flow up to `lead` days before
each target day (GAUGED protocol). Compared with persistence at the same lead and with NWM.
usage: python3 evalnow.py <ckpt> dev|exam"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import torch
torch.set_num_threads(1)
import waterdata as wd
from waterdata import metrics as M
import bankdata as bd
import evalreal as ER

LEADS = (1, 3, 7)


def run(net, b, s0, s1, obs, lead):
    F, doy, st, idx = real_inputs(b, s0, (pd.Timestamp(s1) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
    pm = math.exp(st[11])
    full = obs.reindex(idx).values / pm                     # obs over warm-up + target (+1)
    n_target = len(pd.date_range(s0, s1))
    W = bd.WARM
    X, blocks, a = [], [], 0
    while a < n_target:
        b_ = min(a + 365, n_target); blocks.append((a, b_)); a = b_
    L = W + 365
    for a, b_ in blocks:
        Fw = F[a:a + W + (b_ - a) + 1]; qw = full[a:a + W + (b_ - a)]
        if len(Fw) < L + 1:
            Fw = np.concatenate([Fw, np.repeat(Fw[-1:], L + 1 - len(Fw), 0)])
            qw = np.concatenate([qw, np.full(L - len(qw), np.nan)])
        x = bd.inputs(Fw, doy[a] - 1, False)
        X.append(np.concatenate([x, bd.nowcast_channels(qw.astype(np.float32), lead)], 1))
    with torch.no_grad():
        m, q = net(torch.from_numpy(np.stack(X)), torch.from_numpy(np.repeat(st[None], len(blocks), 0)))
    mean = np.concatenate([m.numpy()[k, W:W + b_ - a] for k, (a, b_) in enumerate(blocks)]) * pm
    return pd.date_range(s0, s1), mean


def main(name, which):
    net, cfg = ER.load_net(name)
    nwm = pd.read_parquet(f'{WATER_DATA}/data/proc/nwm/daily_local.parquet')
    if which == 'dev':
        jobs, ids = [('dev', DEVWIN, 'caravan')], wd.basins('camels', split='dev')
    else:
        jobs, ids = [('val', VAL, 'caravan'), ('recent', REC, 'nwis')], wd.basins('camels', split='exam')
    rows = []
    t0 = time.time()
    for n, b in enumerate(ids):
        for per, (s0, s1), src in jobs:
            try:
                obs = wd.streamflow(b, src).Q_mm
            except FileNotFoundError:
                continue
            o = obs.loc[s0:s1]
            if o.notna().sum() < 365:
                continue
            for lead in LEADS:
                idx, mean = run(net, b, s0, s1, obs, lead)
                pers = obs.shift(lead, freq='D').reindex(idx)
                r1 = M.summary(o, pd.Series(mean, index=idx)); r2 = M.summary(o, pers)
                rows.append({'basin': b, 'period': per, 'lead': lead, 'nse': r1['nse'], 'kge': r1['kge'],
                             'nse_persist': r2['nse'], 'kge_persist': r2['kge']})
        if n % 50 == 0:
            print(n, f'{time.time()-t0:.0f}s', flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(f'{ROOT}/results/{name}_now_{which}.parquet')
    print(d.groupby(['period', 'lead'])[['nse', 'nse_persist', 'kge', 'kge_persist']].median().round(3))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
