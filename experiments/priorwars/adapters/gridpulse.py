"""GridPulse (grid platform) adapter, GB events. Read-only use of models/gridpulse/training and shared griddata.

Rows = (event, N) with N = seconds of frequency seen after onset, N in {10, 60} (the exam's inertia rows).
Features = the model's own 86 inputs (dataset.features with ctx: window, is_gb, N, demand).
Targets = the model's own log_loss (MW) and log_E (inertia, MW s).
Dev = the 4 GB dev events (GB_DEV); exam = the 412 GB exam events (410 with NESO inertia; loss labels by tier).
G-sim context = simulated GB events from the release's banks (train_p1 + train_p1b) built exactly as train.py builds
an epoch (DS.make: phase, real quiet noise, onset; demand x N(1, 0.03), 25 % unknown), N fixed to 10 or 60 per row.

Paths (repository): the training code is models/gridpulse/training (its _paths.py puts shared/grid on sys.path). The
released rung r5 is read by evaluate.load('r5') from the model's training workspace (WORK/ckpt/r5.pt), and the frozen
simulated set from WORK/data/val_p2s.npz; banks from WORK too. None of these is in the repository: the release ships
the same network as numpy weights (gridpulse_weights.npz).
"""
import sys
sys.dont_write_bytecode = True
import pw_paths as _P
_P.use_training('gridpulse')
import functools
import numpy as np
import pandas as pd

NAME = 'gridpulse'
TITLE = 'GridPulse'
TARGETS = [{'name': 'log_E', 'kind': 'reg'}, {'name': 'log_loss', 'kind': 'reg'}]
HEADLINE_ITEMS = ['inertia N=60', 'inertia N=10', 'MW lost medium N=60']
HEADLINE = 'mean of MdAPE (%) over GB inertia at N=60 s, inertia at N=10 s and MW lost (medium-confidence labels, N=60) (negated)'
LC_KS = []          # only 4 dev events: no learning curve below the full dev set
NS = (10, 60)
E_DEV = 230.0       # GW s, the card's constant-inertia baseline


def _mods():
    import common  # noqa
    import dataset as DS, evaluate as EV, bank as BK
    import griddata as gd
    return DS, EV, BK, gd


@functools.lru_cache(maxsize=1)
def _net():
    import torch
    DS, EV, BK, gd = _mods()
    net, s = EV.load('r5')
    return net, s


def norm(X):
    net, s = _net()
    return ((np.asarray(X, np.float64) - s['xm']) / s['xs']).astype(np.float32)


def P_predict(X):
    import torch
    net, s = _net()
    with torch.no_grad():
        mu, ls = net(torch.tensor(norm(X), dtype=torch.float32))
    mu = mu.numpy() * s['ys'] + s['ym']
    return {'log_loss': mu[:, 0].astype(float), 'log_E': mu[:, 1].astype(float)}


def _feat(w, N, f0, demand):
    DS = _mods()[0]
    return DS.features(w, N, np.ones(len(w), np.float32), f0, demand, True)


@functools.lru_cache(maxsize=1)
def sim_pool():
    DS, EV, BK, gd = _mods()
    dev, L = zip(*[BK.load(b) for b in ('train_p1', 'train_p1b')])
    dev = np.concatenate(dev); L = pd.concat(L, ignore_index=True)
    keep = DS.selected(dev, L) & (L.system.values == 'GB')
    dev, L = dev[keep], L[keep].reset_index(drop=True)
    rng = np.random.default_rng(0)
    noise = DS.noise_split('train')
    half = rng.random(len(L)) < 0.5
    Xs, Ys = [], []
    for N, sel in ((10, half), (60, ~half)):
        idx = np.where(sel)[0]
        Ls = L.iloc[idx].reset_index(drop=True)
        meas, Nn, y = DS.make(dev[idx], Ls, rng, noise, 'mean', 0.0, Nfix=N)
        dem = Ls.demand_MW.values
        d = np.where(rng.random(len(dem)) < 0.25, np.nan, dem * rng.normal(1, 0.03, len(dem)))
        Xs.append(_feat(meas['w'], np.full(len(idx), N), Ls.f0.values, d)); Ys.append(y)
    X = np.concatenate(Xs); y = np.concatenate(Ys)
    return X, {'log_loss': y[:, 0].astype(float), 'log_E': y[:, 1].astype(float)}, None


@functools.lru_cache(maxsize=1)
def sim_val():
    import common
    z = np.load(f'{common.DATA}/val_p2s.npz', allow_pickle=True)
    gb = z['system'] == 'GB'
    w, y, f0, dem = z['w'][gb], z['y'][gb], z['f0'][gb], z['demand'][gb]
    X = np.concatenate([_feat(w, np.full(len(w), N), f0, dem) for N in NS])
    yy = np.concatenate([y, y])
    Nc = np.concatenate([np.full(len(w), N) for N in NS])
    # simulated loss labels are exact: scored as the 'medium' tier so the headline items exist on the frozen set
    return X, {'log_loss': yy[:, 0].astype(float), 'log_E': yy[:, 1].astype(float), '_N': Nc,
               '_conf': np.array(['medium'] * len(Nc)), '_ok_loss': np.ones(len(Nc), bool)}, None


def _rows(ev, demand_of):
    DS, EV, BK, gd = _mods()
    W, ids, dem, f0 = [], [], [], []
    for _, r in ev.iterrows():
        m = EV.real_gb(r.event_id)
        W.append(m['w'][0]); ids.append(r.event_id); dem.append(demand_of(r)); f0.append(50.0)
    W = np.array(W); dem = np.array(dem, float)
    X = np.concatenate([_feat(W, np.full(len(W), N), np.full(len(W), 50.0), dem) for N in NS])
    Nc = np.concatenate([np.full(len(W), N) for N in NS])
    # demand per row (tiled over N like the features); before 19:26 this was per event, which broke the k x demand
    # baseline at score time (shape error, raised before any score was written)
    return X, np.array(ids * len(NS)), Nc, np.concatenate([dem] * len(NS))


@functools.lru_cache(maxsize=1)
def _dev():
    DS, EV, BK, gd = _mods()
    ev = gd.events('GB', split='dev').reset_index()
    ev = ev[ev.event_id.isin(EV.GB_DEV)].reset_index(drop=True)
    X, ids, Nc, dem = _rows(ev, lambda r: EV.DEV_DEMAND[r.event_id])
    e = ev.set_index('event_id')
    E = np.array([e.loc[i].inertia_outturn_GVAs for i in ids], float) * 1e3
    loss = np.array([EV.DEV_LOSS[i] for i in ids], float)
    return X, ids, Nc, E, loss


def dev():
    X, ids, Nc, E, loss = _dev()
    return X, {'log_E': np.log(E), 'log_loss': np.log(loss), '_N': Nc, '_conf': np.array(['medium'] * len(ids)),
               '_ok_loss': np.ones(len(ids), bool)}, ids


def dev_groups():
    return _dev()[1]


@functools.lru_cache(maxsize=1)
def _exam():
    DS, EV, BK, gd = _mods()
    ev = gd.events('GB', split='exam').reset_index(drop=True)
    gm = gd.gb_genmix().set_index('time_utc')['GENERATION']
    X, ids, Nc, dem = _rows(ev, lambda r: float(gm.get(pd.Timestamp(r.t0_utc).floor('30min'), np.nan)))
    return X, ids, Nc, dem, ev


def exam_X():
    X, ids, Nc, dem, ev = _exam()
    return X, ids


def exam_units():
    return _exam()[1]


def exam_truth():
    """Only called by score.py after PREREG is frozen."""
    X, ids, Nc, dem, ev = _exam()
    e = ev.set_index('event_id')
    E = np.array([e.loc[i].inertia_outturn_GVAs for i in ids], float) * 1e3
    loss = np.array([e.loc[i].loss_MW for i in ids], float)
    conf = np.array([str(e.loc[i].loss_confidence) for i in ids])
    rocof = np.array([e.loc[i].rocof_1s_Hz_s for i in ids], float)
    return {'log_E': np.log(E), 'log_loss': np.log(loss), '_N': Nc, '_conf': conf, '_ok_loss': rocof > 0,
            '_rocof': rocof, '_dem': dem}, ids


def items(Y, P):
    out = {}
    E, Ep = np.exp(Y['log_E']) / 1e3, np.exp(np.asarray(P['log_E'], float)) / 1e3
    L_, Lp = np.exp(Y['log_loss']), np.exp(np.asarray(P['log_loss'], float))
    for N in NS:
        s = Y['_N'] == N
        out[f'inertia N={N}'] = (np.where(s, E, np.nan), np.where(s, Ep, np.nan), ['mape'])
    for tier in ('medium', 'low'):
        for N in (60, 10):
            s = (Y['_N'] == N) & (Y['_conf'] == tier) & Y['_ok_loss']
            out[f'MW lost {tier} N={N}'] = (np.where(s, L_, np.nan), np.where(s, Lp, np.nan), ['mape'])
    return out


def baselines_exam():
    """The card's baselines on the same events: constant inertia 230 GW s (dumb), k x demand (k from the 4 dev events),
    dev-median loss (dumb), loss from RoCoF assuming 230 GW s (classical)."""
    DS, EV, BK, gd = _mods()
    X, ids, Nc, dem, ev = _exam()
    Y, _ = exam_truth()
    evd = gd.events('GB', split='dev').set_index('event_id')
    k = float(np.median([evd.loc[e].inertia_outturn_GVAs / EV.DEV_DEMAND[e] for e in EV.GB_DEV]))
    loss_med = float(gd.events('GB', split='dev').loss_MW.median())
    n = len(ids)
    roc = 2 * E_DEV * 1e3 * Y['_rocof'] / 50.0
    return {'constant 230 GW s / dev-median loss (dumb)': {'log_E': np.full(n, np.log(E_DEV * 1e3)),
                                                          'log_loss': np.full(n, np.log(loss_med))},
            'k x demand / RoCoF with 230 GW s (classical)': {'log_E': np.log(np.maximum(k * dem, 1e-9) * 1e3),
                                                            'log_loss': np.log(np.maximum(roc, 1e-9))}}


def metrics(t):
    return ['mape']
