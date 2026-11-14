"""Scoring: frozen simulated validation set (skill per target) and the real dev events (gate).
Skill = 1 - MAE / MAE(guess the training median), floored at 0. Objective = mean of skills of
loss@60s, inertia@60s, MW/Hz@60s, nadir@10s, settling@10s (GB and ERCOT pooled)."""
from common import *
import torch, dataset as DS, net as NET, measure as M
import griddata as gd
torch.set_num_threads(1)
NAMES = DS.TARGETS
# dev truth: 3 Dec loss = 854 MW (Elexon B1610 metered output; griddata's 1,788 MW REMIT capacity label is ~2x what the
# measured RoCoF implies at 333 GVA s -- decided on dev only, as Phase 1 suggested)
DEV_LOSS = {'GB_20190809T155233': 1131.0, 'GB_20190701T072734': 1000.0, 'GB_20190711T075329': 1153.0,
            'GB_20191203T200030': 854.0}
# GB transmission demand (NESO generation mix, GENERATION, half hour of the event) -- an input, not a label
DEV_DEMAND = {'GB_20190809T155233': 33664.0, 'GB_20190701T072734': 32737.0, 'GB_20190711T075329': 32998.0,
              'GB_20191203T200030': 43733.0}


def load(rung):
    s = torch.load(f'{CKPT}/{rung}.pt', weights_only=False)
    net = NET.Net(nin=s.get('nin', NET.NIN), hid=tuple(s['hid'])); net.load_state_dict(s['net']); net.eval()
    return net, s


def predict(rung_or_pair, w, N, is_gb, f0, demand=None):
    net, s = load(rung_or_pair) if isinstance(rung_or_pair, str) else rung_or_pair
    if demand is None:
        demand = np.full(len(w), np.nan)
    X = (DS.features(w, N, is_gb, f0, demand, s.get('ctx', False)) - s['xm']) / s['xs']
    with torch.no_grad():
        mu, ls = net(torch.tensor(X, dtype=torch.float32))
    return mu.numpy() * s['ys'] + s['ym'], np.exp(ls.numpy()) * s['ys']


def val_scores(rung, valname):
    z = np.load(f'{DATA}/val_{valname}.npz', allow_pickle=True)
    P = load(rung); ym = P[1]['ym']
    out = {}
    gb = (z['system'] == 'GB').astype(np.float32)
    for N in (5, 10, 60):
        mu, sg = predict(P, z['w'], np.full(len(gb), N), gb, z['f0'], z['demand'] if 'demand' in z else None)
        y = z['y']
        mae = np.abs(mu - y).mean(0); mae0 = np.abs(np.median(y, 0) - y).mean(0)
        cov = (np.abs(mu - y) <= sg).mean(0)
        out[N] = {n: dict(mae=float(mae[i]), skill=float(max(0, 1 - mae[i] / mae0[i])), cov1=float(cov[i]))
                  for i, n in enumerate(NAMES)}
        for sysn in ('GB', 'ERCO'):
            s = z['system'] == sysn
            out[N][sysn] = {n: float(max(0, 1 - np.abs(mu[s, i] - y[s, i]).mean() / np.abs(np.median(y[s, i]) - y[s, i]).mean()))
                            for i, n in enumerate(NAMES)}
    obj = np.mean([out[60]['log_loss']['skill'], out[60]['log_E']['skill'], out[60]['log_beta']['skill'],
                   out[10]['nadir']['skill'], out[10]['settle']['skill']])
    return dict(objective=float(obj), by_N=out)


def real_gb(eid):
    """Real GB window (griddata onset) -> measure.window dict."""
    meta, t, f = gd.event_trace(eid)
    off = int(round((pd.Timestamp(meta['event_t0_utc']) - pd.Timestamp(meta['start_utc'])).total_seconds()))
    x = (f - 50.0)[None, :].astype(np.float64)
    if np.isnan(x).any():
        s = pd.Series(x[0]).interpolate(limit_direction='both').values; x = s[None, :]
    return M.window(x, np.array([int(np.where(np.round(t) == off)[0][0])]))


def readout(rung, w, N, is_gb, f0, demand=None):
    """Physical units: loss MW, E GW s, beta MW/Hz, headroom MW, FFR MW, nadir Hz, settle Hz (+ sigmas as factors /
    Hz). Returns dict of arrays: value, lo, hi (1 sigma)."""
    mu, sg = predict(rung, w, N, is_gb, f0, demand)
    f0 = np.asarray(f0, float); r = {}
    for i, n in enumerate(['loss_MW', 'E_GWs', 'beta_MW_Hz', 'head_MW']):
        sc = 1e-3 if n == 'E_GWs' else 1.0
        r[n] = np.exp(mu[:, i]) * sc; r[n + '_lo'] = np.exp(mu[:, i] - sg[:, i]) * sc; r[n + '_hi'] = np.exp(mu[:, i] + sg[:, i]) * sc
    r['ffr_MW'] = np.exp(mu[:, 4]) - 100; r['ffr_MW_lo'] = np.exp(mu[:, 4] - sg[:, 4]) - 100; r['ffr_MW_hi'] = np.exp(mu[:, 4] + sg[:, 4]) - 100
    for i, n in ((5, 'nadir_Hz'), (6, 'settle_Hz')):
        k = f0 / 50.0 / 10
        r[n] = mu[:, i] * k; r[n + '_sig'] = sg[:, i] * k
    r['log_sig'] = sg[:, :5]
    return r


def dev_gb(rung):
    """Dev errors on the 4 GB dev events (inertia vs NESO outturn; loss vs DEV_LOSS; nadir/settle at N=10 vs measured)."""
    ev = gd.events('GB', split='dev').set_index('event_id')
    rows = []
    for eid in GB_DEV:
        r = real_gb(eid)
        for N in (10, 60):
            o = readout(rung, r['w'], [N], [1.0], [50.0], [DEV_DEMAND[eid]])
            rows.append(dict(event=eid, N=N, loss=float(o['loss_MW'][0]), loss_true=DEV_LOSS[eid],
                             E=float(o['E_GWs'][0]), E_true=float(ev.loc[eid].inertia_outturn_GVAs),
                             E_lo=float(o['E_GWs_lo'][0]), E_hi=float(o['E_GWs_hi'][0]),
                             loss_lo=float(o['loss_MW_lo'][0]), loss_hi=float(o['loss_MW_hi'][0]),
                             nadir=float(o['nadir_Hz'][0]), nadir_true=float(r['nadir'][0]),
                             settle=float(o['settle_Hz'][0]), settle_true=float(r['settle'][0]),
                             beta=float(o['beta_MW_Hz'][0]), head=float(o['head_MW'][0]), ffr=float(o['ffr_MW'][0])))
    d = pd.DataFrame(rows)
    a = d[d.N == 60]; b = d[d.N == 10]
    summ = dict(loss_mdape=float(np.median(np.abs(a.loss / a.loss_true - 1))),
                E_mdape=float(np.median(np.abs(a.E / a.E_true - 1))),
                nadir10_mae=float(np.abs(b.nadir - b.nadir_true).mean()),
                settle10_mae=float(np.abs(b.settle - b.settle_true).mean()))
    return summ, d
