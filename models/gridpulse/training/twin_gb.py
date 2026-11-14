"""Dev twin / prior-predictive check for the 4 GB dev events: hold each event's own NESO inertia, demand (NESO
generation mix) and loss label fixed, draw everything else from the prior, simulate, measure like NESO, and compare
with the real 1 s trace. Usage: python3 twin_gb.py <prior_version> [mode] [tau]"""
from common import *
import gpsim, measure as M
import griddata as gd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

DEMAND = {'GB_20190809T155233': 33664, 'GB_20190701T072734': 32737, 'GB_20190711T075329': 32998,
          'GB_20191203T200030': 43733}
LOSSES = {'GB_20190809T155233': [(0.0, 1131.0), (1.2, 350.0), (58.0, 210.0)],
          'GB_20190701T072734': [(0.0, 1000.0)], 'GB_20190711T075329': [(0.0, 1153.0)],
          'GB_20191203T200030': [(0.0, 854.0)]}           # B1610 output; B's 1,788 MW label is checked separately


def real_window(eid):
    meta, t, f = gd.event_trace(eid)
    off = int(round((pd.Timestamp(meta['event_t0_utc']) - pd.Timestamp(meta['start_utc'])).total_seconds()))
    x = (f - 50.0)[None, :]
    return M.window(x, np.array([int(np.where(t == off)[0][0])]))


def twins(eid, version, n=120, seed=7, mode='mean', tau=0.0, losses=None):
    ev = gd.events('GB', split='dev').set_index('event_id').loc[eid]
    E = float(ev.inertia_outturn_GVAs) * 1e3
    losses = losses or LOSSES[eid]
    S, _, L = gpsim.sample(n, seed, 'GB', version, fix=dict(inertia_MWs=E, demand_MW=DEMAND[eid], loss_MW=losses[0][1]))
    out = gpsim.run(S, [losses] * n)
    dev = np.array([o['f'] - 50.0 for o in out])
    rng = np.random.default_rng(seed)
    d1 = M.to_1s(dev, rng.uniform(0, 1, n), mode, tau)
    x = M.compose(d1, np.zeros((n, 140)))
    t0 = M.onset(x)
    return M.window(x, t0), L


if __name__ == '__main__':
    ver = sys.argv[1] if len(sys.argv) > 1 else 'p0'
    mode = sys.argv[2] if len(sys.argv) > 2 else 'mean'
    tau = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    fig, ax = plt.subplots(2, 2, figsize=(12, 7)); res = {}
    tt = np.arange(-M.PRE, M.POST + 1)
    for a, eid in zip(ax.flat, GB_DEV):
        r = real_window(eid); s, L = twins(eid, ver, mode=mode, tau=tau)
        q = np.percentile(s['w'], [5, 25, 50, 75, 95], axis=0)
        a.fill_between(tt, q[0], q[4], alpha=.2, color='C1'); a.fill_between(tt, q[1], q[3], alpha=.3, color='C1')
        a.plot(tt, q[2], color='C1', label=f'sim twins ({ver})'); a.plot(tt, r['w'][0], 'k.-', ms=3, label='NESO 1 s')
        a.set_title(eid); a.grid(); a.set_xlim(-10, 60)
        pn = float((s['nadir'] < r['nadir'][0]).mean()); ps = float((s['settle'] < r['settle'][0]).mean())
        early = r['w'][0][M.PRE + 1:M.PRE + 6]; es = np.median(s['w'][:, M.PRE + 1:M.PRE + 6], 0)
        res[eid] = dict(real_nadir=float(r['nadir'][0]), sim_nadir_med=float(np.median(s['nadir'])),
                        nadir_pctile=pn, real_settle=float(r['settle'][0]), sim_settle_med=float(np.median(s['settle'])),
                        settle_pctile=ps, real_first5=np.round(early, 3).tolist(), sim_first5=np.round(es, 3).tolist())
        print(eid, json.dumps(res[eid]))
    ax[0, 0].legend(); plt.tight_layout(); plt.savefig(f'{FIGS}/twin_gb_{ver}_{mode}{tau}.png', dpi=80)
    json.dump(res, open(f'{ROOT}/logs/twin_gb_{ver}_{mode}{tau}.json', 'w'), indent=1)
