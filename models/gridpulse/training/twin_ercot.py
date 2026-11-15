"""ERCOT dev prior-predictive check: hold each dev event's reported load, inertia, IBR share and MW loss; draw the
rest from the prior; compare the simulated frequency drop (1 s mean, noise-free) with ERCOT's reported delta f."""
from common import *
import gpsim, measure as M
import griddata as gd
ver = sys.argv[1] if len(sys.argv) > 1 else 'p0'
ev = gd.events('ERCO', split='dev')
res = {}
for _, r in ev.iterrows():
    n = 60
    S, E, L = gpsim.sample(n, 11, 'ERCO', ver, fix=dict(demand_MW=r.load_MW, inertia_MWs=r.inertia_MWs,
                                                         loss_MW=r.mw_loss, ibr_share=r.renew_share_pct / 100))
    out = gpsim.run(S, E)
    dev = np.array([o['f'] - 60 for o in out])
    d1 = M.to_1s(dev, np.random.default_rng(1).uniform(0, 1, n))
    dn = -d1.min(1)
    res[r.event_id] = dict(reported=float(r.delta_f_Hz), sim_med=float(np.median(dn)),
                           sim_p10_p90=np.percentile(dn, [10, 90]).round(3).tolist(), pctile=float((dn < r.delta_f_Hz).mean()),
                           ratio=float(np.median(dn) / r.delta_f_Hz))
    print(r.event_id, res[r.event_id], flush=True)
print('mean |log ratio|', np.mean([abs(np.log(v['ratio'])) for v in res.values()]))
json.dump(res, open(f'{ROOT}/logs/twin_ercot_{ver}.json', 'w'), indent=1)
