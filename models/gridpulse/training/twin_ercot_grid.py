from common import *
import gpsim, measure as M, griddata as gd
ev = gd.events('ERCO', split='dev')
grid = [dict(dc_p=0.0, D_med=2.5, ffr_p=0.3, ffr=(0, 300)), dict(dc_p=0.0, D_med=2.0, ffr_p=0.3, ffr=(0, 300)),
        dict(dc_p=0.0, D_med=1.5, ffr_p=0.3, ffr=(0, 300)), dict(dc_p=0.0, D_med=2.0, ffr_p=0.3, ffr=(0, 300), gov_head=(1000, 3500)),
        dict(dc_p=0.0, D_med=2.5, ffr_p=0.3, ffr=(0, 300), gov_head=(1000, 3500))]
for i, kn in enumerate(grid):
    name = gpsim.derive('p0', f'e{i}', ERCO=kn); rat = []
    for _, r in ev.iterrows():
        n = 40
        S, E, L = gpsim.sample(n, 11, 'ERCO', name, fix=dict(demand_MW=r.load_MW, inertia_MWs=r.inertia_MWs,
                                                              loss_MW=r.mw_loss, ibr_share=r.renew_share_pct / 100))
        out = gpsim.run(S, E)
        dev = np.array([o['f'] - 60 for o in out])
        dn = -M.to_1s(dev, np.random.default_rng(1).uniform(0, 1, n)).min(1)
        rat.append(float(np.median(dn) / r.delta_f_Hz))
    print(i, kn, np.round(rat, 2).tolist(), 'mean|log|', round(float(np.mean(np.abs(np.log(rat)))), 3), flush=True)
