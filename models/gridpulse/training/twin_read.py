"""Twin test (dev): simulate twins of each GB dev event (its NESO inertia, demand and loss; everything else from the
prior), measure them like NESO with real noise, and check whether the network reads twins and the real trace alike.
Reports the percentile of the real reading among the twin readings (50 % = read alike). Usage: twin_read.py <rung> <prior>"""
from common import *
import gpsim, measure as M, dataset as DS, evaluate as EV, twin_gb as T
rung, ver = sys.argv[1], sys.argv[2]
noise = DS.noise_split('val'); res = {}
for eid in GB_DEV:
    n = 80
    E = gd_E = float(__import__('griddata').events('GB', split='dev').set_index('event_id').loc[eid].inertia_outturn_GVAs) * 1e3
    loss = T.LOSSES[eid]
    S, _, L = gpsim.sample(n, 31, 'GB', ver, fix=dict(inertia_MWs=E, demand_MW=EV.DEV_DEMAND[eid], loss_MW=loss[0][1]))
    out = gpsim.run(S, [loss] * n); dev = np.array([o['f'] - 50 for o in out])
    rng = np.random.default_rng(3)
    d1 = M.to_1s(dev, rng.uniform(0, 1, n), 'sample')
    nz = noise[rng.integers(len(noise), size=n)]; nz = nz - nz[:, :60].mean(1, keepdims=True)
    x = M.compose(d1, nz); w = M.window(x, M.onset(x))['w']
    real = EV.real_gb(eid)['w']
    row = {}
    for N in (10, 60):
        tw = EV.readout(rung, w, np.full(n, N), np.ones(n), np.full(n, 50.0), np.full(n, EV.DEV_DEMAND[eid]))
        rr = EV.readout(rung, real, [N], [1.0], [50.0], [EV.DEV_DEMAND[eid]])
        for k in ('loss_MW', 'E_GWs', 'beta_MW_Hz'):
            row[f'{k}_N{N}'] = dict(real=float(rr[k][0]), twin_median=float(np.median(tw[k])),
                                    real_pctile=float((tw[k] < rr[k][0]).mean()))
    res[eid] = row; print(eid, json.dumps({k: (round(v['real']), round(v['twin_median']), round(v['real_pctile'], 2)) for k, v in row.items()}), flush=True)
json.dump(res, open(f'{ROOT}/checks/twin_read_{rung}.json', 'w'), indent=1)
