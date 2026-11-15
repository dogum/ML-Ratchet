"""Dev-fitted sigma widening (widen-only): factor = max(1, 68th percentile of |error| / sigma) on the dev events.
GB net: loss, inertia (N=60), nadir and settling (N=10) on the 4 GB dev events; MW/Hz, headroom, FFR have no real
truth -> 1. ERCOT summary heads: 5 ERCOT dev events. Usage: python3 widen.py <rung> <ercot_head>"""
from common import *
import evaluate as EV, ercot_head as H
import griddata as gd


def gb(rung):
    _, d = EV.dev_gb(rung)
    a = d[d.N == 60]; b = d[d.N == 10]
    out = [1.0] * 7
    w = np.array([EV.real_gb(e)['w'][0] for e in GB_DEV]); dem = np.array([EV.DEV_DEMAND[e] for e in GB_DEV])
    mu60, sg60 = EV.predict(rung, w, np.full(4, 60), np.ones(4), np.full(4, 50.0), dem)
    mu10, sg10 = EV.predict(rung, w, np.full(4, 10), np.ones(4), np.full(4, 50.0), dem)
    zl = np.abs(mu60[:, 0] - np.log(a.loss_true.values)) / sg60[:, 0]
    zE = np.abs(mu60[:, 1] - np.log(a.E_true.values * 1e3)) / sg60[:, 1]
    zn = np.abs(mu10[:, 5] - b.nadir_true.values * 10) / sg10[:, 5]
    zs = np.abs(mu10[:, 6] - b.settle_true.values * 10) / sg10[:, 6]
    q = lambda z: float(max(1.0, np.quantile(z, 0.68)))
    out[0], out[1], out[5], out[6] = q(zl), q(zE), q(zn), q(zs)
    return out, dict(z_loss=zl.tolist(), z_E=zE.tolist(), z_nadir10=zn.tolist(), z_settle10=zs.tolist())


def ercot(name):
    Z = np.load(f'{CKPT}/ercot_head_{name}.npz')
    e = gd.events('ERCO', split='dev'); T = H.features_from_events(e); out = {}; zs = {}
    for k, (xs, y) in H.SPECS.items():
        W = {a.split('__')[1]: Z[a] for a in Z if a.startswith(k + '__')}
        mu, sg = H.np_pred(W, T[xs].values); z = np.abs(mu - T[y].values) / sg
        out[k] = float(max(1.0, np.quantile(z, 0.68))); zs[k] = z.tolist()
    return out, zs


if __name__ == '__main__':
    g, gz = gb(sys.argv[1]); e, ez = ercot(sys.argv[2])
    res = dict(rung=sys.argv[1], gb_factors=g, gb_z=gz, ercot_head=sys.argv[2], ercot_factors=e, ercot_z=ez)
    print(json.dumps(res, indent=1)); json.dump(res, open(f'{ROOT}/widen.json', 'w'), indent=1)
