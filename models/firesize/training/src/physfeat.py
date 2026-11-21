"""Physics features (rung candidate): Rothermel head rate of spread (m/min) of the LANDFIRE fuel models around the
ignition under the discovery day's afternoon weather, from a lookup table built with FireSim's behave().
Afternoon 1-h moisture = Simard EMC at (Tmax, RHmin); 10-h = 1-h + 1; 100-h = gridMET fm100; live herb 60 %, woody 90 %;
10 m wind = gridMET daily mean x 1.45 (the afternoon peak of FireSim's hourly disaggregation); flat; unsheltered WAF.
Features: log1p ROS at the 250 m ignition pixel, mean log1p ROS over the 5 x 5 and 13 x 13 boxes, and the same at the
point for the day before. Same code for sim and real ignitions."""
import os
from common import *
import firesim as fs
from firesim.moisture import simard_emc
import features as F

M1 = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35.0])
U10 = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20.0])
LUTF = f'{DATA}/ros_lut.npz'


def build_lut():
    if os.path.exists(LUTF):
        z = np.load(LUTF)
        return z['lut']
    lut = np.zeros((256, len(M1), len(U10)), np.float32)
    nums = [n for n in range(101, 205)]
    for n in nums:
        b0 = fs.behave(n, m1=6, m10=7, m100=8, wind_speed=1.0)
        if not b0.get('burnable', True) or 'ros_mmin' not in b0:
            continue
        for i, m in enumerate(M1):
            for j, u in enumerate(U10):
                b = fs.behave(n, m1=m, m10=m + 1, m100=m + 2, mherb=60, mwoody=90, wind_speed=u, wind_height='10m',
                              crown='none')
                lut[n, i, j] = b.get('ros_mmin', 0.0)
    np.savez(LUTF, lut=lut)
    return lut


def ros_lookup(lut, code, m1, u):
    i = np.clip(np.interp(m1, M1, np.arange(len(M1))), 0, len(M1) - 1)
    j = np.clip(np.interp(u, U10, np.arange(len(U10))), 0, len(U10) - 1)
    i0 = np.floor(i).astype(int); j0 = np.floor(j).astype(int)
    i1 = np.minimum(i0 + 1, len(M1) - 1); j1 = np.minimum(j0 + 1, len(U10) - 1)
    fi = i - i0; fj = j - j0
    return ((1 - fi) * (1 - fj) * lut[code, i0, j0] + fi * (1 - fj) * lut[code, i1, j0] +
            (1 - fi) * fj * lut[code, i0, j1] + fi * fj * lut[code, i1, j1])


def phys_features(x, y, tmmx, rmin, vs, tmmx1, rmin1, vs1, chunk=50000):
    lut = build_lut()
    a = np.load(f'{DATA}/nat/fuel.npz')['a']
    NY, NX = a.shape
    r = np.floor((F.Y1 - np.asarray(y)) / F.R).astype(np.int64)
    c = np.floor((np.asarray(x) - F.X0) / F.R).astype(np.int64)
    d13 = np.arange(-6, 7)
    m1 = np.clip(simard_emc(np.asarray(tmmx, float), np.asarray(rmin, float)), 1, 35)
    m1b = np.clip(simard_emc(np.asarray(tmmx1, float), np.asarray(rmin1, float)), 1, 35)
    u = np.asarray(vs, float) * 1.45; ub = np.asarray(vs1, float) * 1.45
    out = {k: [] for k in ['ros_pt', 'ros5', 'ros13', 'ros_pt_m1']}
    for s in range(0, len(r), chunk):
        sl = slice(s, s + chunk)
        rr = np.clip(r[sl, None, None] + d13[None, :, None], 0, NY - 1)
        cc = np.clip(c[sl, None, None] + d13[None, None, :], 0, NX - 1)
        p = a[rr, cc].reshape(len(rr), -1).astype(int)              # m x 169 codes
        mm = np.nan_to_num(m1[sl], nan=10.0)[:, None].repeat(169, 1)
        uu = np.nan_to_num(u[sl], nan=3.0)[:, None].repeat(169, 1)
        R = np.log1p(ros_lookup(lut, p, mm, uu))
        R = R.reshape(-1, 13, 13)
        out['ros_pt'].append(R[:, 6, 6])
        out['ros5'].append(R[:, 4:9, 4:9].reshape(len(R), -1).mean(1))
        out['ros13'].append(R.reshape(len(R), -1).mean(1))
        out['ros_pt_m1'].append(np.log1p(ros_lookup(lut, p[:, 84], np.nan_to_num(m1b[sl], nan=10.0),
                                                    np.nan_to_num(ub[sl], nan=3.0))))
    return pd.DataFrame({k: np.concatenate(v).astype(np.float32) for k, v in out.items()})


if __name__ == '__main__':
    import time
    t = time.time(); lut = build_lut(); print('lut', time.time() - t, lut[102, 5, 5], lut[165, 5, 5], lut[91, 5, 5])


def build_tables():
    """Append physics features to data/feat_sim.parquet and data/feat_dev.parquet."""
    I = pd.read_parquet(f'{DATA}/ignitions.parquet')
    X = pd.read_parquet(f'{DATA}/feat_sim.parquet')
    assert (X.ign.values == I.ign.values).all()
    Pp = phys_features(I.x.values + 45, I.y.values + 45, X.tmmx, X.rmin, X.vs, X.tmmx_m1, X.rmin_m1, X.vs_m1)
    for c in Pp:
        X[c] = Pp[c].values
    X.to_parquet(f'{DATA}/feat_sim.parquet')
    d = fod_records('dev')
    Xd = pd.read_parquet(f'{DATA}/feat_dev.parquet')
    assert (Xd.FOD_ID.values == d.FOD_ID.values).all()
    Pd = phys_features(d.x.values, d.y.values, Xd.tmmx, Xd.rmin, Xd.vs, Xd.tmmx_m1, Xd.rmin_m1, Xd.vs_m1)
    for c in Pd:
        Xd[c] = Pd[c].values
    Xd.to_parquet(f'{DATA}/feat_dev.parquet')
    print(Pp.describe().T, Pd.describe().T)
