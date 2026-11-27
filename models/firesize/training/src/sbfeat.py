"""Features of the prior_sb.py ignitions (kind 'trainsb'), computed as trainfeat.py and physfeat.build_tables compute
them for the other simulated ignitions, appended to data/feat_sim.parquet (27 Nov 2026). The existing rows are not
recomputed; a copy of the file is kept in data/pre_sb/ first."""
import os, shutil
from common import *
import features as F
import physfeat as PF


def main():
    I = pd.read_parquet(f'{DATA}/ignitions.parquet')
    I = I[I.kind == 'trainsb'].reset_index(drop=True)
    X0 = pd.read_parquet(f'{DATA}/feat_sim.parquet')
    if X0.ign.isin(I.ign).any():
        raise SystemExit('trainsb features already present')
    if not os.path.exists(f'{DATA}/pre_sb/feat_sim.parquet'):
        shutil.copy2(f'{DATA}/feat_sim.parquet', f'{DATA}/pre_sb/feat_sim.parquet')
    z = np.load(f'{DATA}/wx_points_trainsb.npz')
    assert (z['ign'] == I.ign.values).all()
    wx, off, vars_ = z['wx'], int(z['off']), list(z['vars'])
    d0 = pd.DataFrame(wx[:, :, off], columns=vars_)
    dm1 = pd.DataFrame(wx[:, :, off - 1], columns=vars_)
    L = F.land_features(I.x.values + 45, I.y.values + 45)
    W = F.wx_features(d0, dm1)
    R = F.record_features(I.cause.values, I.fed.values, I.hour.values, I.doy.values, I.region.values)
    X = pd.concat([L, W, R], axis=1)
    Pp = PF.phys_features(I.x.values + 45, I.y.values + 45, X.tmmx, X.rmin, X.vs, X.tmmx_m1, X.rmin_m1, X.vs_m1)
    for c in Pp:
        X[c] = Pp[c].values
    X['ign'] = I.ign.values
    X = X[X0.columns]
    pd.concat([X0, X], ignore_index=True).to_parquet(f'{DATA}/feat_sim.parquet')
    print(X.shape, 'pt_SB', float(X.pt_SB.mean()), 'nan frac', float(X.isna().mean().mean()))


if __name__ == '__main__':
    main()
