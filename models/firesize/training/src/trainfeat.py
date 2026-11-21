"""Features of the simulated (train / twin) ignitions, computed exactly like the real ones (features.py).
-> data/feat_sim.parquet (row per ignition, column ign)"""
from common import *
import features as F


def main():
    I = pd.read_parquet(f'{DATA}/ignitions.parquet')
    z = np.load(f'{DATA}/wx_points.npz')
    wx, off, vars_ = z['wx'], int(z['off']), list(z['vars'])
    assert (z['ign'] == I.ign.values).all()
    d0 = pd.DataFrame(wx[:, :, off], columns=vars_)
    dm1 = pd.DataFrame(wx[:, :, off - 1], columns=vars_)
    L = F.land_features(I.x.values + 45, I.y.values + 45)
    W = F.wx_features(d0, dm1)
    R = F.record_features(I.cause.values, I.fed.values, I.hour.values, I.doy.values, I.region.values)
    X = pd.concat([L, W, R], axis=1)
    X['ign'] = I.ign.values
    X.to_parquet(f'{DATA}/feat_sim.parquet')
    print(X.shape, 'nan frac', float(X.isna().mean().mean()))


if __name__ == '__main__':
    main()
