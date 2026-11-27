"""Features of the dev ignitions (both halves) -> data/feat_dev.parquet (27 Nov 2026).

The shipped workspace's feat_dev.parquet was not in the release; this rebuilds it with the exam's own code path
(features.fod_features on fod_gridmet records, plus physfeat.phys_features), in fod_records('dev') order, so
train.real_dev and twin.py can read it. Dev attributes and weather only; sizes are read by the scorers, not here."""
from common import *
import features as F
import physfeat as PF


def main():
    d = F.fod_gridmet(fod_records('dev'))
    X = F.fod_features(d)
    Pp = PF.phys_features(d.x.values, d.y.values, d.tmmx.values, d.rmin.values, d.vs.values,
                          d.tmmx_m1.values, d.rmin_m1.values, d.vs_m1.values)
    X = pd.concat([X, Pp], axis=1)
    X['FOD_ID'] = d.FOD_ID.values
    X.to_parquet(f'{DATA}/feat_dev.parquet')
    print(X.shape, 'nan frac', float(X.isna().mean().mean()), 'pt_SB', int(X.pt_SB.sum()))


if __name__ == '__main__':
    main()
