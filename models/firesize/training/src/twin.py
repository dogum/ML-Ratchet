"""Twin test on dev twins (calib half): does the network read the real ignition and its simulated twin alike?
For each twin: network prediction from the REAL features; outcome A = real reported size; outcome B = the calibrated
simulator's size distribution for the same ignition in its own landscape and weather.
usage: twin.py <model npz> <physics> <theta>  -> results/twin_<model>.json"""
import sys, json, os
from common import *
import supp, calib, net, train
from sklearn.metrics import roc_auc_score


def main():
    mp, phys, theta = sys.argv[1], sys.argv[2], sys.argv[3]
    W = dict(np.load(mp))
    T = calib.Twins(phys)
    th = train.theta_of(theta)
    Pi, ess = T.probs(th)                       # simulated twin distribution per twin ignition
    I = pd.read_parquet(f'{DATA}/ignitions.parquet').set_index('ign').loc[T.uign]
    Xd = pd.read_parquet(f'{DATA}/feat_dev.parquet').set_index('FOD_ID').loc[I.FOD_ID.values]
    Pf = net.forward_np(W, Xd[list(W['feats'])].values)
    w = T.wt_ign
    y_real = (I.escape.values == 1).astype(int)
    p_twin = Pi[:, supp.ESC_BIN:].sum(1)
    pe = Pf[:, supp.ESC_BIN:].sum(1)
    rs = np.random.default_rng(0)
    y_twin = (rs.uniform(size=len(p_twin)) < p_twin).astype(int)
    med_net = supp.median_log10(Pf); med_twin = supp.median_log10(Pi)
    lo = np.log10(I.size_ha.values)
    res = dict(n=int(len(w)),
               esc_rate=dict(real=float(np.sum(w * y_real) / w.sum()), twin=float(np.sum(w * p_twin) / w.sum()),
                             net=float(np.sum(w * pe) / w.sum())),
               net_auc_vs_real=float(roc_auc_score(y_real, pe, sample_weight=w)),
               net_auc_vs_twin_draw=float(roc_auc_score(y_twin, pe, sample_weight=w)),
               twin_auc_vs_real=float(roc_auc_score(y_real, p_twin, sample_weight=w)),
               corr_net_twin_pesc=float(np.corrcoef(np.log(pe), np.log(np.maximum(p_twin, 1e-4)))[0, 1]),
               mae_net_vs_real=float(np.median(np.abs(med_net - lo))),
               mae_twin_vs_real=float(np.median(np.abs(med_twin - lo))),
               mae_net_vs_twinmed=float(np.median(np.abs(med_net - med_twin))))
    os.makedirs(f'{ROOT}/results', exist_ok=True)
    json.dump(res, open(f'{ROOT}/results/twin_{os.path.basename(mp)[:-4]}.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
