"""Scoring for SwellCast: network predictions, metrics, skill objective on the frozen simulated set (5 skills: log D, t0,
direction, log storm Hs, detection = 2 AUC - 1 against version-matched null windows), and real dev/exam sets."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M, events as EV
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CK = os.path.join(ROOT, 'ckpt/'); BANK = os.path.join(ROOT, 'bank/')
VAL = os.environ.get('SWELL_VAL', 'val1')
SIGF = os.path.join(ROOT, 'sigma_factor.json')

def load_net(ck):
    d = torch.load(ck if os.path.exists(ck) else CK + ck, weights_only=False); net = M.SwellNet(); net.load_state_dict(d['state']); net.eval(); return net

def predict(net, X, bs=128, sigf=None):
    o = []
    with torch.no_grad():
        for i in range(0, len(X), bs): o.append(net(torch.from_numpy(M.features(np.asarray(X[i:i + bs])))).numpy())
    o = np.concatenate(o); mu, sd = net.mu_y.numpy(), net.sd_y.numpy(); n = len(M.OUT)
    m = o[:, :n] * sd + mu; s = np.exp(np.clip(o[:, n:2 * n], -6, 3)) * sd; v = o[:, 2 * n:2 * n + 2]
    f = sigf or {}
    return dict(logD=m[:, 0], logD_sig=s[:, 0] * f.get('logD', 1), D_km=10 ** m[:, 0], t0=m[:, 1], t0_sig=s[:, 1] * f.get('t0', 1), loghs=m[:, 2], loghs_sig=s[:, 2] * f.get('loghs', 1),
                dir=np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360, kappa=np.sqrt((v ** 2).sum(1)) / f.get('dir', 1) ** 2,
                pdet=1 / (1 + np.exp(-o[:, 2 * n + 2])) if o.shape[1] > 2 * n + 2 else np.full(len(o), np.nan))

def angdiff(a, b): return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)

def metrics(P, D_km, t0, dirt, loghs=None, has_dir=None):
    eD = np.log10(P['D_km']) - np.log10(D_km); et = P['t0'] - t0; ok = np.isfinite(eD)
    r = dict(n=int(len(D_km)), n_answered=int(ok.sum()), logD_mae=float(np.nanmean(np.abs(eD))), D_med_pct=float(np.nanmedian(np.abs(10 ** eD - 1)) * 100),
             D_within_20pct=float(np.nanmean(np.abs(10 ** eD - 1) < 0.2)), D_bias_pct=float((10 ** np.nanmedian(eD) - 1) * 100),
             t0_mae_h=float(np.nanmean(np.abs(et))), t0_med_h=float(np.nanmedian(np.abs(et))))
    if has_dir is None: has_dir = np.ones(len(D_km), bool)
    if 'dir' in P and has_dir.any():
        ea = angdiff(P['dir'][has_dir], np.asarray(dirt)[has_dir]); r.update(dir_med_deg=float(np.nanmedian(ea)), dir_mean_deg=float(np.nanmean(ea)), dir_within_30=float(np.nanmean(ea < 30)), n_dir=int(has_dir.sum()))
    if loghs is not None and 'loghs' in P: r['loghs_mae'] = float(np.nanmean(np.abs(P['loghs'] - loghs)))
    if 'logD_sig' in P:
        r['logD_in_1sig'] = float(np.mean(np.abs(eD) < P['logD_sig'])); r['t0_in_1sig'] = float(np.mean(np.abs(et) < P['t0_sig']))
        if 'kappa' in P and has_dir.any(): r['dir_in_1sig'] = float(np.mean(angdiff(P['dir'], dirt)[has_dir] < np.degrees(1 / np.sqrt(np.maximum(P['kappa'][has_dir], 1e-3)))))
    return r

def val_set(name=None):
    name = name or VAL; X = np.load(BANK + name + '_x.npy', mmap_mode='r'); Y = np.load(BANK + name + '_y.npy'); return X, Y

def auc(pos, neg):
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))

def null_name(name): return 'null_val2' if (name or VAL) >= 'val2' else 'null_val1'

def det_threshold(net, fa=0.10):
    """detection threshold giving a 10% false-alarm rate on the frozen REAL null set (quiet 2024 windows, nothing injected)"""
    Xn = np.load(BANK + 'null_val1_x.npy', mmap_mode='r'); return float(np.quantile(predict(net, Xn)['pdet'], 1 - fa))

def quiet(net, thr):
    Xq = np.load(BANK + 'quiet_test_x.npy', mmap_mode='r'); p = predict(net, Xq)['pdet']; return dict(n=len(p), false_alarm=float(np.mean(p > thr)), false_alarm_p05=float(np.mean(p > 0.5)))

def val(net, name=None):
    X, Y = val_set(name); vis = Y[:, 4] >= 0.3; X, Y = X[vis], Y[vis]; P = predict(net, X)
    Pn = predict(net, np.load(BANK + null_name(name) + '_x.npy', mmap_mode='r'))
    hasd = np.isfinite(np.asarray(X[:, 1, :, 4], np.float32)).mean(1) > 0.5
    r = metrics(P, Y[:, 0], Y[:, 1], Y[:, 2], np.log10(Y[:, 3]), hasd)
    mD = np.mean(np.abs(np.log10(Y[:, 0]) - np.median(np.log10(Y[:, 0])))); mt = np.mean(np.abs(Y[:, 1] - np.median(Y[:, 1]))); mh = np.mean(np.abs(np.log10(Y[:, 3]) - np.median(np.log10(Y[:, 3]))))
    sk = dict(D=max(0, 1 - r['logD_mae'] / mD), t0=max(0, 1 - r['t0_mae_h'] / mt), dir=max(0, 1 - r['dir_mean_deg'] / 90), hs=max(0, 1 - r['loghs_mae'] / mh), det=max(0, 2 * auc(P['pdet'], Pn['pdet']) - 1))
    r['det_auc'] = auc(P['pdet'], Pn['pdet'])
    r['skill'] = sk; r['objective'] = float(np.mean(list(sk.values()))); return r

_RW = {}
def real_windows(split, visible='confirmed'):
    k = (split, visible)
    if k not in _RW:
        ev = EV.load(split, visible); X = EV.windows(ev); _RW[k] = (ev, X)
    return _RW[k]

def real(net, split, visible='confirmed', sigf=None, return_pred=False):
    ev, X = real_windows(split, visible); P = predict(net, X, sigf=sigf)
    D = np.array([e['D'] / 1e3 for e in ev]); t0 = np.array([e['t0_rel'] for e in ev]); dr = np.array([e['dir'] for e in ev])
    hasd = np.array([e['dir_coverage'] > 0.5 for e in ev]); r = metrics(P, D, t0, dr, None, hasd)
    w = np.array([e['wind'] for e in ev]); thr = sigf.get('det_thr', 0.5) if sigf else 0.5
    r['detect_rate'] = float(np.mean(P['pdet'] > thr)); r['det_thr'] = thr
    from scipy.stats import spearmanr
    r['hs_vs_wind_spearman'] = float(spearmanr(P['loghs'], w).correlation) if len(w) > 3 else np.nan
    return (r, P, ev) if return_pred else r
