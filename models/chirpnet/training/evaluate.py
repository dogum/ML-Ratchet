"""Fixed validation: sensitivity at a fixed false-alarm probability, and parameter accuracy, on held-out waveforms."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC)
from model import ChirpNet, ChirpNet2S, Cascade
from sampler import Sampler
import gwphys as G
SNRS = [5, 6, 7, 8, 9, 10, 12, 15, 20]

def load(path):
    ck = torch.load(path, weights_only=False)
    if ck.get('arch') == 'cascade':
        return Cascade(load(ck['parts'][0]), load(ck['parts'][1]), ck['veto']).eval()
    if ck.get('arch') == '2s': net = ChirpNet2S(widths=tuple(ck['widths']))
    else: net = ChirpNet(widths=tuple(ck['widths']), pos=bool(ck.get('pos', 0)), hidden=ck.get('hidden', 128))
    net.load_state_dict(ck['ema']); net.eval()
    return net

@torch.no_grad()
def run(net, x, bs=512, tta=False):
    outs = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs])
        o = net(xb)
        if tta:
            o2 = net(torch.roll(xb, 24, -1)); o3 = net(torch.roll(xb, -24, -1))
            o = (o + o2 + o3) / 3
        outs.append(o.numpy())
    return np.concatenate(outs)

def evaluate(net, n_noise=20000, n_per=800, seed=777, tta=False, glitch=False):
    bank = np.load(_P.WORK + '/bank_val.npy'); par = np.load(_P.WORK + '/par_val.npy'); sky = np.load(_P.WORK + '/sky_val.npy')
    S = Sampler(bank, par, sky, seed=seed)
    scores = []
    for i in range(0, n_noise, 2000):
        x, y, t = S.batch(2000, p_sig=0.0, p_glitch=0.5 if glitch else 0.0)
        scores.append(run(net, x, tta=tta)[:, 0])
    s0 = np.concatenate(scores)
    thr3 = np.quantile(s0, 1 - 1e-3); thr4 = np.quantile(s0, 1 - 1e-4) if n_noise >= 20000 else None
    res = {'thr_fap1e-3': float(thr3), 'eff': {}, 'eff_1e-4': {}}
    allo, allt = [], []
    for snr in SNRS:
        x, y, t = S.batch(n_per, snr_lo=snr, snr_hi=snr, p_sig=1.0)
        o = run(net, x, tta=tta)
        res['eff'][snr] = float(np.mean(o[:, 0] > thr3)); res['eff_1e-4'][snr] = float(np.mean(o[:, 0] > thr4))
        allo.append(o); allt.append(t)
    o = np.concatenate(allo); t = np.concatenate(allt)
    hi = t[:, 3] >= 10
    dl = o[hi, 1] - t[hi, 0]; sig = np.exp(np.clip(o[hi, 2], -7, 3))
    res['mc_relerr_med'] = float(np.median(np.abs(np.expm1(dl))))
    res['mc_cov68'] = float(np.mean(np.abs(dl) < sig))
    res['q_abserr_med'] = float(np.median(np.abs(o[hi, 3] - t[hi, 1])))
    res['t_abserr_ms'] = float(1000 * np.median(np.abs(o[hi, 5] - t[hi, 2])))
    fa = {}
    for snr in (8, 12, 20):
        S.rng = np.random.default_rng(seed + snr)
        xs = G.bandlimited_noise(S.rng, (n_per, 2, G.WIN)) + S.single(n_per, snr, snr)
        fa[snr] = float(np.mean(run(net, xs.astype(np.float32), tta=tta)[:, 0] > thr3))
    res['single_fa'] = fa
    mid = [6, 7, 8, 9, 10, 12]
    res['sens'] = float(np.mean([res['eff'][s] for s in mid]))
    # mc error at the threshold region too
    lo = (t[:, 3] >= 8) & (t[:, 3] < 10)
    res['mc_relerr_snr8'] = float(np.median(np.abs(np.expm1(o[lo, 1] - t[lo, 0]))))
    return res

if __name__ == '__main__':
    net = load(sys.argv[1])
    r = evaluate(net, tta='--tta' in sys.argv, glitch='--glitch' in sys.argv)
    print(json.dumps(r))
