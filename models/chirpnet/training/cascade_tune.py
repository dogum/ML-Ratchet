"""Pick the coincidence-veto threshold for the cascade (rung 4 finds, rung 8 confirms) on simulations with a fresh seed,
not on the ratchet's validation draw and never on real data."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC)
from evaluate import load, run
from sampler import Sampler
import gwphys as G
torch.set_num_threads(2)
a, b = load(_P.WORK + '/ckpt/r4_polish.pt'), load(_P.WORK + '/ckpt/r8_two_stream.pt')
S = Sampler(np.load(_P.WORK + '/bank_val.npy'), np.load(_P.WORK + '/par_val.npy'), np.load(_P.WORK + '/sky_val.npy'), seed=4242)
xn = np.concatenate([S.batch(2000, p_sig=0.0)[0] for _ in range(5)]); thr = np.quantile(run(a, xn)[:, 0], 1 - 1e-3)
inj = {}; 
for snr in (6, 7, 8, 9, 10, 12):
    x, _, _ = S.batch(600, snr, snr, p_sig=1.0); inj[snr] = (run(a, x)[:, 0], run(b, x)[:, 0])
sg = {}
for snr in (12, 20):
    x = (G.bandlimited_noise(S.rng, (600, 2, G.WIN)) + S.single(600, snr, snr)).astype(np.float32); sg[snr] = (run(a, x)[:, 0], run(b, x)[:, 0])
base = np.mean([np.mean(inj[s][0] > thr) for s in inj]); base_fa = np.mean([np.mean(sg[s][0] > thr) for s in sg])
print('r4 alone: sens %.3f  loud single FA %.3f' % (base, base_fa))
rows = []
for v in np.arange(-12, 4.01, 0.5):
    sens = np.mean([np.mean((inj[s][0] > thr) & (inj[s][1] > v)) for s in inj]); fa = np.mean([np.mean((sg[s][0] > thr) & (sg[s][1] > v)) for s in sg])
    rows.append((float(v), float(sens), float(fa))); print('v %5.1f  sens %.3f (%.3f)  loud single FA %.3f' % (v, sens, sens - base, fa))
# rule: largest sensitivity whose loud-single false alarms are at most a quarter of r4's (twice the margin rule 4 demands)
ok = [r for r in rows if r[2] <= 0.25 * base_fa]; best = max(ok, key=lambda r: (r[1], -r[0]))
print('chosen veto', best); json.dump({'veto': best[0], 'rows': rows, 'base': [base, base_fa]}, open(_P.WORK + '/cascade_tune.json', 'w'))
