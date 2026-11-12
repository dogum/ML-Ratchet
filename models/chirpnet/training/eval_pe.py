"""Posterior checks: calibration (P-P) on held-out injections, and posteriors for the real events."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC)
from flow import ChirpPE, PARAMS
from pe_sampler import PESampler
import gwphys as G
from realeval import prepare
sys.path.insert(0, _P.SRC); from bank_phys import asd_features

def load(p):
    net = ChirpPE(); net.load_state_dict(torch.load(p, weights_only=False)['ema']); net.eval(); return net

@torch.no_grad()
def pp(net, n=400, ns=1000, seed=99):
    S = PESampler(np.load(_P.WORK + '/pbank_val.npy'), np.load(_P.WORK + '/ppar_val.npy'), np.load(_P.WORK + '/ppid_val.npy'),
                  np.load(_P.WORK + '/sky_val.npy'), np.load(_P.WORK + '/asd_feats.npy'), seed=seed)
    x, th, asd, snr = S.batch(n)
    ranks = np.zeros((n, th.shape[1])); widths = []
    for i in range(n):
        s = net.sample(torch.from_numpy(x[i:i + 1]), torch.from_numpy(asd[i:i + 1]), ns).numpy()
        ranks[i] = (s < th[i]).mean(0)
        widths.append(np.quantile(s, 0.95, 0) - np.quantile(s, 0.05, 0))
    # KS-style max deviation of each parameter's rank CDF from the diagonal
    grid = np.linspace(0, 1, 51)
    cdf = np.stack([(ranks[:, j][:, None] <= grid[None]).mean(0) for j in range(th.shape[1])])
    dev = np.abs(cdf - grid[None]).max(1)
    return {'grid': grid.tolist(), 'cdf': cdf.round(4).tolist(), 'maxdev': dict(zip(PARAMS, dev.round(3).tolist())),
            'width90_median': dict(zip(PARAMS, np.median(np.array(widths), 0).round(3).tolist())), 'n': n}

def real_inputs():
    from realeval import windows
    D = prepare(); out = {}
    for name, d in D.items():
        tev = d['tev'] - d['t0']
        H, L = d['H1'], d['L1']
        feats = []
        for det in ('H1', 'L1'):
            s, _ = G.load_strain(f"{_P.WORK}/data_raw/{G.events()[name]['fn_' + det]}")
            fp, pp_ = G.psd(s); feats.append(asd_features(fp, pp_))
        out[name] = (H, L, tev, np.concatenate(feats).astype(np.float32))
    return out

@torch.no_grad()
def real(net, ns=20000, place=1.35):
    res = {}
    for name, (H, L, tev, asd) in real_inputs().items():
        s0 = int(round((tev - place) * G.FS))
        x = np.stack([H[s0:s0 + G.WIN], L[s0:s0 + G.WIN]])[None].astype(np.float32)
        smp = net.sample(torch.from_numpy(x), torch.from_numpy(asd[None]), ns).numpy()
        res[name] = smp
    return res

if __name__ == '__main__':
    net = load(sys.argv[1])
    r = pp(net); print(json.dumps({k: r[k] for k in ('maxdev', 'width90_median')}))
    R = real(net)
    from astropy.cosmology import Planck15
    from astropy.cosmology import z_at_value
    import astropy.units as u
    zs = np.linspace(0, 1.5, 3001); dls = Planck15.luminosity_distance(zs).to(u.Mpc).value
    for k, s in R.items():
        mc = np.exp(s[:, 0]); q = np.clip(s[:, 1], 0.05, 1); dl = 10 ** s[:, 3]; z = np.interp(dl, dls, zs)
        eta = q / (1 + q) ** 2; M = mc / eta ** 0.6; m1 = M / (1 + q) / (1 + z); m2 = q * M / (1 + q) / (1 + z)
        f = lambda v: f"{np.median(v):.1f} [{np.quantile(v, 0.05):.1f}, {np.quantile(v, 0.95):.1f}]"
        print(f"{k:10s} Mc_det {f(mc)}  m1_src {f(m1)}  m2_src {f(m2)}  chi_eff {np.median(s[:,2]):+.2f} [{np.quantile(s[:,2],0.05):+.2f},{np.quantile(s[:,2],0.95):+.2f}]  dL {f(dl)}  tc {np.median(s[:,5]):.3f}")
