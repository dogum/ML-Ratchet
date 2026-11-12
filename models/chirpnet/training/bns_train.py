"""Binary-neutron-star ChirpNet: 16 s two-detector windows at 1024 Hz (25-500 Hz), tidal IMRPhenomD_NRTidalv2 waveforms.
Same recipe as the black-hole model, scaled to signals that sweep through the band for tens of seconds."""
import _paths as _P  # portable paths (see _paths.py)
import sys, time, json, math, argparse, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, _P.SRC)
import bns as BN
from model import Block, ChirpNet
from train import nll

WINB, FSB = BN.WINB, BN.FSB
T_LO, T_HI = 8.0, 15.8           # merger time inside the 16 s window (s)


class BNSNet(ChirpNet):
    """ChirpNet with a stride-8 stem and one more block, so a 16 s window (16384 samples) costs about what a 2 s BBH window does
    and the deepest features see ~5 s of inspiral."""
    def __init__(self, widths=(16, 24, 32, 48, 64, 96, 128), kernels=(9, 9, 7, 7, 5, 3), stem_k=64, hidden=128):
        nn.Module.__init__(self)
        w = widths; self.pos = False
        self.stem = nn.Sequential(nn.Conv1d(2, w[0], stem_k, stride=8, padding=stem_k // 2 - 4, bias=False), nn.BatchNorm1d(w[0]), nn.GELU())
        self.blocks = nn.Sequential(*[Block(w[i], w[i + 1], kernels[i], 2) for i in range(len(kernels))])
        c = w[len(kernels)]
        self.head = nn.Sequential(nn.Linear(2 * c, hidden), nn.GELU(), nn.Linear(hidden, self.OUT))


class Block2(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sk = nn.Conv2d(cin, cout, 1, stride=stride, bias=False); self.bs = nn.BatchNorm2d(cout)
    def forward(self, x):
        return F.gelu(self.b2(self.c2(F.gelu(self.b1(self.c1(x))))) + self.bs(self.sk(x)))


NFFT, HOP, NBAND, F_LO, F_HI, NFR = 384, 64, 64, 24.0, 450.0, 256

def band_matrix():
    """average STFT power into NBAND log-spaced bands between F_LO and F_HI (each band gets at least one bin)"""
    f = np.arange(NFFT // 2 + 1) * FSB / NFFT; edges = np.geomspace(F_LO, F_HI, NBAND + 1); M = np.zeros((NBAND, len(f)), np.float32)
    for i in range(NBAND):
        k = np.where((f >= edges[i]) & (f < edges[i + 1]))[0]
        if len(k) == 0: k = [int(np.argmin(np.abs(f - np.sqrt(edges[i] * edges[i + 1]))))]
        M[i, k] = 1.0 / len(k)
    return M, np.sqrt(edges[:-1] * edges[1:])


class BNSNetTF(nn.Module):
    """Neutron-star ChirpNet, time-frequency version. A fixed front end turns each detector's 16 s of whitened strain into
    power in 0.375 s frames (2.7 Hz resolution, long enough to integrate a slowly sweeping chirp coherently), averaged into
    64 log-spaced bands; a 2-D residual CNN then learns to follow the curved chirp track across both detectors' maps."""
    OUT = 7
    def __init__(self, widths=(16, 24, 32, 48, 64, 96), hidden=128):
        super().__init__()
        M, fc = band_matrix(); self.register_buffer('M', torch.from_numpy(M)); self.register_buffer('win', torch.hann_window(NFFT))
        self.register_buffer('pnorm', torch.tensor(1.0))
        w = widths
        self.stem = nn.Sequential(nn.Conv2d(2, w[0], (3, 7), stride=(1, 2), padding=(1, 3), bias=False), nn.BatchNorm2d(w[0]), nn.GELU())
        self.blocks = nn.Sequential(*[Block2(w[i], w[i + 1], 2) for i in range(len(w) - 1)])
        c = w[-1]
        self.head = nn.Sequential(nn.Linear(2 * c, hidden), nn.GELU(), nn.Linear(hidden, self.OUT))
    def tf(self, x):
        B = x.shape[0]
        S = torch.stft(x.reshape(2 * B, -1), NFFT, HOP, NFFT, self.win, center=True, return_complex=True)   # (2B, 193, 513)
        P = (S.real ** 2 + S.imag ** 2) / self.pnorm
        P = torch.einsum("kf,bft->bkt", self.M, P)[:, :, :NFR]
        return torch.log1p(P).reshape(B, 2, NBAND, NFR)
    def forward(self, x):
        h = self.blocks(self.stem(self.tf(x)))
        return self.head(torch.cat([h.mean((-1, -2)), h.amax((-1, -2))], 1))


class BNSSampler:
    def __init__(self, bank, par, sky, seed=0):
        self.bank, self.par, self.sky = bank, par, sky
        self.rng = np.random.default_rng(seed); self.ar = np.arange(WINB)

    def signals(self, idx, sk, tm):
        B = len(idx); out = np.zeros((B, 2, WINB), np.float32); S = self.sky[sk]
        m = np.round(tm * FSB).astype(int)
        for d in range(2):
            delay = np.round(S[:, 4 + d] * FSB).astype(int)
            start = BN.PEAKB - (m + delay)
            ii = start[:, None] + self.ar[None, :]
            ok = (ii >= 0) & (ii < BN.SEGB); ii = np.clip(ii, 0, BN.SEGB - 1)
            wp = np.take_along_axis(self.bank[idx, d, 0], ii, 1).astype(np.float32) * ok
            wc = np.take_along_axis(self.bank[idx, d, 1], ii, 1).astype(np.float32) * ok
            out[:, d] = S[:, 2 * d, None] * wp + S[:, 2 * d + 1, None] * wc
        return out

    def glitch(self, B):
        rng = self.rng; t = np.arange(WINB) / FSB
        f0 = np.exp(rng.uniform(np.log(30), np.log(400), B)); Q = rng.uniform(2, 20, B)
        t0 = rng.uniform(0.5, 15.9, B); tau = Q / (2 * np.pi * f0); ph = rng.uniform(0, 2 * np.pi, B)
        g = np.exp(-((t[None] - t0[:, None]) / tau[:, None]) ** 2) * np.cos(2 * np.pi * f0[:, None] * (t[None] - t0[:, None]) + ph[:, None])
        snr = np.exp(rng.uniform(np.log(6), np.log(60), B))
        return (g * (snr / np.sqrt(BN.PHIB * np.sum(g ** 2, 1) + 1e-12))[:, None]).astype(np.float32)

    def single(self, n, lo, hi):
        rng = self.rng
        idx = rng.integers(len(self.par), size=n); sk = rng.integers(len(self.sky), size=n); tm = rng.uniform(T_LO, T_HI, n)
        s = self.signals(idx, sk, tm); det = rng.integers(2, size=n); s[np.arange(n), 1 - det] = 0
        rho = np.sqrt(BN.PHIB * np.sum(s ** 2, (1, 2)))
        return s * (rng.uniform(lo, hi, n) / np.maximum(rho, 1e-9))[:, None, None]

    def batch(self, B, snr_lo=8, snr_hi=30, p_sig=0.5, p_glitch=0.0, p_single=0.0, single_snr=(8, 25)):
        rng = self.rng
        x = BN.noise_b(rng, (B, 2, WINB)); y = (rng.random(B) < p_sig).astype(np.float32)
        tgt = np.zeros((B, 6), np.float32)
        sel = np.where(y > 0)[0]
        if len(sel):
            idx = rng.integers(len(self.par), size=len(sel)); sk = rng.integers(len(self.sky), size=len(sel)); tm = rng.uniform(T_LO, T_HI, len(sel))
            s = self.signals(idx, sk, tm)
            rho = np.sqrt(BN.PHIB * np.sum(s ** 2, (1, 2))); snr = rng.uniform(snr_lo, snr_hi, len(sel))
            s *= (snr / np.maximum(rho, 1e-9))[:, None, None]; x[sel] += s
            tgt[sel, 0] = np.log(self.par[idx, 0]); tgt[sel, 1] = self.par[idx, 1]; tgt[sel, 2] = tm; tgt[sel, 3] = snr
            tgt[sel, 4:6] = np.sqrt(BN.PHIB * np.sum(s ** 2, -1))
        if p_single > 0:
            neg = np.where(y == 0)[0]; k = neg[rng.random(len(neg)) < p_single / max(1e-9, 1 - p_sig)]
            if len(k): x[k] += self.single(len(k), *single_snr)
        if p_glitch > 0:
            g = np.where(rng.random(B) < p_glitch)[0]
            if len(g): x[g, rng.integers(2, size=len(g))] += self.glitch(len(g))
        return x, y, tgt


def load_bns(path):
    ck = torch.load(path, weights_only=False)
    net = BNSNetTF() if ck.get('arch') == 'bns_tf' else BNSNet(widths=tuple(ck['widths']))
    net.load_state_dict(ck['ema']); net.eval(); return net


@torch.no_grad()
def run(net, x, bs=128):
    return np.concatenate([net(torch.from_numpy(x[i:i + bs])).numpy() for i in range(0, len(x), bs)])


SNRS = [6, 7, 8, 9, 10, 12, 15, 20]

def evaluate(net, n_noise=6000, n_per=400, seed=777):
    bank = np.load(_P.WORK + '/bns_bank_val.npy'); par = np.load(_P.WORK + '/bns_par_val.npy'); sky = np.load(_P.WORK + '/sky_val.npy')
    S = BNSSampler(bank, par, sky, seed=seed)
    s0 = np.concatenate([run(net, S.batch(500, p_sig=0.0)[0])[:, 0] for _ in range(n_noise // 500)])
    thr = float(np.quantile(s0, 1 - 1e-3)); res = {'thr_fap1e-3': thr, 'eff': {}}
    O, T = [], []
    for snr in SNRS:
        x, y, t = S.batch(n_per, snr, snr, p_sig=1.0); o = run(net, x)
        res['eff'][snr] = float(np.mean(o[:, 0] > thr)); O.append(o); T.append(t)
    o = np.concatenate(O); t = np.concatenate(T); hi = t[:, 3] >= 10
    dl = o[hi, 1] - t[hi, 0]; sig = np.exp(np.clip(o[hi, 2], -7, 3))
    res['mc_relerr_med'] = float(np.median(np.abs(np.expm1(dl)))); res['mc_cov68'] = float(np.mean(np.abs(dl) < sig))
    res['q_abserr_med'] = float(np.median(np.abs(o[hi, 3] - t[hi, 1]))); res['t_abserr_ms'] = float(1000 * np.median(np.abs(o[hi, 5] - t[hi, 2])))
    S.rng = np.random.default_rng(seed + 1)
    xg, _, _ = S.batch(2000, p_sig=0.0, p_glitch=1.0); res['glitch_fa'] = float(np.mean(run(net, xg)[:, 0] > thr))
    xs = BN.noise_b(S.rng, (n_per, 2, WINB)) + S.single(n_per, 12, 12); res['single_fa12'] = float(np.mean(run(net, xs.astype(np.float32))[:, 0] > thr))
    res['sens'] = float(np.mean([res['eff'][s] for s in (6, 7, 8, 9, 10, 12)]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True); ap.add_argument('--init', default=None); ap.add_argument('--steps', type=int, default=2500)
    ap.add_argument('--bs', type=int, default=64); ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--warm', type=int, default=150)
    ap.add_argument('--snr', default='curriculum'); ap.add_argument('--glitch', type=float, default=0.0); ap.add_argument('--single', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--ema', type=float, default=0.998); ap.add_argument('--threads', type=int, default=2); ap.add_argument('--arch', default='cnn')
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    net = BNSNetTF() if a.arch == 'tf' else BNSNet()
    if a.arch == 'tf':      # noise power per STFT bin for unit-variance whitened noise, so maps read 1 in pure noise
        with torch.no_grad():
            z = torch.from_numpy(BN.noise_b(np.random.default_rng(5), (64, 2, WINB)))
            S = torch.stft(z.reshape(128, -1), NFFT, HOP, NFFT, net.win, center=True, return_complex=True); P = S.abs() ** 2
            k = (np.arange(NFFT // 2 + 1) * FSB / NFFT > 40) & (np.arange(NFFT // 2 + 1) * FSB / NFFT < 400)
            net.pnorm.fill_(float(P[:, torch.from_numpy(k), 20:-20].mean()))
    if a.init: net.load_state_dict(torch.load(a.init, weights_only=False)['ema'])
    else:
        with torch.no_grad():
            b = net.head[-1].bias; b.zero_(); b[1] = 0.27; b[3] = 0.8; b[5] = 11.9; b[2] = -2; b[4] = -1.5; b[6] = 0.5
    ema = copy.deepcopy(net).eval()
    S = BNSSampler(np.load(_P.WORK + '/bns_bank_train.npy'), np.load(_P.WORK + '/bns_par_train.npy'), np.load(_P.WORK + '/sky_train.npy'), seed=2000 + a.seed)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4); t0 = time.time(); log = []; acc = np.zeros(4)
    for step in range(1, a.steps + 1):
        frac = step / a.steps
        lr = a.lr * (step / a.warm if step < a.warm else 0.5 * (1 + math.cos(math.pi * (step - a.warm) / max(1, a.steps - a.warm))) * 0.97 + 0.03)
        for g in opt.param_groups: g['lr'] = lr
        if a.snr == 'curriculum': lo = 16 - 10 * min(1, frac / 0.6); hi = 40 - 18 * min(1, frac / 0.6)
        else: lo, hi = map(float, a.snr.split(','))
        x, y, tgt = S.batch(a.bs, lo, hi, p_glitch=a.glitch, p_single=a.single)
        x, y, tgt = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(tgt)
        net.train(); out = net(x)
        bce = F.binary_cross_entropy_with_logits(out[:, 0], y); pos = y > 0; o, t = out[pos], tgt[pos]
        reg = nll(o[:, 1], o[:, 2], t[:, 0]).mean() + 0.5 * nll(o[:, 3], o[:, 4], t[:, 1]).mean() + 0.5 * nll(o[:, 5], o[:, 6], t[:, 2]).mean()
        loss = bce + 0.1 * reg
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(a.ema).add_(pn, alpha=1 - a.ema)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        acc += (loss.item(), bce.item(), reg.item(), 1)
        if step % 100 == 0 or step == a.steps:
            rec = {'step': step, 'loss': acc[0] / acc[3], 'bce': acc[1] / acc[3], 'reg': acc[2] / acc[3], 'lr': lr, 'snr': [round(lo, 1), round(hi, 1)], 't': round(time.time() - t0)}
            log.append(rec); print(json.dumps(rec), flush=True); acc[:] = 0
    torch.save({'ema': ema.state_dict(), 'widths': (16, 24, 32, 48, 64, 96, 128), 'arch': 'bns_tf' if a.arch == 'tf' else 'bns', 'args': vars(a), 'log': log}, f'{_P.WORK}/ckpt/{a.name}.pt')
    print('saved', a.name, flush=True)
    r = evaluate(ema); r['name'] = a.name; print('EVAL', json.dumps(r), flush=True)
    json.dump(r, open(f'{_P.WORK}/ckpt/{a.name}_eval.json', 'w'))

if __name__ == '__main__':
    main()
