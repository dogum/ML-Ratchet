import numpy as np
import gwphys as G

PAD = 4400

class Sampler:
    """Draws two-detector training windows: band-limited white noise (what whitened LIGO noise looks like)
    plus, for positives, a GR waveform projected onto Hanford and Livingston for a random sky position,
    scaled to a target network SNR."""
    def __init__(self, bank, par, sky, seed=0):
        self.bank = np.pad(bank, ((0, 0), (0, 0), (0, 0), (PAD, PAD)))
        self.par, self.sky = par, sky
        self.rng = np.random.default_rng(seed)
        self.ar = np.arange(G.WIN)

    def signals(self, idx, sk, tm):
        """whitened unit-scale strain for bank rows idx, sky rows sk, merger at tm seconds into the window"""
        B = len(idx); out = np.zeros((B, 2, G.WIN), np.float32)
        S = self.sky[sk]
        m = np.round(tm * G.FS).astype(int)
        for d in range(2):
            delay = np.round(S[:, 4 + d] * G.FS).astype(int)
            start = PAD + G.PEAK - (m + delay)
            ii = start[:, None] + self.ar[None, :]
            wp = np.take_along_axis(self.bank[idx, d, 0], ii, 1).astype(np.float32)
            wc = np.take_along_axis(self.bank[idx, d, 1], ii, 1).astype(np.float32)
            out[:, d] = S[:, 2 * d, None] * wp + S[:, 2 * d + 1, None] * wc
        return out

    def glitch(self, B):
        """sine-Gaussian / blip transients like the ones that plague real detectors"""
        rng = self.rng; t = np.arange(G.WIN) / G.FS
        f0 = np.exp(rng.uniform(np.log(30), np.log(500), B)); Q = rng.uniform(2, 20, B)
        t0 = rng.uniform(0.1, 1.95, B); tau = Q / (2 * np.pi * f0)
        ph = rng.uniform(0, 2 * np.pi, B)
        g = np.exp(-((t[None] - t0[:, None]) / tau[:, None]) ** 2) * np.cos(2 * np.pi * f0[:, None] * (t[None] - t0[:, None]) + ph[:, None])
        snr = np.exp(rng.uniform(np.log(6), np.log(60), B))
        g *= (snr / np.sqrt(G.PHI * np.sum(g ** 2, 1) + 1e-12))[:, None]
        return g.astype(np.float32)

    def single(self, n, snr_lo=6, snr_hi=25):
        """a chirp in ONE detector only (the other is pure noise): physically implausible for Hanford+Livingston,
        so it is a negative. Teaches the network to demand coincidence."""
        rng = self.rng
        idx = rng.integers(len(self.par), size=n); sk = rng.integers(len(self.sky), size=n); tm = rng.uniform(0.6, 1.9, n)
        s = self.signals(idx, sk, tm); det = rng.integers(2, size=n)
        s[np.arange(n), 1 - det] = 0
        rho = np.sqrt(G.PHI * np.sum(s ** 2, (1, 2)))
        snr = rng.uniform(snr_lo, snr_hi, n)
        return s * (snr / np.maximum(rho, 1e-9))[:, None, None]

    def batch(self, B, snr_lo=6, snr_hi=30, p_sig=0.5, p_glitch=0.0, bank_rows=None, p_single=0.0, single_snr=(6, 25)):
        rng = self.rng
        x = G.bandlimited_noise(rng, (B, 2, G.WIN))
        y = (rng.random(B) < p_sig).astype(np.float32)
        tgt = np.zeros((B, 6), np.float32)          # logMc, q, t_merge, snr, snr_H1, snr_L1
        sel = np.where(y > 0)[0]
        if len(sel):
            nb = len(self.par)
            idx = rng.integers(nb, size=len(sel)) if bank_rows is None else rng.choice(bank_rows, size=len(sel))
            sk = rng.integers(len(self.sky), size=len(sel))
            tm = rng.uniform(0.6, 1.9, len(sel))
            s = self.signals(idx, sk, tm)
            rho = np.sqrt(G.PHI * np.sum(s ** 2, (1, 2)))
            snr = rng.uniform(snr_lo, snr_hi, len(sel))
            s = s * (snr / np.maximum(rho, 1e-9))[:, None, None]
            x[sel] += s
            tgt[sel, 4:6] = np.sqrt(G.PHI * np.sum(s ** 2, -1))
            tgt[sel, 0] = np.log(self.par[idx, 0]); tgt[sel, 1] = self.par[idx, 1]; tgt[sel, 2] = tm; tgt[sel, 3] = snr
        if p_single > 0:
            neg = np.where(y == 0)[0]
            k = neg[rng.random(len(neg)) < p_single / max(1e-9, 1 - p_sig)]
            if len(k):
                sg = self.single(len(k), *single_snr); x[k] += sg
                tgt[k, 4:6] = np.sqrt(G.PHI * np.sum(sg ** 2, -1))
        if p_glitch > 0:
            g = np.where(rng.random(B) < p_glitch)[0]
            if len(g):
                det = rng.integers(2, size=len(g))
                x[g, det] += self.glitch(len(g))
        return x, y, tgt
