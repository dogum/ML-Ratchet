"""Injections with physical amplitudes for posterior estimation.
Priors: chirp mass log-uniform 5.5-45 (detector frame), q uniform 0.2-1, aligned spins uniform +-0.85,
isotropic sky / inclination / polarization, distance uniform in volume out to the optimal horizon of each
waveform; only events with network SNR >= 6 are kept (we only ever ask about detected events)."""
import numpy as np
import gwphys as G
from sampler import Sampler

class PESampler(Sampler):
    def __init__(self, bank, par, pid, sky, asd, seed=0):
        super().__init__(bank, par, sky, seed)
        self.pid = pid; self.asd = asd                      # asd: (2 detectors, nPSD, 10 features)
        b = bank.astype(np.float32)
        self.rho_opt = np.sqrt(G.PHI * np.sum(b[:, :, 0] ** 2 + b[:, :, 1] ** 2, axis=(1, 2)) / 2 * 2)   # ~ best-case network SNR at 400 Mpc
        m1, m2, c1, c2 = par[:, 2], par[:, 3], par[:, 4], par[:, 5]
        self.chi = (m1 * c1 + m2 * c2) / (m1 + m2)

    def batch(self, B, snr_min=6.0, psd_jitter=(0.6, 1.6)):
        rng = self.rng
        X, TH, AS = [], [], []
        need = B
        while need > 0:
            n = need * 5
            idx = rng.integers(len(self.par), size=n); sk = rng.integers(len(self.sky), size=n); tm = rng.uniform(0.75, 1.85, n)
            s = self.signals(idx, sk, tm)                    # physical, at 400 Mpc, whitened with PSD pid
            a = np.exp(rng.uniform(np.log(psd_jitter[0]), np.log(psd_jitter[1]), (n, 2))).astype(np.float32)
            s /= np.sqrt(a)[:, :, None]
            dh = 400.0 * self.rho_opt[idx] / snr_min
            d = np.maximum(dh * rng.random(n) ** (1 / 3), 10.0)
            s *= (400.0 / d)[:, None, None]
            rho = np.sqrt(G.PHI * np.sum(s ** 2, (1, 2)))
            ok = np.where(rho >= snr_min)[0][:need]
            if len(ok) == 0: continue
            asd = np.concatenate([self.asd[0, self.pid[idx[ok], 0]] + 0.5 * np.log10(a[ok, 0])[:, None],
                                  self.asd[1, self.pid[idx[ok], 1]] + 0.5 * np.log10(a[ok, 1])[:, None]], 1)
            th = np.stack([np.log(self.par[idx[ok], 0]), self.par[idx[ok], 1], self.chi[idx[ok]], np.log10(d[ok]), self.par[idx[ok], 6], tm[ok]], 1)
            X.append(s[ok]); TH.append(th); AS.append(asd); need -= len(ok)
        s = np.concatenate(X)[:B]; th = np.concatenate(TH)[:B].astype(np.float32); asd = np.concatenate(AS)[:B].astype(np.float32)
        x = G.bandlimited_noise(rng, (B, 2, G.WIN)) + s
        snr = np.sqrt(G.PHI * np.sum(s ** 2, (1, 2)))
        return x.astype(np.float32), th, asd, snr.astype(np.float32)
