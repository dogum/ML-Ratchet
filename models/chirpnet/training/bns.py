"""Binary neutron stars: 16 s windows at 1024 Hz (band 25-500 Hz), IMRPhenomD_NRTidalv2 waveforms."""
import _paths as _P  # portable paths (see _paths.py)
import sys, time, numpy as np
sys.path.insert(0, _P.SRC)
import gwphys as G

FSB, WINB, BANDB = 1024, 16384, (25.0, 500.0)
SEGB_PRE, SEGB_POST = 17.0, 0.5
SEGB = int((SEGB_PRE + SEGB_POST) * FSB); PEAKB = int(SEGB_PRE * FSB)

def mask_b(f):
    return G.band_mask(f, lo=BANDB[0], hi=BANDB[1], taper=6.0)

def whiten_dft_b(X, n_in, fs_in, S):
    f = np.fft.rfftfreq(n_in, 1 / fs_in)
    W = X / np.sqrt(S * fs_in / 2.0) * mask_b(f)
    n_out = n_in * FSB // fs_in
    y = np.fft.irfft(W[: n_out // 2 + 1], n=n_out) * (n_out / n_in)
    return y

def _norm_b():
    n = 2 ** 18; f = np.fft.rfftfreq(n, 1 / 4096); m = mask_b(f)[: (n * FSB // 4096) // 2 + 1]; n_out = n * FSB // 4096
    return np.sqrt((np.sum(m[1:-1] ** 2) * 2 + m[0] ** 2 + m[-1] ** 2) * n / n_out ** 2 * (n_out / n) ** 2)
NORMB = _norm_b()
PHIB = float(np.mean(mask_b(np.fft.rfftfreq(WINB, 1 / FSB))[1:] ** 2))

def whiten_b(x, fs_in=4096, psd_fp=None):
    from scipy.signal.windows import tukey
    n = len(x) - len(x) % 2; x = np.asarray(x[:n], np.float64)
    fp, pp = psd_fp if psd_fp is not None else G.psd(x, fs_in)
    f = np.fft.rfftfreq(n, 1 / fs_in); X = np.fft.rfft(x * tukey(n, alpha=1.0 / 16))
    return (whiten_dft_b(X, n, fs_in, np.interp(f, fp, pp)) / NORMB).astype(np.float32)

def noise_b(rng, shape):
    z = rng.standard_normal(shape).astype(np.float32); Z = np.fft.rfft(z, axis=-1)
    m = mask_b(np.fft.rfftfreq(shape[-1], 1 / FSB)).astype(np.float32)
    return (np.fft.irfft(Z * m, n=shape[-1], axis=-1) / np.sqrt(np.mean(m[1:] ** 2))).astype(np.float32)

def make_bank(n, seed, psds):
    import lal, lalsimulation as ls
    rng = np.random.default_rng(seed); T = 128; fs_in = 4096; n_in = T * fs_in
    f = np.fft.rfftfreq(n_in, 1 / fs_in)
    S = {d: [np.interp(f, fp, pp) for fp, pp in psds[d]] for d in psds}
    bank = np.zeros((n, 2, 2, SEGB), np.float16); par = np.zeros((n, 5), np.float32)
    for i in range(n):
        while True:
            mc = rng.uniform(0.85, 2.0); q = rng.uniform(0.6, 1.0); m1, m2 = G.masses_from_mc_q(mc, q)
            if m2 >= 0.95 and m1 <= 2.6: break
        c1, c2 = rng.uniform(-0.05, 0.05, 2); ci = rng.uniform(-1, 1); lam = rng.uniform(0, 1500, 2)
        p = lal.CreateDict(); ls.SimInspiralWaveformParamsInsertTidalLambda1(p, lam[0]); ls.SimInspiralWaveformParamsInsertTidalLambda2(p, lam[1])
        hp, hc = ls.SimInspiralChooseFDWaveform(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, c1, 0, 0, c2, 100e6 * lal.PC_SI, float(np.arccos(ci)), 0.0, 0, 0, 0,
                                                1.0 / T, 23.0, fs_in / 2, 23.0, p, ls.IMRPhenomD_NRTidalv2)
        Hp = hp.data.data[: len(f)] * fs_in; Hc = hc.data.data[: len(f)] * fs_in; sh = np.exp(-2j * np.pi * f * (T / 2))
        for d, det in enumerate(('H1', 'L1')):
            Sd = S[det][rng.integers(len(S[det]))]
            wp = whiten_dft_b(Hp * sh, n_in, fs_in, Sd) / NORMB; wc = whiten_dft_b(Hc * sh, n_in, fs_in, Sd) / NORMB
            if d == 0: pk = int(np.argmax(wp ** 2 + wc ** 2))
            sp, sc = wp[pk - PEAKB: pk - PEAKB + SEGB], wc[pk - PEAKB: pk - PEAKB + SEGB]
            nrm = 1.0 / max(np.sqrt(PHIB * np.sum(sp ** 2 + sc ** 2) / 2), 1e-30)
            bank[i, d, 0] = sp * nrm; bank[i, d, 1] = sc * nrm
        par[i] = (mc, q, m1, m2, ci)
    return bank, par

if __name__ == '__main__':
    t = time.time(); psds = G.training_psds()
    b, p = make_bank(int(sys.argv[1]), 41, psds); np.save(_P.WORK + '/bns_bank_train.npy', b); np.save(_P.WORK + '/bns_par_train.npy', p)
    b, p = make_bank(int(sys.argv[2]), 42, psds); np.save(_P.WORK + '/bns_bank_val.npy', b); np.save(_P.WORK + '/bns_par_val.npy', p)
    print('done %.0fs' % (time.time() - t), flush=True)
