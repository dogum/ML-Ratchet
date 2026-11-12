"""Physical-amplitude waveform bank for posterior estimation: whitened strain at 400 Mpc (not SNR-normalized),
plus which noise spectrum whitened it, so distance can be inferred."""
import _paths as _P  # portable paths (see _paths.py)
import sys, time, numpy as np
sys.path.insert(0, _P.SRC)
import gwphys as G
ASD_F = np.array([25, 35, 50, 70, 100, 140, 200, 280, 400, 560.])

def asd_features(fp, pp):
    return 0.5 * np.log10(np.interp(ASD_F, fp, pp)) + 23.0     # log10 ASD, roughly O(1)

def make(n, seed, psds):
    import lal, lalsimulation as ls
    rng = np.random.default_rng(seed)
    T, fs_in = 16, G.FS_IN; n_in = T * fs_in; df = 1.0 / T
    f = np.fft.rfftfreq(n_in, 1 / fs_in)
    S = {d: [np.interp(f, fp, pp) for fp, pp in psds[d]] for d in psds}
    bank = np.zeros((n, 2, 2, G.SEG), np.float16); par = np.zeros((n, 7), np.float32); pid = np.zeros((n, 2), np.int8)
    i = 0
    while i < n:
        mc = np.exp(rng.uniform(np.log(5.5), np.log(45.0))); q = rng.uniform(0.2, 1.0)
        m1, m2 = G.masses_from_mc_q(mc, q)
        if m1 > 110 or m2 < 4.0: continue
        c1, c2 = rng.uniform(-0.85, 0.85, 2); ci = rng.uniform(-1, 1)
        hp, hc = ls.SimInspiralChooseFDWaveform(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, c1, 0, 0, c2, 400e6 * lal.PC_SI, float(np.arccos(ci)), 0.0, 0, 0, 0, df, 20.0, fs_in / 2, 20.0, None, ls.IMRPhenomD)
        Hp = hp.data.data[: len(f)] * fs_in; Hc = hc.data.data[: len(f)] * fs_in
        sh = np.exp(-2j * np.pi * f * (T / 2))
        for d, det in enumerate(('H1', 'L1')):
            k = rng.integers(len(S[det])); pid[i, d] = k
            wp = G.whiten_dft(Hp * sh, n_in, fs_in, S[det][k]); wc = G.whiten_dft(Hc * sh, n_in, fs_in, S[det][k])
            if d == 0: pk = int(np.argmax(wp ** 2 + wc ** 2))
            bank[i, d, 0] = wp[pk - G.PEAK: pk - G.PEAK + G.SEG]; bank[i, d, 1] = wc[pk - G.PEAK: pk - G.PEAK + G.SEG]
        par[i] = (mc, q, m1, m2, c1, c2, ci); i += 1
    return bank, par, pid

if __name__ == '__main__':
    t = time.time(); psds = G.training_psds()
    feats = {d: np.stack([asd_features(fp, pp) for fp, pp in psds[d]]) for d in psds}
    np.save(_P.WORK + '/asd_feats.npy', np.stack([feats['H1'], feats['L1']]))
    b, p, k = make(8000, 31, psds); np.save(_P.WORK + '/pbank_train.npy', b); np.save(_P.WORK + '/ppar_train.npy', p); np.save(_P.WORK + '/ppid_train.npy', k)
    b, p, k = make(1500, 32, psds); np.save(_P.WORK + '/pbank_val.npy', b); np.save(_P.WORK + '/ppar_val.npy', p); np.save(_P.WORK + '/ppid_val.npy', k)
    print('done %.0fs' % (time.time() - t), flush=True)
