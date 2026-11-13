"""
ChirpNet: find binary black-hole mergers in LIGO strain and weigh them.

A small 1-D convolutional network trained only on simulated mergers (IMRPhenomD waveforms from general
relativity, projected onto the Hanford and Livingston detectors for random sky positions) in simulated,
whitened detector noise. It takes 2 seconds of whitened strain from both detectors and returns a detection
statistic plus the chirp mass, mass ratio and merger time, each with an uncertainty.

    import chirpnet
    net = chirpnet.ChirpNet('chirpnet_weights.npz')
    h1, t0 = chirpnet.load_gwosc('H-H1_LOSC_4_V2-1126259446-32.hdf5')
    l1, _  = chirpnet.load_gwosc('L-L1_LOSC_4_V2-1126259446-32.hdf5')
    for hit in net.scan(h1, l1, gps_start=t0):
        print(hit)

Dependencies: numpy, scipy (PSD estimate), h5py (only to read GWOSC files).
Masses are detector-frame (redshifted) solar masses, like the "detector frame" values in LIGO catalogs.
"""
import os, json
import numpy as np

FS_IN, FS, WIN = 4096, 2048, 4096
BAND = (20.0, 900.0)
G_MSUN_C3 = 4.925490947e-6   # G * Msun / c^3 in seconds

# ------------------------------------------------------------------ whitening (identical to training)
def _band_mask(f, lo=BAND[0], hi=BAND[1], taper=8.0):
    m = np.zeros_like(f)
    m[(f >= lo) & (f <= hi)] = 1.0
    a = (f > lo - taper) & (f < lo); m[a] = 0.5 - 0.5 * np.cos(np.pi * (f[a] - (lo - taper)) / taper)
    b = (f > hi) & (f < hi + taper); m[b] = 0.5 + 0.5 * np.cos(np.pi * (f[b] - hi) / taper)
    return m

def _norm_const():
    n = 2 ** 16; f = np.fft.rfftfreq(n, 1 / FS_IN)
    m = _band_mask(f)[: (n * FS // FS_IN) // 2 + 1]
    n_out = n * FS // FS_IN
    return np.sqrt((np.sum(m[1:-1] ** 2) * 2 + m[0] ** 2 + m[-1] ** 2) * n / n_out ** 2 * (n_out / n) ** 2)
_NORM = _norm_const()

def psd(x, fs=FS_IN, seg=4.0):
    from scipy.signal import welch
    return welch(x, fs=fs, nperseg=int(seg * fs), noverlap=int(seg * fs / 2), window='hann', average='median')

def whiten(strain, fs=FS_IN, psd_fp=None):
    """Raw strain (any length >= 8 s, sampled at 4096 Hz) -> whitened, 20-900 Hz, unit-variance series at 2048 Hz."""
    from scipy.signal.windows import tukey
    if fs != FS_IN:
        raise ValueError('ChirpNet expects 4096 Hz strain (GWOSC 4 kHz files).')
    x = np.asarray(strain, np.float64); n = len(x) - (len(x) % 2)
    x = x[:n]
    fp, pp = psd_fp if psd_fp is not None else psd(x, fs)
    f = np.fft.rfftfreq(n, 1 / fs)
    X = np.fft.rfft(x * tukey(n, alpha=1.0 / 8))
    W = X / np.sqrt(np.interp(f, fp, pp) * fs / 2.0) * _band_mask(f)
    n_out = n * FS // fs
    y = np.fft.irfft(W[: n_out // 2 + 1], n=n_out) * (n_out / n)
    return (y / _NORM).astype(np.float32)

def gate(w, thresh=25.0, half=0.25, taper=0.1, fs=FS):
    """Auto-gating, as in LIGO's own searches: zero the whitened data around any excursion beyond `thresh` sigma.
    Real mergers at detectable distances never come close to this; loud instrumental glitches do."""
    w = np.array(w, np.float32, copy=True)
    bad = np.where(np.abs(w) > thresh)[0]
    if len(bad) == 0:
        return w, []
    gates, start, prev = [], bad[0], bad[0]
    for i in bad[1:]:
        if i - prev > int(half * fs): gates.append((start, prev)); start = i
        prev = i
    gates.append((start, prev))
    n, h, tp = len(w), int(half * fs), int(taper * fs)
    mask = np.ones(n, np.float32)
    for a, b in gates:
        lo, hi = max(0, a - h), min(n, b + h)
        mask[lo:hi] = 0
        if tp:
            # Tukey edges: fall from 1 to 0 approaching the gate, rise from 0 to 1 leaving it
            # (as in training/bns_real.py, which produced the published exam numbers; fixed 12 Nov 2026 —
            # the first release had both ramps reversed)
            down = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, tp))
            l0 = max(0, lo - tp); mask[l0:lo] = np.minimum(mask[l0:lo], down[tp - (lo - l0):])
            h1 = min(n, hi + tp); mask[hi:h1] = np.minimum(mask[hi:h1], down[::-1][: h1 - hi])
    return w * mask, [(a / fs, b / fs) for a, b in gates]

def load_gwosc(path):
    """Read a GWOSC HDF5 strain file -> (strain, gps_start)."""
    import h5py
    with h5py.File(path, 'r') as h:
        return h['strain/Strain'][:], float(h['meta/GPSstart'][()])

# ------------------------------------------------------------------ network
def _erf(x):
    s = np.sign(x); x = np.abs(x); t = 1 / (1 + 0.3275911 * x)
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x))

def _gelu(x):
    return 0.5 * x * (1 + _erf(x * 0.7071067811865476))

def _conv(x, w, b, stride, pad):
    """x: (B, Cin, L), w: (Cout, Cin, K) -> (B, Cout, Lout)"""
    B, C, L = x.shape; K = w.shape[2]
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    win = np.lib.stride_tricks.sliding_window_view(xp, K, axis=2)[:, :, ::stride, :]   # (B, C, Lout, K)
    y = np.einsum('bclk,ock->bol', win, w, optimize=True)
    return y + b[None, :, None]

def _chirp_track(mc, t_c, t):
    """Leading-order frequency of the dominant (2,2) mode at times t before coalescence t_c."""
    tau = np.maximum(t_c - t, 1e-4)
    return (1 / np.pi) * (5 / (256 * tau)) ** 0.375 * (mc * G_MSUN_C3) ** -0.625

class ChirpNet:
    def __init__(self, path='chirpnet_weights.npz', checker=None):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z['meta']))
        self.layers = [(z[f'w{i}'], z[f'b{i}'], s, p) for i, (s, p) in enumerate(self.meta['conv'])]
        self.lin = [(z['lw0'], z['lb0']), (z['lw1'], z['lb1'])]
        self.red = (z['rw'], z['rb']) if 'rw' in z.files else None
        self.arch = self.meta.get('arch', 'cnn')
        if self.arch == '2s':
            self.mix = (z['mixw'], z['mixb']); self.post = [(z[k + 'w'], z[k + 'b']) for k in ('p1', 'p2', 'ps')]; self.aux = (z['auxw'], z['auxb'])
        self.bg = z['background']             # ranking statistic of time-slid real LIGO noise (no true signals)
        self.bg_live_s = self.meta['background_livetime_s']
        # v2: a coincidence checker (two-stream network) confirms candidates; windows it vetoes are pushed down
        self.checker = None
        if self.meta.get('checker') and checker is not False:
            cp = checker or os.path.join(os.path.dirname(os.path.abspath(path)), self.meta['checker'])
            self.checker = ChirpNet(cp, checker=False); self.veto = float(self.meta['veto']); self.drop = float(self.meta.get('drop', 30.0))

    def _trunk(self, x):
        L = self.layers
        h = _gelu(_conv(x, *L[0])); i = 1
        while i < len(L):
            (w1, b1, s1, p1), (w2, b2, s2, p2), (ws, bs, ss, ps) = L[i], L[i + 1], L[i + 2]
            y = _gelu(_conv(h, w1, b1, s1, p1)); y = _conv(y, w2, b2, s2, p2)
            h = _gelu(y + _conv(h, ws, bs, ss, ps)); i += 3
        return h

    def forward(self, x):
        """x: (B, 2, 4096) whitened strain [H1, L1] -> raw outputs (B, 7)"""
        if self.arch == '2s':
            x = np.asarray(x, np.float32); B = len(x)
            e = self._trunk(np.ascontiguousarray(x.transpose(1, 0, 2)).reshape(2 * B, 1, x.shape[-1])); h, l = e[:B], e[B:]
            zz = _gelu(_conv(np.concatenate([h, l, h * l, np.abs(h - l)], 1), self.mix[0], self.mix[1], 1, 0))
            (a1, c1), (a2, c2), (a3, c3) = self.post
            y = _gelu(_conv(zz, a1, c1, 1, a1.shape[2] // 2)); y = _conv(y, a2, c2, 1, a2.shape[2] // 2)
            zz = _gelu(y + _conv(zz, a3, c3, 1, 0))
            g = np.concatenate([zz.mean(-1), zz.max(-1), h.max(-1), l.max(-1)], 1)
            g = _gelu(g @ self.lin[0][0].T + self.lin[0][1])
            return g @ self.lin[1][0].T + self.lin[1][1]
        L = self.layers
        h = _gelu(_conv(x, *L[0]))
        i = 1
        while i < len(L):
            (w1, b1, s1, p1), (w2, b2, s2, p2), (ws, bs, ss, ps) = L[i], L[i + 1], L[i + 2]
            y = _gelu(_conv(h, w1, b1, s1, p1))
            y = _conv(y, w2, b2, s2, p2)
            h = _gelu(y + _conv(h, ws, bs, ss, ps)); i += 3
        f = [h.mean(-1), h.max(-1)]
        if self.red is not None:
            f.append(_gelu(_conv(h, self.red[0], self.red[1], 1, 0)).reshape(len(h), -1))
        h = np.concatenate(f, 1)
        h = _gelu(h @ self.lin[0][0].T + self.lin[0][1])
        return h @ self.lin[1][0].T + self.lin[1][1]

    def statistic(self, x, floor=0.0):
        """Raw outputs with the ranking statistic (column 0) after the coincidence check, plus the checker's score.
        The checker only runs where the finder scores above `floor`; below it the statistic is the finder's own score
        (the cascade can only lower a score, so every value above `floor` is exact)."""
        x = np.asarray(x, np.float32); o = self.forward(x); c = np.full(len(o), np.nan, np.float32)
        if self.checker is not None:
            k = np.where(o[:, 0] > floor)[0]
            if len(k):
                c[k] = self.checker.forward(x[k])[:, 0]
                o[k, 0] -= self.drop * (c[k] < self.veto)
        return o, c

    def predict(self, x):
        """Whitened windows (B, 2, 4096) -> dict of arrays (score includes the coincidence check)."""
        o, c = self.statistic(np.asarray(x, np.float32))
        s_l = np.exp(np.clip(o[:, 2], -7, 3)); s_q = np.exp(np.clip(o[:, 4], -7, 3))
        return {'score': o[:, 0], 'chirp_mass': np.exp(o[:, 1]), 'chirp_mass_lo': np.exp(o[:, 1] - s_l), 'chirp_mass_hi': np.exp(o[:, 1] + s_l),
                'q': np.clip(o[:, 3], 0.05, 1.0), 'q_sigma': s_q, 't_merger': o[:, 5], 'coincidence': c}

    def far_per_hour(self, score):
        """False-alarm rate estimated from the time-slid real-noise background shipped with the weights."""
        n = int(np.sum(self.bg >= score))
        return (n + 1) / (self.bg_live_s / 3600.0), n == 0

    def _alone(self, H, L, t, slides=(5, 9, 13)):
        """Diagnostic: the statistic each detector earns on its own (partner slid by whole seconds).
        A coincident astrophysical merger usually has both views high, or a joint score well above both.
        A chirp-like artifact in one detector has one view high and the other low."""
        def best(A, B):
            st = np.arange(max(0, int((t - 1.9) * FS)), min(len(A) - WIN, int((t - 0.6) * FS)), 64)
            if len(st) == 0: return float('nan')
            x = np.stack([np.stack([A[s:s + WIN], B[s:s + WIN]]) for s in st]).astype(np.float32)
            return float(self.statistic(x)[0][:, 0].max())
        return (max(best(H, np.roll(L, k * FS)) for k in slides), max(best(np.roll(H, k * FS), L) for k in slides))

    def scan(self, h1, l1, gps_start=0.0, fs=FS_IN, stride=0.0625, threshold=None, whitened=False):
        """Slide the network over two detectors' strain. Returns candidate mergers, loudest first.
        Each hit carries the coincidence checker's score and h1_alone / l1_alone (see _alone)."""
        H = h1 if whitened else whiten(h1, fs); Lw = l1 if whitened else whiten(l1, fs)
        H, gH = gate(H); Lw, gL = gate(Lw)
        self.last_gates = {'H1': [(gps_start + a, gps_start + b) for a, b in gH], 'L1': [(gps_start + a, gps_start + b) for a, b in gL]}
        n = min(len(H), len(Lw)); edge = 2 * FS
        starts = np.arange(edge, n - edge - WIN, int(stride * FS))
        out, cc = [], []
        for i in range(0, len(starts), 64):
            st = starts[i:i + 64]
            x = np.stack([np.stack([H[s:s + WIN], Lw[s:s + WIN]]) for s in st])
            o_, c_ = self.statistic(x); out.append(o_); cc.append(c_)
        o = np.concatenate(out); cchk = np.concatenate(cc)
        thr = self.meta['threshold'] if threshold is None else threshold
        t_m = starts / FS + o[:, 5]
        order = np.argsort(-o[:, 0]); used = np.zeros(len(o), bool); hits = []
        for k in order:
            if o[k, 0] < thr: break
            if used[k]: continue
            grp = (np.abs(t_m - t_m[k]) < 0.5) & (o[:, 0] > o[k, 0] - 2.0)
            used |= np.abs(t_m - t_m[k]) < 1.0
            w = np.exp(-2 * np.clip(o[grp, 2], -7, 3))
            lmc = np.sum(w * o[grp, 1]) / np.sum(w); sl = np.sqrt(1 / np.mean(w))
            q = float(np.clip(np.mean(o[grp, 3]), 0.05, 1.0)); mc = float(np.exp(lmc))
            eta = q / (1 + q) ** 2; M = mc / eta ** 0.6
            far, bound = self.far_per_hour(o[k, 0])
            wt = np.exp(-2 * np.clip(o[grp, 6], -7, 3)); t_hit = float(np.sum(wt * t_m[grp]) / np.sum(wt))
            hv = self._alone(H, Lw, t_m[k])
            hits.append({'gps': float(gps_start + t_hit), 'score': float(o[k, 0]),
                         'coincidence': None if self.checker is None else float(cchk[k]),
                         'h1_alone': hv[0], 'l1_alone': hv[1],
                         'far_per_hour': far, 'far_is_upper_bound': bound,
                         'chirp_mass': mc, 'chirp_mass_68': [float(np.exp(lmc - sl)), float(np.exp(lmc + sl))],
                         'mass_ratio': q, 'm1': float(M / (1 + q)), 'm2': float(M * q / (1 + q)), 'total_mass': float(M)})
        return hits

def chirp_track(chirp_mass, t_coalescence, times):
    """Frequency (Hz) of the gravitational-wave chirp for a given chirp mass, leading post-Newtonian order."""
    return _chirp_track(chirp_mass, t_coalescence, np.asarray(times))


# ================================================================== ChirpNet-PE: full posteriors
_ASD_F = np.array([25, 35, 50, 70, 100, 140, 200, 280, 400, 560.])

def asd_features(strain, fs=FS_IN):
    """log10 amplitude spectral density at ten frequencies (+23), the noise summary ChirpNet-PE conditions on."""
    fp, pp = psd(np.asarray(strain, np.float64), fs)
    return (0.5 * np.log10(np.interp(_ASD_F, fp, pp)) + 23.0).astype(np.float32)

def _dl_to_z():
    """Planck 2015 flat LambdaCDM (H0 = 67.74, Om = 0.3075): luminosity distance (Mpc) -> redshift table."""
    z = np.linspace(0, 2.5, 5001); H0, Om = 67.74, 0.3075; c = 299792.458
    Ez = np.sqrt(Om * (1 + z) ** 3 + (1 - Om))
    dc = np.concatenate([[0], np.cumsum(0.5 * (1 / Ez[1:] + 1 / Ez[:-1]) * np.diff(z))]) * c / H0
    return (1 + z) * dc, z
_DL, _Z = _dl_to_z()

class ChirpPE:
    """Neural posterior estimation (in the spirit of DINGO): a CNN embedding of 2 s of two-detector strain plus each
    detector's noise spectrum conditions a RealNVP normalizing flow. Sampling it gives the posterior over
    detector-frame chirp mass, mass ratio, effective spin, luminosity distance, inclination and merger time."""
    def __init__(self, path='chirpnet_pe_weights.npz'):
        z = np.load(path, allow_pickle=False); self.meta = json.loads(str(z['meta']))
        self.layers = [(z[f'w{i}'], z[f'b{i}'], s, p) for i, (s, p) in enumerate(self.meta['conv'])]
        self.ctx = [(z['cw0'], z['cb0']), (z['cw1'], z['cb1'])]
        self.flow = [(z[f'm{i}'], [(z[f'f{i}w{j}'], z[f'f{i}b{j}']) for j in range(3)]) for i in range(self.meta['n_layers'])]
        self.mu = np.array(self.meta['mu'], np.float32); self.sd = np.array(self.meta['sd'], np.float32)

    def embed(self, x):
        L = self.layers; h = _gelu(_conv(x, *L[0])); i = 1
        while i < len(L):
            (w1, b1, s1, p1), (w2, b2, s2, p2), (ws, bs, ss, ps) = L[i], L[i + 1], L[i + 2]
            y = _gelu(_conv(h, w1, b1, s1, p1)); y = _conv(y, w2, b2, s2, p2)
            h = _gelu(y + _conv(h, ws, bs, ss, ps)); i += 3
        return np.concatenate([h.mean(-1), h.max(-1)], 1)

    def sample(self, window, asd, n=5000, z=None, seed=0):
        """window: (2, 4096) whitened [H1, L1]; asd: (20,) features [H1 ten, L1 ten] -> (n, 6) samples."""
        f = np.concatenate([self.embed(np.asarray(window, np.float32)[None]), np.asarray(asd, np.float32)[None]], 1)
        c = _gelu(f @ self.ctx[0][0].T + self.ctx[0][1]) @ self.ctx[1][0].T + self.ctx[1][1]
        z = np.random.default_rng(seed).standard_normal((n, 6)).astype(np.float32) if z is None else np.asarray(z, np.float32)
        C = np.repeat(c, len(z), 0)
        for m, net in reversed(self.flow):
            hdn = np.concatenate([z * m, C], 1)
            hdn = _gelu(hdn @ net[0][0].T + net[0][1]); hdn = _gelu(hdn @ net[1][0].T + net[1][1]); o = hdn @ net[2][0].T + net[2][1]
            s, t = o[:, :6], o[:, 6:]
            s = 2.5 * np.tanh(s / 2.5) * (1 - m); t = t * (1 - m)
            z = z * m + (1 - m) * ((z - t) * np.exp(-s))
        return z * self.sd + self.mu

    def weigh(self, h1, l1, gps_start, gps_merger, n=20000, fs=FS_IN, seed=0, place=1.35):
        """Full posterior for a merger at gps_merger (e.g. a ChirpNet scan hit). Returns samples and 90% summaries,
        including source-frame masses via the Planck 2015 cosmology."""
        H = gate(whiten(h1, fs))[0]; L = gate(whiten(l1, fs))[0]
        s0 = int(round((gps_merger - gps_start - place) * FS))
        win = np.stack([H[s0:s0 + WIN], L[s0:s0 + WIN]])
        asd = np.concatenate([asd_features(h1, fs), asd_features(l1, fs)])
        S = self.sample(win, asd, n, seed=seed)
        mc = np.exp(S[:, 0]); q = np.clip(S[:, 1], 0.05, 1.0); dl = 10 ** S[:, 3]; zz = np.interp(dl, _DL, _Z)
        eta = q / (1 + q) ** 2; M = mc / eta ** 0.6
        out = {'chirp_mass_det': mc, 'mass_ratio': q, 'chi_eff': np.clip(S[:, 2], -1, 1), 'luminosity_distance_mpc': dl,
               'cos_inclination': np.clip(S[:, 4], -1, 1), 'redshift': zz, 'm1_source': M / (1 + q) / (1 + zz), 'm2_source': M * q / (1 + q) / (1 + zz),
               'chirp_mass_source': mc / (1 + zz), 't_merger_gps': gps_start + s0 / FS + S[:, 5]}
        summ = {k: [float(np.quantile(v, 0.05)), float(np.median(v)), float(np.quantile(v, 0.95))] for k, v in out.items() if k != 't_merger_gps'}
        return {'samples': out, 'summary_90': summ}


# ================================================================== neutron stars (v2)
# A separate detector for binary neutron stars: 16 s windows at 1024 Hz, 25-500 Hz, read as a time-frequency map.
FSB, WINB, BANDB = 1024, 16384, (25.0, 500.0)

def _norm_b():
    n = 2 ** 18; f = np.fft.rfftfreq(n, 1 / 4096); m = _band_mask(f, *BANDB, taper=6.0)[: (n * FSB // 4096) // 2 + 1]; n_out = n * FSB // 4096
    return np.sqrt((np.sum(m[1:-1] ** 2) * 2 + m[0] ** 2 + m[-1] ** 2) * n / n_out ** 2 * (n_out / n) ** 2)
_NORMB = _norm_b()

def whiten_bns(strain, fs=FS_IN, psd_fp=None):
    """Raw 4096 Hz strain (ideally minutes of it) -> whitened, 25-500 Hz, unit-variance series at 1024 Hz."""
    from scipy.signal.windows import tukey
    x = np.asarray(strain, np.float64); n = len(x) - len(x) % 2; x = x[:n]
    fp, pp = psd_fp if psd_fp is not None else psd(x, fs, seg=8.0)
    f = np.fft.rfftfreq(n, 1 / fs); X = np.fft.rfft(x * tukey(n, alpha=1.0 / 16))
    W = X / np.sqrt(np.interp(f, fp, pp) * fs / 2.0) * _band_mask(f, *BANDB, taper=6.0)
    n_out = n * FSB // fs
    return (np.fft.irfft(W[: n_out // 2 + 1], n=n_out) * (n_out / n) / _NORMB).astype(np.float32)

def _conv2d(x, w, b, stride, pad):
    """x: (B, C, H, W), w: (O, C, kh, kw) -> (B, O, H', W')"""
    kh, kw = w.shape[2:]; sh, sw = stride; ph, pw = pad
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(2, 3))[:, :, ::sh, ::sw]   # (B, C, H', W', kh, kw)
    return np.einsum('bchwij,ocij->bohw', win, w, optimize=True) + b[None, :, None, None]

class ChirpNetBNS:
    """Binary-neutron-star detector and chirp-mass estimator. Input: whitened (B, 2, 16384) windows [H1, L1] at 1024 Hz."""
    def __init__(self, path='chirpnet_bns_weights.npz'):
        z = np.load(path, allow_pickle=False); self.meta = json.loads(str(z['meta']))
        self.layers = [(z[f'w{i}'], z[f'b{i}'], tuple(s), tuple(p)) for i, (s, p) in enumerate(self.meta['conv'])]
        self.lin = [(z['lw0'], z['lb0']), (z['lw1'], z['lb1'])]; self.M = z['band_matrix']; self.pnorm = float(self.meta['pnorm'])
        self.nfft, self.hop, self.nfr = self.meta['nfft'], self.meta['hop'], self.meta['frames']
        self.bg = z['background']; self.bg_live_s = self.meta['background_livetime_s']
        from scipy.signal import get_window
        self.win = get_window('hann', self.nfft).astype(np.float64)       # periodic Hann, as torch.hann_window

    def tf_map(self, x):
        """(B, 2, 16384) -> (B, 2, 64, 256): log(1 + power), 0.375 s frames every 1/16 s, 64 log-spaced bands 24-450 Hz"""
        B = x.shape[0]; h = self.nfft // 2
        xp = np.pad(np.asarray(x, np.float64).reshape(2 * B, -1), ((0, 0), (h, h)), mode='reflect')
        fr = np.lib.stride_tricks.sliding_window_view(xp, self.nfft, axis=1)[:, ::self.hop][:, :self.nfr]
        P = np.abs(np.fft.rfft(fr * self.win, axis=-1)) ** 2 / self.pnorm            # (2B, frames, bins)
        return np.log1p(np.einsum('kf,btf->bkt', self.M, P)).reshape(B, 2, self.M.shape[0], self.nfr).astype(np.float32)

    def forward(self, x):
        h = _gelu(_conv2d(self.tf_map(x), *self.layers[0])); i = 1; L = self.layers
        while i < len(L):
            y = _gelu(_conv2d(h, *L[i])); y = _conv2d(y, *L[i + 1]); h = _gelu(y + _conv2d(h, *L[i + 2])); i += 3
        g = np.concatenate([h.mean((-1, -2)), h.max((-1, -2))], 1)
        g = _gelu(g @ self.lin[0][0].T + self.lin[0][1]); return g @ self.lin[1][0].T + self.lin[1][1]

    def far_per_hour(self, score):
        n = int(np.sum(self.bg >= score)); return (n + 1) / (self.bg_live_s / 3600.0), n == 0

    def scan(self, h1, l1, gps_start=0.0, fs=FS_IN, stride=0.25, threshold=None, whitened=False, gate_sigma=25.0):
        """Slide 16 s windows over two detectors' strain (minutes of 4 kHz data are best: the PSD needs it).
        Returns hits with chirp mass (detector frame), merger time and a false-alarm rate from time-slid O2 noise."""
        H = h1 if whitened else whiten_bns(h1, fs); Lw = l1 if whitened else whiten_bns(l1, fs)
        H, gH = gate(H, gate_sigma, fs=FSB); Lw, gL = gate(Lw, gate_sigma, fs=FSB)
        self.last_gates = {'H1': [(gps_start + a, gps_start + b) for a, b in gH], 'L1': [(gps_start + a, gps_start + b) for a, b in gL]}
        n = min(len(H), len(Lw)); starts = np.arange(FSB, n - WINB - FSB, int(stride * FSB)); out = []
        for i in range(0, len(starts), 16):
            st = starts[i:i + 16]; out.append(self.forward(np.stack([np.stack([H[s:s + WINB], Lw[s:s + WINB]]) for s in st])))
        o = np.concatenate(out); thr = self.meta['threshold'] if threshold is None else threshold
        t_m = starts / FSB + o[:, 5]; order = np.argsort(-o[:, 0]); used = np.zeros(len(o), bool); hits = []
        for k in order:
            if o[k, 0] < thr: break
            if used[k]: continue
            grp = (np.abs(t_m - t_m[k]) < 2.0) & (o[:, 0] > o[k, 0] - 3.0); used |= np.abs(t_m - t_m[k]) < 8.0
            w = np.exp(-2 * np.clip(o[grp, 2], -7, 3)); lmc = np.sum(w * o[grp, 1]) / np.sum(w); sl = np.sqrt(1 / np.mean(w))
            far, bound = self.far_per_hour(o[k, 0])
            hits.append({'gps': float(gps_start + np.median(t_m[grp])), 'score': float(o[k, 0]), 'far_per_hour': far, 'far_is_upper_bound': bound,
                         'chirp_mass': float(np.exp(lmc)), 'chirp_mass_68': [float(np.exp(lmc - sl)), float(np.exp(lmc + sl))], 'mass_ratio': float(np.clip(np.mean(o[grp, 3]), 0.05, 1))})
        return hits
