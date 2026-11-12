"""BearingNet simulator: localized rolling-bearing defects injected into REAL healthy-machine vibration.
Physics (McFadden & Smith 1984; Randall & Antoni 2011, sec. 2; Sawalhi & Randall 2011 for entry/exit):
 * impacts at the kinematic defect frequency (outer race BPFO, inner race BPFI, rolling element 2*BSF), with cage slip
   (mean frequency 0-2% off theory) and 0.5-2% random jitter of every inter-arrival time;
 * amplitude modulation: inner-race defect rotates with the shaft through the Stribeck load zone (shaft-rate AM); a ball
   defect travels with the cage (cage-rate AM) and alternately strikes inner and outer race; outer race: fixed load;
 * each strike is an entry (soft, low-frequency) + exit (sharp) pulse pair Delta = defect length / rolling speed apart;
 * each pulse rings 1-3 randomized structural resonances (damped oscillators 0.7-5.6 kHz, damping 0.8-8%);
 * transmission path: random spectral tilt and attenuation (the SNR draw);
 * shaft: extra imbalance (1x) and misalignment (2x, 3x) lines;
 * background: a real healthy recording, time-scaled by rho (speed change), so its own shaft lines follow the new speed.
 * v2 (calibrated on dev): stronger faults (SNR), longer ringing (lower damping), 30% loud distractor trains (the other
   bearing), shaft-rate modulation of the healthy background (50%, orders 1-6), random sensor/path gain curve (70%),
   outer-race impacts modulated at shaft rate by a rotating (unbalance) load (70%, depth 0-0.8).
Simulation runs at 48 kHz and is decimated to 12 kHz."""
import numpy as np, scipy.signal as ss
from fractions import Fraction
import features as FE
FS = FE.FS; UP = 4; FSH = FS * UP; NW = FE.NWIN; CLS = ('N', 'OR', 'IR', 'B')

def stribeck(theta, eps):
    """load distribution factor of a rolling element at angle theta from the load-zone centre (Harris), n = 1.5"""
    a = 1 - (1 - np.cos(theta)) / (2 * eps); return np.where(a > 0, np.abs(a) ** 1.5, 0.0)

ZETA = {'v1': (0.008, 0.08), 'v2': (0.0033, 0.04)}     # v2: dev twin test, envelope ring time real 3.5 ms vs twin 2.3 ms (median, CWRU near)
SNR = {'v1': (-26, 0), 'v2': (-18, 26)}               # v2: dev faults add 2-25 dB band power (CWRU near 10-25, far 2-12; PB up to 8)

def resonances(rng, n=None, version='v1'):
    zl, zh = ZETA[version]
    K = n or rng.integers(1, 4); return [dict(f=float(np.exp(rng.uniform(np.log(700), np.log(5600)))), z=float(np.exp(rng.uniform(np.log(zl), np.log(zh)))),
                                               a=float(np.exp(rng.uniform(np.log(0.3), 0)))) for _ in range(K)]

def impulse_response(res, tilt=0.0):
    tmax = min(0.25, max(7.0 / (2 * np.pi * r['f'] * r['z']) for r in res)); t = np.arange(int(tmax * FSH)) / FSH
    h = sum(r['a'] * np.exp(-2 * np.pi * r['f'] * r['z'] * t) * np.sin(2 * np.pi * r['f'] * np.sqrt(1 - r['z'] ** 2) * t) for r in res)
    if tilt:
        H = np.fft.rfft(h, 2 * len(h)); f = np.fft.rfftfreq(2 * len(h), 1 / FSH); H *= (np.maximum(f, 50) / 2000.0) ** tilt; h = np.fft.irfft(H)[:len(h)]
    return h

def pulse(width):
    w = max(int(round(width)), 1); p = np.hanning(w + 2)[1:-1]; return p / p.sum()

def fault_train(rng, cls, f_r, geom_mm, size_mm, T, P):
    """impact times (s) and amplitudes for T seconds (plus pre-roll) of a defect of class cls at shaft speed f_r Hz"""
    o = FE.orders(geom_mm['n'], geom_mm['d'], geom_mm['D'], geom_mm.get('phi', 0)); s = P['slip']
    ftf = o['FTF'] * (1 - s); f_def = {'OR': geom_mm['n'] * ftf, 'IR': geom_mm['n'] * (1 - ftf), 'B': 2 * o['BSF'] * (1 - s)}[cls] * f_r
    t0 = -0.3; n = int((T - t0) * f_def * 1.1) + 4
    dt = (1 + P['jitter'] * rng.standard_normal(n)) / f_def; t = t0 + rng.uniform(0, 1 / f_def) + np.concatenate([[0], np.cumsum(dt[:-1])]); t = t[t < T]
    A = np.exp(P['ampsd'] * rng.standard_normal(len(t)))
    if cls == 'IR':
        th = 2 * np.pi * f_r * t + rng.uniform(0, 2 * np.pi); A *= P['amin'] + (1 - P['amin']) * stribeck(th, P['eps'])
        A *= (1 + P['kpath'] * np.cos(th - P['thpath'])) / (1 + P['kpath'])
    elif cls == 'B':
        th = 2 * np.pi * ftf * f_r * t + rng.uniform(0, 2 * np.pi); A *= P['amin'] + (1 - P['amin']) * stribeck(th, P['eps'])
        A *= np.where(np.arange(len(t)) % 2 == 0, 1.0, P['inner_ratio'])          # alternate outer / inner race strikes
        if P['gate']:                                                                  # spin axis wanders: defect sometimes misses
            g = np.cumsum(rng.standard_normal(len(t))) * 0.15; A *= 1 / (1 + np.exp(-(g - g.mean()) * 4))
    elif P.get('or_am', 0) > 0:   # v2: rotating load (unbalance) modulates a fixed outer-race defect at shaft rate (dev: CWRU OR shows BPFO +- k fr)
        A *= 1 + P['or_am'] * np.cos(2 * np.pi * f_r * t + rng.uniform(0, 2 * np.pi))
    # entry/exit spacing: defect length / speed of the contact over it
    D, d = geom_mm['D'], geom_mm['d']
    v = {'OR': np.pi * (D + d) * ftf * f_r, 'IR': np.pi * (D - d) * (1 - ftf) * f_r, 'B': np.pi * d * o['BSF'] * f_r}[cls]   # mm/s
    return t, A, size_mm / v, f_def

def render(t, A, delta, T, res, P):
    """place entry/exit pulse pairs on a 48 kHz grid, ring the structure, decimate to 12 kHz"""
    pre = int(0.3 * FSH); n = int(T * FSH) + pre; s = np.zeros(n)
    i = np.round((t * FSH) + pre).astype(int); ok = (i >= 0) & (i < n); np.add.at(s, i[ok], A[ok])
    ex = pulse(P['exit_w']); en = -P['entry_a'] * pulse(P['entry_w']); kd = int(round(delta * FSH))
    shape = np.zeros(kd + len(ex) + len(en) + 2); shape[:len(en)] += en; shape[kd:kd + len(ex)] += ex
    h = impulse_response(res, P['tilt']); y = ss.fftconvolve(s, np.convolve(shape, h))[:n]
    return ss.resample_poly(y, 1, UP)[pre // UP:pre // UP + int(T * FS)]

def band_power(x, lo=500, hi=5700):
    X = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(len(x), 1 / FS); return X[(f >= lo) & (f <= hi)].sum() / len(x) ** 2

def time_scale(seg, rho):
    """y(t) = seg(rho t): every frequency (shaft lines, resonances) of the background is multiplied by rho"""
    q = Fraction(rho).limit_denominator(60); return ss.resample_poly(seg, q.denominator, q.numerator), q.numerator / q.denominator

def draw_params(rng, cls, version='v1'):
    return dict(slip=rng.uniform(0.0, 0.02), jitter=rng.uniform(0.005, 0.02), ampsd=rng.uniform(0.05, 0.3), amin=rng.uniform(0.0, 0.3), eps=rng.uniform(0.3, 1.2),
                kpath=rng.uniform(0, 0.5), thpath=rng.uniform(0, 2 * np.pi), inner_ratio=rng.uniform(0.2, 1.0), gate=bool(rng.random() < 0.4),
                exit_w=rng.uniform(1, 4), entry_w=rng.uniform(4, 30), entry_a=rng.uniform(0.0, 0.6), tilt=rng.uniform(-1.0, 0.5),
                snr_db=rng.uniform(*SNR[version]), or_am=(rng.uniform(0, 0.8) if rng.random() < 0.7 else 0.0) if version != 'v1' else 0.0, imb=np.exp(rng.uniform(np.log(0.02), np.log(1.0))), mis=np.exp(rng.uniform(np.log(0.01), np.log(0.5))))

def random_geom(rng):
    n = int(rng.integers(7, 17)); r = rng.uniform(0.15, 0.33); D = np.exp(rng.uniform(np.log(20), np.log(120)))
    return dict(n=n, d=r * D, D=D, phi=0.0 if rng.random() < 0.8 else rng.uniform(0, 20))

def simulate(rng, bg, f_bg, cls=None, geom=None, size_mm=None, rho=None, P=None, distract=None, aperiodic=None, version='v1'):
    """bg: healthy recording at 12 kHz (>= 1.25 * NW samples), f_bg its shaft speed in Hz.
    Returns x (NW samples at 12 kHz) and the truth."""
    cls = cls or CLS[rng.integers(4)]; geom = geom or random_geom(rng); P = P or draw_params(rng, cls, version)
    size_mm = size_mm or float(np.exp(rng.uniform(np.log(0.1), np.log(6.0))))
    rho = rho or float(np.exp(rng.uniform(np.log(0.85), np.log(1.18))))
    need = int(NW * rho * 1.02) + 64; st = rng.integers(0, len(bg) - need); y, rho = time_scale(bg[st:st + need], rho); y = y[:NW]; y = y - y.mean()
    f_r = rho * f_bg; T = NW / FS; t = np.arange(NW) / FS; pb = band_power(y); sig = np.zeros(NW)
    if cls != 'N':
        tt, A, delta, f_def = fault_train(rng, cls, f_r, geom, size_mm, T, P); sig = render(tt, A, delta, T, resonances(rng, version=version), P)
        sig *= np.sqrt(pb * 10 ** (P['snr_db'] / 10) / max(band_power(sig), 1e-30))
    distract = (rng.random() < (0.15 if version == 'v1' else 0.3)) if distract is None else distract
    if distract:   # a periodic impact train from somewhere else (another bearing, a gear) that is NOT this bearing's fault
        o = FE.orders(geom['n'], geom['d'], geom['D']); bad = np.array([o['BPFO'], o['BPFI'], 2 * o['BSF'], o['BSF']])
        while True:
            ordr = np.exp(rng.uniform(np.log(1.5), np.log(9)))
            if np.all(np.abs(np.log(ordr / bad)) > 0.06) and abs(ordr - round(ordr)) > 0.08: break
        g2 = dict(n=10, d=10, D=50); Q = draw_params(rng, 'OR'); Q['slip'] = 0
        f_d = ordr * f_r; n = int(T * f_d) + 30; tt = -0.3 + rng.uniform(0, 1 / f_d) + np.cumsum((1 + Q['jitter'] * rng.standard_normal(n)) / f_d)
        s2 = render(tt[tt < T], np.exp(Q['ampsd'] * rng.standard_normal((tt < T).sum())), 1e-4, T, resonances(rng), Q)
        lo, hi = (-24, -3) if version == 'v1' else (-20, 20)     # v2: the other bearing in CWRU dev files is as loud as 20 dB
        sig = sig + s2 * np.sqrt(pb * 10 ** (rng.uniform(lo, hi) / 10) / max(band_power(s2), 1e-30))
    aperiodic = (rng.random() < 0.15) if aperiodic is None else aperiodic
    if aperiodic:  # random knocks (Poisson), not periodic
        k = rng.poisson(rng.uniform(3, 40) * T); tt = np.sort(rng.uniform(-0.2, T, k)); Q = draw_params(rng, 'OR')
        s3 = render(tt, np.exp(0.5 * rng.standard_normal(k)), 1e-4, T, resonances(rng), Q)
        sig = sig + s3 * np.sqrt(pb * 10 ** (rng.uniform(-20, 0 if version == 'v1' else 10) / 10) / max(band_power(s3), 1e-30))
    if version != 'v1' and rng.random() < 0.5:   # v2: healthy machines show shaft-order lines in the envelope (dev: CWRU 99 / PB K006 false alarms)
        m = rng.uniform(0, 0.25, 6) / np.sqrt(np.arange(1, 7)) * (rng.random(6) < 0.6); ph = rng.uniform(0, 2 * np.pi, 6)
        y = y * (1 + sum(m[k] * np.cos(2 * np.pi * (k + 1) * f_r * t + ph[k]) for k in range(6)))
    rms = np.std(y); ph = rng.uniform(0, 2 * np.pi, 3)
    shaft = rms * (P['imb'] * np.cos(2 * np.pi * f_r * t + ph[0]) + P['mis'] * (np.cos(4 * np.pi * f_r * t + ph[1]) + 0.5 * np.cos(6 * np.pi * f_r * t + ph[2])))
    x = y + sig + shaft
    if version != 'v1' and rng.random() < 0.7:   # v2: random sensor mounting / transmission path (smooth +-8 dB gain bumps)
        X = np.fft.rfft(x); f = np.maximum(np.fft.rfftfreq(len(x), 1 / FS), 1.0); gdb = np.zeros(len(f))
        for _ in range(3):
            fc = np.exp(rng.uniform(np.log(300), np.log(5500))); w = rng.uniform(0.2, 1.0); gdb += rng.uniform(-8, 8) * np.exp(-0.5 * (np.log2(f / fc) / w) ** 2)
        x = np.fft.irfft(X * 10 ** (gdb / 20), len(x))
    truth = dict(cls=cls, size_mm=size_mm if cls != 'N' else 0.0, f_r=f_r, rho=rho, geom=geom, snr_db=P['snr_db'] if cls != 'N' else None, distract=bool(distract), aperiodic=bool(aperiodic))
    return x.astype(np.float32), truth
