"""Rhythm strips: event timing (sinus / AF / flutter base rhythm with S, V and F ectopy), synthesis of the heart vector at
1 kHz, lead projection, real noise from the MIT-BIH Noise Stress Test Database, and the Holter recording chain."""
import _paths as _P  # portable paths (see _paths.py)
import numpy as np, scipy.signal as ss, wfdb
import sim, heart as Hm

NOISE = None

def noise_bank():
    """nstdb: bw (baseline wander), em (electrode motion), ma (muscle artefact); 2 channels each, 360 Hz, mV.
    Each is band-limited RMS-normalised (0.5-40 Hz) so levels can be set directly."""
    global NOISE
    if NOISE is None:
        NOISE = {}
        for k in ('bw', 'em', 'ma'):
            x = wfdb.rdrecord(f'{_P.WORK}/data/nstdb/{k}').p_signal.T.astype(np.float32)
            sos = ss.butter(2, [0.5, 40], 'bandpass', fs=360, output='sos'); r = np.array([np.std(ss.sosfiltfilt(sos, c[:200000])) for c in x])
            NOISE[k] = (x, r)
    return NOISE


QUIET = None
def quiet_bank():
    global QUIET
    if QUIET is None: QUIET = np.load(_P.WORK + '/data/quiet_dev.npz')['x']
    return QUIET


def program(rng):
    """random strip program: base rhythm and ectopy"""
    u = rng.uniform; base = rng.choice(['sinus', 'af', 'flutter'], p=[0.82, 0.14, 0.04])
    mode = rng.choice(['none', 'S', 'V', 'SV', 'F'], p=[0.14, 0.30, 0.34, 0.10, 0.12]) if base == 'sinus' else rng.choice(['none', 'V', 'F'], p=[0.4, 0.45, 0.15])
    P = dict(base=base, mode=mode, hr=float(np.exp(rng.normal(np.log(75), 0.22))), std=u(0.01, 0.08), lfhf=u(0.5, 2.5),
             ps=u(0.05, 0.35), pv=u(0.05, 0.4), pattern=rng.choice(['iso', 'big', 'tri', 'pair', 'run'], p=[0.6, 0.13, 0.1, 0.1, 0.07]),
             nfoci=int(rng.choice([1, 1, 1, 2, 3])), junc=u() < 0.15, aberr=u(0, 0.3), interp=u() < 0.2)
    if base == 'af': P['hr'] = float(np.exp(rng.normal(np.log(90), 0.25))); P['af_sd'] = u(0.08, 0.3)
    return P


def events(p, P, rng, n):
    """list of ventricular events dict(t (ms, QRS onset), kind, label, rr) and atrial events dict(t, focus) or f-wave info"""
    u = rng.uniform; RRs = sim.rr_ecgsyn(4 * n + 20, P['hr'], P['std'], P['lfhf'], rng) * 1000; ri = 0
    def rr_now():
        nonlocal ri; ri += 1; return max(RRs[min(ri, len(RRs) - 1)], 300.0)
    foci = [(p.random_vfocus(), u(*sim.CAL['engage']) if u() > 0.15 else 8.0, u(0.38, 0.75)) for _ in range(P['nfoci'])]   # (voxel, Purkinje engagement, coupling)
    afoc = [p.random_afocus() for _ in range(2)]
    V, A = [], []; t = 600.0; lastV = -1e4; lastP = t - 800; PR = p.PR
    want_s = P['mode'] in ('S', 'SV'); want_v = P['mode'] in ('V', 'SV', 'F')
    pat = P["pattern"]; cnt = 0; run = 0; run_kind = None
    while len(V) < n:
        rr = rr_now(); tp = lastP + rr if P['base'] == 'sinus' else None
        # ---- choose the next event
        ect = None
        if run > 0: ect = run_kind; run -= 1
        elif cnt > 0 or not V:
            k = cnt % {'big': 2, 'tri': 3}.get(pat, 10 ** 6)
            fire = (k == 0 and pat in ('big', 'tri') and u() < 0.85) or (pat not in ('big', 'tri') and u() < (P['pv'] if want_v else P['ps']))
            if fire and (want_s or want_v):
                ect = 'V' if (want_v and (not want_s or u() < 0.5)) else 'S'
                if pat == 'pair': run = 1; run_kind = ect
                if pat == 'run' and u() < 0.5: run = int(rng.integers(2, 7)); run_kind = ect
        cnt += 1
        if P['base'] != 'sinus':                      # AF / flutter: irregular or ratio-conducted ventricular response
            if P['base'] == 'af': tv = (V[-1]['t'] if V else t) + max(60000 / P['hr'] * np.exp(rng.normal(0, P['af_sd'])), 280)
            else:
                fl = P.setdefault('fl', u(200, 260)); ratio = P.setdefault('ratio', int(rng.choice([2, 2, 3, 4]))); r2 = ratio if u() > 0.3 else int(rng.choice([2, 3, 4]))
                tv = (V[-1]['t'] if V else t) + r2 * fl
            if ect == 'V':
                f = foci[int(rng.integers(len(foci)))]; tv = (V[-1]['t'] if V else t) + max(f[2] * 60000 / P['hr'] + rng.normal(0, 15), 260)
                V.append(dict(t=tv, kind=('pvc', f[0], f[1]), label=2))
            else: V.append(dict(t=tv, kind=('normal',), label=0))
            continue
        if ect == 'V':
            f = foci[int(rng.integers(len(foci)))]; c = f[2] if run_kind != 'V' or run == 0 else u(0.4, 0.6)
            tv = (V[-1]['t'] if V else t) + max(c * rr + rng.normal(0, 15), 260); tn = tp + PR
            if P['mode'] == 'F' and run == 0 and u() < 0.8: tv = tn + u(-50, 25)       # late-coupled focus: lands on the sinus beat
            if tv < tn - 45:                                       # pure premature ventricular beat; sinus P continues on time
                V.append(dict(t=tv, kind=('pvc', f[0], f[1]), label=2)); A.append(dict(t=tp, focus=p.sinus)); lastP = tp
                erp = p.rep['apd1'] * (rr / 1000) ** 0.35 + 0.5 * p.qrs_width(p.vmap('pvc', f[0], f[1])) + 40
                if tn - tv < erp and not (P['interp'] and tn - tv > erp - 60): continue    # blocked: compensatory pause
                V.append(dict(t=tn + (rng.uniform(20, 80) if tn - tv < erp + 100 else 0), kind=('normal',), label=0)); continue
            if tv <= tn + 25:                                      # fusion
                d = tv - tn; lab = 3
                if sim.CAL.get('fusion_share'):                    # v3: a fusion is only visible (and annotated F) when each front takes a sizeable share
                    sh = ectopic_share(p, f[0], f[1], d); lo, hi = sim.CAL['fusion_share']; lab = 0 if sh < lo else (2 if sh > hi else 3)
                V.append(dict(t=tn, kind=('fusion', f[0], f[1], d), label=lab)); A.append(dict(t=tp, focus=p.sinus)); lastP = tp; continue
            ect = None                                             # too late: the sinus beat came first
        if ect == 'S':
            if P['junc'] and u() < 0.5:                            # junctional premature: normal QRS, retrograde P near it
                tv = (V[-1]['t'] if V else t) + u(*sim.CAL.get('j_coup', (0.55, 0.85))) * rr; V.append(dict(t=tv, kind=('normal',), label=1)); A.append(dict(t=tv + u(-25, 50), focus=p.avn)); lastP = tv - PR * 0.3
            else:
                c = u(*sim.CAL.get('s_coup', (0.5, 0.88))) if run == 0 else u(*sim.CAL.get('s_run', (0.45, 0.65))); ta = lastP + c * rr; fa = afoc[0] if u() < 0.75 else afoc[1]
                ab = u() < P['aberr'] * (1.6 - c); V.append(dict(t=ta + PR + u(10, 50) * (1 - c) / 0.5, kind=('aberrant',) if ab else ('normal',), label=1))
                A.append(dict(t=ta, focus=fa)); lastP = ta + rr * u(0.0, 0.12)   # sinus node reset: next sinus P ~ one cycle later
            continue
        V.append(dict(t=tp + PR + rng.normal(0, 4), kind=('normal',), label=0)); A.append(dict(t=tp, focus=p.sinus)); lastP = tp
    return V[:n], A


def ectopic_share(p, focus, engage, d):
    """share of ventricular voxels that the ectopic front reaches first when it starts d ms after the normal one"""
    tn = p.vmap('normal'); te = p.vmap('pvc', focus, engage); m = np.isfinite(tn) & np.isfinite(te); return float(np.mean(te[m] + d < tn[m]))


def vmap_of(p, kind):
    if kind[0] == 'normal': return p.vmap('normal')
    if kind[0] == 'aberrant': return p.vmap('aberrant')
    if kind[0] == 'pvc': return p.vmap('pvc', kind[1], kind[2])
    tn = p.vmap('normal'); te = p.vmap('pvc', kind[1], kind[2]); return np.fmin(tn, te + kind[3])


def synth(p, P, V, A, rng, leads=('II',)):
    """heart vector at 1 kHz -> lead voltages (mV, 1 kHz) with respiration; returns (sig (nlead, T), qrs onsets ms, widths ms)"""
    T = int(V[-1]['t'] + 1500); H = np.zeros((T, 3)); cache = {}; widths = []
    for i, e in enumerate(V):
        key = e['kind'] if e['kind'][0] != 'fusion' else ('fusion', e['kind'][1], e['kind'][2], round(e['kind'][3] / 5) * 5)
        tau = vmap_of(p, key); prev = V[i - 1]['t'] if i else e['t'] - 800; rr = (e['t'] - prev) / 1000; rk = (key, round(rr, 2))
        if rk not in cache: cache[rk] = p.vdipole(tau, rr, key)
        if key not in cache: cache[key] = p.qrs_width(tau)
        d = cache[rk] * (p.vgain if key[0] == 'pvc' else 1.0); t0 = int(e['t']); n = min(len(d), T - t0); H[t0:t0 + n] += d[:n]; widths.append(cache[key])
    for e in A:
        d = p.adip(e['focus']); t0 = int(e['t']); n = min(len(d), T - max(t0, 0))
        if t0 >= 0: H[t0:t0 + n] += d[:n]
    if P['base'] == 'af':          # fibrillatory waves: band-limited (4-9 Hz) random atrial dipole
        f0 = rng.uniform(4.5, 8.5); b, a = ss.butter(2, [f0 - 1.2, f0 + 1.2], 'bandpass', fs=1000); fw = ss.lfilter(b, a, rng.normal(0, 1, (T, 3)), axis=0)
        H += fw / fw.std() * rng.uniform(0.01, 0.08) * p.atr_gain
    if P['base'] == 'flutter':     # sawtooth atrial macro-reentry, mostly along the inferior-superior axis
        fl = P['fl']; ph = (np.arange(T) % fl) / fl; saw = np.where(ph < 0.75, ph / 0.75, (1 - ph) / 0.25) - 0.5
        H += np.outer(ss.lfilter(*ss.butter(2, 25, fs=1000), saw), Hm.unit([0.2, -1, 0.3])) * rng.uniform(0.08, 0.25)
    # respiration: small rotation in the frontal plane and amplitude modulation
    fr = rng.uniform(0.15, 0.35); ph = rng.uniform(0, 2 * np.pi); tt = np.arange(T) / 1000; th = np.radians(rng.uniform(2, 8)) * np.sin(2 * np.pi * fr * tt + ph)
    am = 1 + rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * fr * tt + ph + 1.0)
    out = []
    for ld in leads:
        Lv = Hm.rot(rng.normal(size=3), np.radians(rng.normal(0, 10))) @ np.array(Hm.DOWER[ld]) * np.exp(rng.normal(0, 0.12))  # electrode placement
        c, s = np.cos(th), np.sin(th); Hx = c * H[:, 0] - s * H[:, 1]; Hy = s * H[:, 0] + c * H[:, 1]
        out.append(am * (Lv[0] * Hx + Lv[1] * Hy + Lv[2] * H[:, 2]))
    return np.array(out), np.array([e['t'] for e in V]), np.array(widths)


def record(sig1k, qrs_on, widths, rng, snr_db=None):
    """1 kHz clean leads (mV) -> 360 Hz Holter record with real noise; annotation fiducials at the clean QRS peak"""
    x = ss.resample_poly(sig1k, 9, 25, axis=-1); clean = x.copy(); n = x.shape[-1]; NB = noise_bank()
    sos = ss.butter(2, [0.5, 40], 'bandpass', fs=360, output='sos'); cb = ss.sosfiltfilt(sos, clean, axis=-1)
    on = np.round(qrs_on * 0.36).astype(int); wd = np.maximum(np.round(widths * 0.36).astype(int), 8)
    ann = np.array([o + int(np.argmax(np.abs(cb[0, o:o + w]))) for o, w in zip(on, wd)]) + np.round(rng.normal(0, 1.2, len(on))).astype(int)
    qamp = np.median([np.ptp(cb[0, max(a - 25, 0):a + 25]) for a in ann])
    snr = snr_db if snr_db is not None else (rng.uniform(*sim.CAL['snr']) if rng.uniform() > 0.15 else rng.uniform(*sim.CAL['snr_bad']))
    for ch in range(x.shape[0]):
        w = rng.dirichlet([1, 1]); tot = qamp * 10 ** (-snr / 20)
        for k, wk in zip(('bw', 'em', 'ma'), (None, *w)):
            arr, r = NB[k]; c = int(rng.integers(2)); s0 = int(rng.integers(0, arr.shape[1] - n)); seg = arr[c, s0:s0 + n] - arr[c, s0:s0 + n].mean()
            lvl = rng.uniform(*sim.CAL['bw']) / max(seg.std(), 1e-6) if k == 'bw' else tot * np.sqrt(wk) / r[c] * (rng.uniform() < 0.9)
            x[ch] += seg * lvl
        x[ch] += rng.normal(0, rng.uniform(*sim.CAL['white']), n)
        if sim.CAL.get('quiet'):           # quiet real background (dev records), in units of the QRS size
            Q = quiet_bank(); k = rng.integers(0, len(Q), n // Q.shape[1] + 1); x[ch] += np.concatenate(Q[k])[:n] * qamp * rng.uniform(*sim.CAL['quiet'])
        if rng.uniform() < 0.3: x[ch] += rng.uniform(0.005, 0.04) * np.sin(2 * np.pi * 60 * np.arange(n) / 360 + rng.uniform(0, 6.3))
    # recorder: single-pole high-pass, 2-pole low-pass, 11-bit ADC at 200 units/mV
    b, a = ss.butter(1, rng.uniform(0.05, 0.5), 'highpass', fs=360); x = ss.lfilter(b, a, x, axis=-1)
    b, a = ss.butter(2, rng.uniform(40, 100), fs=360); x = ss.lfilter(b, a, x, axis=-1)
    for i in range(1, len(ann)): ann[i] = max(ann[i], ann[i - 1] + 40)
    return np.round(x * 200) / 200, ann
