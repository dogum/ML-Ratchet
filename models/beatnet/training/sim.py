"""BeatNet simulator: a random patient (heart shape, orientation, conduction, repolarisation, lead placement), its beat
types from physiology, and a rhythm strip with real-noise injection.

Beat physiology:
  N  sinus P (sinus node, upper lateral right atrium) -> AV delay -> His bundle -> Purkinje root sites (LV septum,
     LV anterior and posterior paraseptal, RV moderator band). Bundle-branch block patients (still N): one side's
     roots are missing, so that ventricle is reached through the septum and its endocardium conducts slowly.
  S  atrial premature beat: an early P from an ectopic atrial focus, longer PR (decremental AV node), normal QRS
     (aberrant, right-bundle-like, when very early); junctional premature: normal QRS, retrograde P near the QRS.
     Supraventricular runs. The sinus node is reset (pause shorter than compensatory).
  V  premature ventricular beat: an ectopic ventricular focus, slow cell-to-cell spread (little Purkinje engagement),
     so a wide QRS with its own axis and a discordant T; no P of its own; the next sinus P finds the ventricles
     refractory, so the pause is fully compensatory. Couplets and runs.
  F  fusion: an ectopic focus fires within a few tens of ms of the normal activation; both wavefronts meet,
     tau = min(tau_normal, tau_ectopic + delta).
Rhythm: sinus RR from the ECGSYN spectral model (McSharry et al. 2003: Gaussian LF and HF bumps), atrial fibrillation
(irregular RR, fibrillatory waves, no P) and atrial flutter (sawtooth, 2:1-4:1 conduction), restitution (APD follows RR)."""
import numpy as np, scipy.signal as ss
import heart as Hm

FS = 360
# Physics versions. v1: textbook values. Later versions add corrections measured on the real dev records (calib.py).
VERSIONS = {1: dict(apd_shift=0.0, k_lo=0.5, k_hi=0.85, tm=(10, 35), engage=(1.8, 4.0), fast=(1.2, 2.2), myo=(0.22, 0.36), wr=(14, 24), snr=(12, 42), snr_bad=(6, 12), white=(0.003, 0.015), vgain=1.0, bw=(0.02, 0.4))}
VERSIONS[2] = dict(VERSIONS[1], apd_shift=47.0, k_lo=0.1, k_hi=0.45, tm=(15, 45), wr=(24, 36), engage=(2.0, 4.0), vgain=1.35, snr=(30, 55), snr_bad=(15, 30), white=(0.0005, 0.004), bw=(0.01, 0.25))
CAL = dict(VERSIONS[1])
VERSIONS[3] = dict(VERSIONS[2], fusion_share=(0.25, 0.75), quiet=(0.2, 0.65), snr=(40, 64), snr_bad=(24, 40), white=(0.00013, 0.0013))   # dev noise floor: v2 noise ~1.7x the dev floor at every percentile -> NSTDB -10 dB, + quiet real background from the dev records (quietnoise.py); trial c3try at -6 dB was still 1.5x;
# fusion_share: r1 called 981 of 9110 non-AF dev N beats F; simulated 'fusions' whose ectopic front took only a sliver (or almost all) of the ventricles looked N (or V) but were labelled F
VERSIONS[4] = dict(VERSIONS[3], s_coup=(0.28, 0.6), s_run=(0.3, 0.5), j_coup=(0.4, 0.7))   # dev: S beats arrive at 0.55 (IQR 0.49-0.64) of the local mean RR, v3 sims at 0.78 (0.69-0.88) -> earlier atrial/junctional coupling
def set_version(v): CAL.clear(); CAL.update(VERSIONS[v]); CAL['version'] = v

def rr_ecgsyn(n, hr, std, lfhf, rng, f1=0.1, f2=0.25, c1=0.01, c2=0.01):
    """RR series (s), ECGSYN method: power spectrum = two Gaussians (Mayer waves at f1, RSA at f2), random phases,
    inverse FFT, scaled to mean 60/hr and given std. Sampled at 1 Hz per beat index (as in ECGSYN, with sfrr = 1)."""
    m = 1 << int(np.ceil(np.log2(max(n, 4) * 2))); f = np.arange(m) / m
    S = lfhf * np.exp(-(f - f1) ** 2 / (2 * c1 ** 2)) + np.exp(-(f - f2) ** 2 / (2 * c2 ** 2)); 
    A = np.sqrt(np.concatenate([S[:m // 2 + 1], S[1:m // 2][::-1]]))
    ph = rng.uniform(0, 2 * np.pi, m // 2 - 1); P = np.concatenate([[0], ph, [0], -ph[::-1]])
    x = np.real(np.fft.ifft(A * np.exp(1j * P)))[:n]; x = (x - x.mean()) / max(x.std(), 1e-12)
    return 60.0 / hr + std * (60.0 / hr) * x


class Patient:
    def __init__(self, seed, bbb=None):
        rng = np.random.default_rng(seed); self.rng = rng; u = rng.uniform
        g = dict(Rc=u(22, 28), Lc=u(52, 64), T=u(9, 13), d=u(17, 23), Rp=u(23, 30), Rq=u(30, 38), Lr=u(36, 46), Trv=u(3.5, 5.5), Ra=u(16, 22), purk_top=u(0.04, 0.15))
        self.h = h = Hm.Heart(g)
        # orientation: anatomical axis variation in the frontal plane, rotation about the long axis, and tilt
        self.R = Hm.rot([0, 0, 1], np.radians(rng.normal(12, 18))) @ Hm.rot(-h.B[2], np.radians(rng.normal(0, 18))) @ Hm.rot([1, 0, 0], np.radians(rng.normal(0, 10)))
        self.M = self.R @ h.B.T                            # heart-frame vector -> body frame
        self.cv = dict(myo=u(*CAL['myo']), fast=u(*CAL['fast']), atr=u(0.75, 1.1))
        self.bbb = bbb if bbb is not None else rng.choice(['none', 'LBBB', 'RBBB'], p=[0.8, 0.1, 0.1])
        G = h.G; post = 1.0 if h.B[1] @ np.array([0, 0, 1.0]) > 0 else -1.0; Rc, Lc = G['Rc'], G['Lc']
        self.roots = dict(lvs=h.nearest((Rc, 0, -u(0.35, 0.55) * Lc), h.endo_lv), lva=h.nearest((0.5 * Rc, -post * 0.85 * Rc, -u(0.35, 0.55) * Lc), h.endo_lv),
                          lvp=h.nearest((0.5 * Rc, post * 0.85 * Rc, -u(0.45, 0.65) * Lc), h.endo_lv), rv=h.nearest((G['d'] + G['Rp'], 0, -u(0.4, 0.65) * G['Lr']), h.endo_rv))
        self.root_t = dict(lvs=0.0, lva=u(0, 7), lvp=u(0, 7), rv=u(3, 11))
        ca = np.array([G['d'] - 2, -5 * post, G['Ra'] + 4]); self.sinus = h.nearest(ca + [0.4 * G['Ra'], 0, 0.9 * G['Ra']], h.atrial)
        self.avn = h.nearest((G['d'] * 0.2, 0, 2 * Hm.HV), h.atrial)
        # repolarisation: APD at RR = 1 s, transmural and apico-basal gradients, electrotonic coupling to activation time
        self.rep = dict(apd1=u(235, 295) + CAL['apd_shift'], tm=u(*CAL['tm']), ab=u(-15, 10), k=u(CAL['k_lo'], CAL['k_hi']), wr=u(*CAL['wr']), apda=u(170, 230), wra=u(10, 20))
        self.PR = u(125, 205); self.atr_gain = np.exp(rng.normal(0, 0.25)); self.vent = ~h.atrial
        self._maps = {}; self.scale = None; self.vgain = float(np.exp(rng.normal(np.log(CAL['vgain']), 0.25))) if CAL['vgain'] != 1.0 else 1.0; self.scale_target = np.exp(rng.normal(np.log(1.4), 0.3))

    # ---------- activation maps
    def vmap(self, kind, focus=None, engage=1.0):
        key = (kind, focus, round(engage, 3))
        if key in self._maps: return self._maps[key]
        h, cv = self.h, self.cv; v = [cv['myo'], cv['fast'], cv['fast'], cv['atr']]
        if kind == 'normal' or kind == 'aberrant':
            roots = dict(self.roots); bbb = self.bbb if kind == 'normal' else ('RBBB' if self.bbb == 'none' else self.bbb)
            if bbb == 'LBBB': [roots.pop(k) for k in ('lvs', 'lva', 'lvp')]; v[Hm.LVF] = cv['myo'] * 2.2
            if bbb == 'RBBB': roots.pop('rv'); v[Hm.RVF] = cv['myo'] * 1.8
            tau = h.activate([roots[k] for k in roots], [self.root_t[k] for k in roots], v)
        else:   # ectopic ventricular focus: slow spread, limited retrograde Purkinje engagement (engage x myocardial speed)
            v[Hm.LVF] = v[Hm.RVF] = min(cv['myo'] * engage, cv['fast']); tau = h.activate([focus], [0.0], v)
        tau = np.where(self.vent, tau, np.nan); tau -= np.nanmin(tau); self._maps[key] = tau; return tau

    def amap(self, focus):
        key = ('atr', focus)
        if key not in self._maps:
            h = self.h; tau = h.activate([focus], [0.0], [1, 1, 1, self.cv['atr']]); self._maps[key] = np.where(h.atrial, tau, np.nan)
        return self._maps[key]

    def random_vfocus(self):
        return int(self.rng.choice(np.nonzero(self.vent)[0]))

    def random_afocus(self):
        a = np.nonzero(self.h.atrial)[0]; d = np.linalg.norm(self.h.pos[a] - self.h.pos[self.sinus], axis=1); return int(self.rng.choice(a[d > 15]))

    # ---------- dipoles (body frame, mV after scaling)
    def vdipole(self, tau, rr, key=None):
        """ventricular dipole for activation map tau after a preceding RR of rr seconds. The repolarisation part only
        shifts in time with rr (restitution changes APD uniformly), so both parts are cached per map."""
        if self.scale is None:   # patient amplitude: the normal beat's peak QRS vector magnitude is set to scale_target mV
            tn = self.vmap('normal'); Hn = Hm.dipole(self.h, tn, tn + 1e4, np.isfinite(tn), 250, 1.5, 15)
            self.scale = self.scale_target / np.abs(np.linalg.norm(Hn, axis=1)).max()
        ck = ('dip', key) if key is not None else None
        if ck is None or ck not in self._maps:
            r = self.rep; h = self.h; m = np.isfinite(tau); apd = r['apd1'] + r['tm'] * (1 - h.e) + r['ab'] * (h.b - 0.5)
            rt = tau + apd - r['k'] * (tau - np.nanmedian(tau)); L = int(np.nanmax(rt) + 6 * r['wr']) + 10
            gt = h.grad(np.where(m, tau, np.nan)); gr = h.grad(np.where(m, rt, np.nan)); vol = Hm.HV ** 3
            Hd = vol * Hm.sweep(tau[m], gt[m], L, 1.5) @ self.M.T * self.scale; Hr = -vol * Hm.sweep(rt[m], gr[m], L, r['wr']) @ self.M.T * self.scale
            if ck is None: return self._shift(Hd, Hr, rr)
            self._maps[ck] = (Hd, Hr)
        return self._shift(*self._maps[ck], rr)

    def _shift(self, Hd, Hr, rr):
        s = int(round(self.rep['apd1'] * (np.clip(rr, 0.25, 2.0) ** 0.35 - 1))); L = len(Hd); out = Hd.copy()
        if s < 0: out[:L + s] += Hr[-s:]
        else: out = np.concatenate([out, np.zeros((s, 3))]); out[s:] += Hr
        return out

    def adipole(self, tau):
        r = self.rep; rt = tau + r['apda']; L = int(np.nanmax(rt) + 6 * r['wra']) + 10
        if self.scale is None: self.vdipole(self.vmap('normal'), 1.0)
        return Hm.dipole(self.h, tau, rt, np.isfinite(tau), L, 1.5, r['wra']) @ self.M.T * self.scale * self.atr_gain

    def adip(self, focus):
        k = ('adip', focus)
        if k not in self._maps: self._maps[k] = self.adipole(self.amap(focus))
        return self._maps[k]

    def qrs_width(self, tau):
        """QRS duration (ms) from the depolarisation vector: span where |H_dep| exceeds 5% of its peak"""
        k = ('qw', hash(tau.tobytes()))
        if k not in self._maps: self._maps[k] = self._qw(tau)
        return self._maps[k]

    def _qw(self, tau):
        Hd = Hm.dipole(self.h, tau, tau + 1e4, np.isfinite(tau), int(np.nanmax(tau)) + 40, 1.5, 15); m = np.linalg.norm(Hd, axis=1); k = np.nonzero(m > 0.05 * m.max())[0]
        return float(k[-1] - k[0]) * Hm.DT
