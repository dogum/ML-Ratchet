"""Classical baselines on the same real trips (all constants fitted on the 16 dev cars only):
 1. physics regression: least squares fit of the road-load equation to the logged source power
        eta * P_source = m v (1.03 a + g grade) + m g Crr v + 0.5 rho CdA v_air^2 v + P0
    on traction seconds (v > 3 m/s, a > -0.3, P > 0, 5 s smoothing), eta = an engine-efficiency guess
    (gasoline 0.24 at the fuel, battery-to-wheel 0.85), solved by non-negative least squares per car (all trips) and per trip.
 2. rule-based architecture: HV battery logged -> plug-in (BEV if no fuel/rpm, else PHEV); else engine off while moving
    (rpm < 100 at v > 5 m/s for > 3 % of moving time, or fuel = 0 for > 20 % with no rpm) -> HEV; else conventional.
 3. displacement from idle fuel flow (classical: idle fuel ~ friction ~ displacement), power law fitted on dev.
 4. dumb priors: majority class (conventional), dev-median curb mass and displacement."""
from common import *
from scipy.optimize import nnls
import features as F

RAWI = {k: i for i, k in enumerate(F.RAW)}


def smooth(x, k=5):
    return np.convolve(x, np.ones(k) / k, mode='same')


def lsq_rows(raw, fl, eta_f=0.24, eta_b=0.85):
    v = raw[0].astype(float); n = len(v)
    a = np.gradient(v); g = raw[1] / 100.; rho = raw[4]; vw = v + raw[3] * 0.6
    ok = raw[10] > 0.5
    P = np.zeros(n); have = False
    if fl[0] > 0.5: P += eta_f * raw[5] * LHV_GAS; have = True
    if fl[3] > 0.5: P += np.where(raw[7] > 0, eta_b * raw[7] * 1000, raw[7] * 1000 / eta_b); have = True
    if not have: return None
    P = smooth(P); vs = smooth(v); as_ = smooth(a)
    m = ok & (vs > 3) & (as_ > -0.3) & (P > 0)
    X = np.stack([vs * (1.03 * as_ + G * g), G * vs, 0.5 * rho * vw * np.abs(vw) * vs, np.ones(n)], 1)[m]
    return X, P[m]


def lsq_fit(rows):
    rows = [r for r in rows if r is not None and len(r[1]) > 20]
    if not rows: return None
    X = np.concatenate([r[0] for r in rows]); y = np.concatenate([r[1] for r in rows])
    sc = np.abs(X).mean(0) + 1e-9
    x, _ = nnls(X / sc, y); x = x / sc
    m = x[0]
    return dict(mass=m, Crr=x[1] / max(m, 1), CdA=x[2], P0=x[3])


def rule_arch(raws, fls):
    """per car or per trip: list of raw arrays and flags"""
    batt = np.mean([f[3] for f in fls]) > 0.5
    fuel = np.mean([f[0] for f in fls]) > 0.5; rpm = np.mean([f[2] for f in fls]) > 0.5
    if batt:
        return 'phev_powersplit' if (fuel or rpm) else 'bev'
    off = []; zf = []
    for r, f in zip(raws, fls):
        mv = (r[0] > 5) & (r[10] > 0.5)
        if mv.sum() < 10: continue
        if f[2] > 0.5: off.append(np.mean(r[6][mv] < 0.1))
        elif f[0] > 0.5: zf.append(np.mean(r[5][mv] <= 0))
    if off and np.mean(off) > 0.03: return 'hev_powersplit'
    if not off and zf and np.mean(zf) > 0.2: return 'hev_powersplit'
    return 'conv'


def idle_fuel(raws, fls):
    x = []
    for r, f in zip(raws, fls):
        if f[0] < 0.5: continue
        m = (r[0] < 0.3) & (r[10] > 0.5) & (r[5] > 0)
        if f[2] > 0.5: m &= r[6] > 0.3
        if m.sum() >= 10: x.append(np.median(r[5][m]))
    return np.median(x) if x else np.nan


class Baselines:
    def fit_dev(self, rb, vt):
        """fit eta (scale of the physics regression) and the idle-fuel displacement law on dev cars"""
        o = rb['offs']; cars = list(dict.fromkeys(rb['vid'].tolist()))
        fits, idle, disp, curb = [], [], [], []
        for c in cars:
            js = np.where(rb['vid'] == c)[0]
            raws = [rb['raw'][o[j]:o[j + 1]].T.astype(np.float32) for j in js]; fls = [rb['flags'][j] for j in js]
            f = lsq_fit([lsq_rows(r, fl) for r, fl in zip(raws, fls)])
            cb = vt.loc[c, 'gen_weight_lb'] * LB + CURB_OFFSET
            if f and np.isfinite(cb) and vt.loc[c, 'ved_type'] in ('ICE',): fits.append(f['mass'] / cb)
            i = idle_fuel(raws, fls); d = vt.loc[c, 'displacement_L']
            if np.isfinite(i) and np.isfinite(d) and vt.loc[c, 'ved_type'] == 'ICE': idle.append(i); disp.append(d)
            if np.isfinite(vt.loc[c, 'gen_weight_lb']): curb.append(vt.loc[c, 'gen_weight_lb'] * LB)
        self.mass_scale = float(np.median(fits)) if fits else 1.0       # absorbs the efficiency guess error (fitted on dev ICE)
        A = np.stack([np.ones(len(idle)), np.log(idle)], 1)
        self.dcoef = np.linalg.lstsq(A, np.log(disp), rcond=None)[0]
        self.curb_med = float(np.median(curb)); self.disp_med = float(np.nanmedian(vt.loc[cars, 'displacement_L']))
        return dict(mass_scale=self.mass_scale, disp_coef=self.dcoef.tolist(), curb_med=self.curb_med, disp_med=self.disp_med, n_fit=len(fits), n_idle=len(idle))

    def predict(self, raws, fls):
        f = lsq_fit([lsq_rows(r, fl) for r, fl in zip(raws, fls)])
        i = idle_fuel(raws, fls)
        return dict(arch=rule_arch(raws, fls),
                    mass=(f['mass'] / self.mass_scale - CURB_OFFSET) if f else np.nan,   # curb estimate
                    CdA=f['CdA'] if f else np.nan, Crr=f['Crr'] if f else np.nan,
                    disp=float(np.exp(self.dcoef[0] + self.dcoef[1] * np.log(i))) if np.isfinite(i) else np.nan)
