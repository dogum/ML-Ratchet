"""DemandCast simulator: random regions built from the physics, real training-year weather and real noise.

A simulated region-year:
  weather   one focus BA's real weather of a TRAINING-INPUT year (2018, 2020, 2022), perturbed: temperature offset and
            anomaly scaling, humidity factor, cloudiness factor, wind-speed scale (same perturbation at every point)
  demand    building-stock model (buildings.py) with random TRUE parameters (around one BA's dev fit, or broad uniform
            ranges) -> truth x (1 + real relative residual from a training-year fit, aligned on local hour);
            the network's PRIOR uses DESCRIPTOR parameters = true ones with drift (level, BTM growth, electrification,
            setpoints, schedules), like a dev-year fit applied years later
  wind      the BA's real wind fleet as of a random date 2018-2022 with random per-group scaling; truth from the farm
            physics with random losses, specific power / hub shifts, and a weather error (true hub wind = reanalysis x
            exp(static point bias + correlated AR(1) common + local terms)), reporting scale and curtailment;
            prior = nominal physics on the reanalysis x a drifted scale k (like the dev-calibrated k)
  solar     the BA's real solar fleet (same snapshot / scaling); truth with a cloud error (AR(1) per point) and a
            reporting scale; prior = nominal pvlib CFs x drifted k
  forecast  half of the samples get synthetic day-ahead forecast errors on the 38 h after issue (temperature,
            humidity, irradiance, wind) - or real 2022 temperature forecast errors (Open-Meteo) when available
"""
import json, os, sys, time
import numpy as np, pandas as pd
from common import *
import region as RG, renew as R, buildings as B
from calib_demand import FREE

DEV = json.load(open(f'{DATA}/calib/demand_dev.json'))
RNOISE = dict(np.load(f'{DATA}/noise/demand_resid.npz'))
RCAT = {b: np.concatenate([v[HOURS.year == y] for y in TRAIN_YEARS]) for b, v in RNOISE.items()}
RDEV = json.load(open(f'{DATA}/calib/renew_dev.json'))
_BUNDLE = {}
# simulator versions: v1 = first guess; v2 = drift/noise/error widths calibrated so the simulated prior-error
# distributions match real dev (cross-year fits, diag_simreal.py)
CFGS = dict(
    v1=dict(p_near=0.6, near_sd=0.3, drift=1.0, p0_drift=0.06, noise=(0.6, 1.3), w_sd=(0.06, 0.30), w_drift=0.08,
            s_sd=(0.05, 0.35), s_drift=0.15),
    v2=dict(p_near=0.8, near_sd=0.2, drift=0.5, p0_drift=0.03, noise=(0.5, 1.1), w_sd=(0.06, 0.40), w_drift=0.10,
            s_sd=(0.08, 0.45), s_drift=0.30),
)
CFGS['v3'] = dict(CFGS['v2'], noise_fix=True)
CFG = CFGS[os.environ.get('DC_SIM', 'v1')]
_GROUPS = {}
TFC = None
if os.path.exists(f'{DATA}/noise/tfc_2022.npz'):
    TFC = dict(np.load(f'{DATA}/noise/tfc_2022.npz'))     # real 2022 temperature forecast errors per BA

RANGES = dict(trend=(-0.01, 0.04), sh_res=(0.15, 0.45), sh_com=(0.15, 0.40), res_m=(0.0, 0.5), res_e=(0.3, 1.0),
              res_mt=(6.5, 10.0), res_et=(18.5, 21.0), com_open=(5.5, 8.5), com_close=(16.5, 21.5),
              com_wkend=(0.35, 0.7), ind_wkend=(0.88, 0.98), wh=(-0.05, 0.05), light=(0.02, 0.09), ua=(0.07, 0.45),
              tau1=(2.0, 9.0), tau2=(60.0, 240.0), f1=(0.5, 0.9), g_win=(2.0, 7.0), mu_h=(13.0, 19.0), sd_h=(1.0, 6.0),
              mu_c=(21.0, 28.0), sd_c=(1.5, 8.0), e_res=(0.0, 0.5), e_hp=(0.0, 0.35), btm=(0.0, 0.35),
              a_sol=(1.0, 5.0), g_int=(2.0, 4.0), setback=(0.5, 2.5), setup=(0.0, 2.0), q_in=(8.0, 10.0),
              lam=(0.5, 1.0), ac=(0.7, 1.0))


def bundle(ba):
    if ba not in _BUNDLE: _BUNDLE[ba] = load_bundle(ba)
    return _BUNDLE[ba]


def groups(ba):
    if ba not in _GROUPS: _GROUPS[ba] = (R.wind_groups(ba), R.solar_groups(ba))
    return _GROUPS[ba]


def ar1(rng, n, phi, sd, size=()):
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), size + (n,))
    x = np.empty_like(e); x[..., 0] = rng.normal(0, sd, size)
    for t in range(1, n): x[..., t] = phi * x[..., t - 1] + e[..., t]
    return x


def draw_params(rng):
    p = dict(B.PARAMS)
    if rng.random() < CFG['p_near']:
        src = DEV[BAS[rng.integers(len(BAS))]]['params']
        for k in RANGES: p[k] = src.get(k, p[k])
        for k in ('ua', 'e_res', 'e_hp', 'btm', 'light'):
            p[k] = p[k] * np.exp(rng.normal(0, CFG['near_sd'])) + (0.02 * rng.random() if k in ('e_res', 'e_hp', 'btm') else 0)
        for k, s in (('mu_h', 1.5), ('mu_c', 1.5), ('res_mt', 0.7), ('res_et', 0.6), ('com_open', 0.6), ('com_close', 0.8)):
            p[k] += rng.normal(0, s)
        for k in ('sd_h', 'sd_c', 'tau1', 'tau2', 'g_win'):
            p[k] *= np.exp(rng.normal(0, 0.25))
        for k in ('a_sol', 'g_int', 'setback', 'setup', 'q_in', 'lam', 'ac'):
            lo, hi = RANGES[k]; p[k] = rng.uniform(lo, hi)
    else:
        for k, (lo, hi) in RANGES.items():
            p[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if k in ('ua', 'tau2') else rng.uniform(lo, hi)
    p['e_res'] = min(p['e_res'], 0.9); p['e_hp'] = min(p['e_hp'], 0.9 - p['e_res'])
    p['P0'] = float(np.exp(rng.uniform(np.log(3000), np.log(80000))))
    p['t_ref'] = 2020.0
    return p


def drift(rng, p):
    """Descriptor params = what a fit on earlier years would give."""
    q = dict(p)
    D = CFG['drift']
    q['P0'] = p['P0'] * np.exp(rng.normal(0, CFG['p0_drift']))
    q['btm'] = p['btm'] * rng.uniform(1 - 0.7 * D, 1.1)
    for k, s in (('ua', 0.12), ('e_res', 0.25), ('e_hp', 0.35), ('light', 0.2), ('res_e', 0.15), ('res_m', 0.3)):
        q[k] = p[k] * np.exp(rng.normal(0, s * D))
    for k, s in (('mu_h', 0.8), ('mu_c', 0.8), ('res_et', 0.3), ('res_mt', 0.3), ('com_open', 0.3), ('com_close', 0.4),
                 ('sh_res', 0.03), ('sh_com', 0.03), ('wh', 0.015), ('trend', 0.01)):
        q[k] = p[k] + rng.normal(0, s * D)
    for k in ('sd_h', 'sd_c', 'tau1', 'tau2'):
        q[k] = p[k] * np.exp(rng.normal(0, 0.15 * D))
    return q


def sim_region(rng):
    ba = BAS[rng.integers(len(BAS))]
    year = TRAIN_YEARS[rng.integers(3)]
    a = max(0, hidx(pd.Timestamp(f'{year}-01-01', tz='UTC')) - 960)
    b = min(NH, hidx(pd.Timestamp(f'{year + 1}-01-01', tz='UTC')) + 48)
    W0 = bundle(ba)
    sl = slice(a, b); n = b - a
    g = RG.Region(); g.ba = ba; g.off = a; g.year = year
    g.cal = RG.calendar_arrays(HOURS[sl], TZ[ba], LST[ba])
    # ---- weather perturbation
    dT = rng.normal(0, 2.0); aT = rng.uniform(0.85, 1.2); frh = rng.uniform(0.85, 1.15); cl = rng.uniform(0.85, 1.1)
    sw = rng.uniform(0.9, 1.1)
    T = W0['T'][sl].astype(float); Tm = RG.B.rc_filter(T, 360.0)
    Tn = Tm + aT * (T - Tm) + dT
    q = W0['q'][sl] * frh * np.exp(0.065 * (Tn - T))
    ghi = np.minimum(W0['ghi'][sl] * cl, np.maximum(W0['ghi'][sl], W0['ghi_clr'][sl]))
    g.wx = dict(T=Tn, q=q, RH=np.clip(W0['RH'][sl] * frh, 1, 100), ghi=ghi, ghi_clr=W0['ghi_clr'][sl].astype(float),
                btm_cf=np.clip(W0['btm_cf'][sl] * cl, 0, 1))
    pT = W0['pT'][:, sl].astype(float); pTm = np.stack([RG.B.rc_filter(x, 360.0) for x in pT])
    g.W = dict(pT=pTm + aT * (pT - pTm) + dT, pRH=W0['pRH'][:, sl], pws10=W0['pws10'][:, sl] * sw,
               pws50=W0['pws50'][:, sl] * sw, pps=W0['pps'][:, sl], pcf_track=np.clip(W0['pcf_track'][:, sl] * cl, 0, 1),
               pcf_fixed=np.clip(W0['pcf_fixed'][:, sl] * cl, 0, 1))
    n_pts = g.W['pws50'].shape[0]
    # ---- demand
    pt = draw_params(rng); pd_ = drift(rng, pt)
    g.pdesc = pd_; g.ptrue = pt
    from real import desc_vec
    g.desc = desc_vec(pd_)
    truth = B.demand({k: g.wx[k] for k in ('T', 'q', 'ghi', 'btm_cf')}, g.cal, pt, t_ref=2020.0)
    nb = BAS[rng.integers(len(BAS))]
    if CFG.get('noise_fix'):
        # v3: draw from the training-year residuals only (2018, 2020, 2022 concatenated; whole days keep local hour)
        res = RCAT[nb]; L = len(res)
        j0 = 24 * rng.integers(0, L // 24) + a + LST[ba] - LST[nb]
        idx = (np.arange(n) + j0) % L
    else:
        # v1/v2 (bug found by the twin test): offsets over all years, so ~60 % of hours fell outside the training
        # years where the library is NaN -> no noise there
        res = RNOISE[nb]
        k = rng.integers(0, NH // 24 - 400)
        j0 = 24 * k + (LST[ba] - LST[nb])
        idx = (np.arange(a, b) - a + j0) % NH
    r = res[idx]; r = np.where(np.isfinite(r), r, 0.0)
    g.yd = truth['total'] * (1 + rng.uniform(*CFG['noise']) * r)
    g.truth = truth
    # ---- fleets: real BA fleet snapshot (2018-2022) with random group scaling
    (wg, wcap), (sg, scap) = groups(ba)
    snap = hidx(pd.Timestamp(f'{rng.integers(2018, 2023)}-{rng.integers(1, 13):02d}-01', tz='UTC'))
    ws = np.exp(rng.normal(0, 0.6, len(wg))) * (rng.random(len(wg)) < 0.9)
    ss = np.exp(rng.normal(0, 0.6, len(sg))) * (rng.random(len(sg)) < 0.9)
    if ws.sum() == 0: ws[:] = 1
    if ss.sum() == 0: ss[:] = 1
    g.wg, g.sg = wg, sg
    g.wcap = np.repeat((wcap[:, snap] * ws)[:, None], n, 1).astype(np.float32)
    g.scap = np.repeat((scap[:, snap] * ss)[:, None], n, 1).astype(np.float32)
    if g.wcap.sum() < 1: g.wcap[:] = 100.0
    if g.scap.sum() < 1: g.scap[:] = 50.0
    # wind truth
    sd = rng.uniform(*CFG['w_sd']); fc_ = rng.uniform(0.3, 0.8)
    common = ar1(rng, n, rng.uniform(0.9, 0.98), sd * np.sqrt(fc_))
    local = ar1(rng, n, rng.uniform(0.85, 0.97), sd * np.sqrt(1 - fc_), (n_pts,))
    vm = np.exp(rng.normal(0, 0.05, (n_pts, 1)) + common[None] + local)
    sub = {k: g.W[k] for k in ('pws10', 'pws50', 'pT', 'pps', 'pRH')}
    wt = R.wind_mw(sub, wg, g.wcap, vmult=vm, wake=rng.uniform(0.06, 0.16), avail=rng.uniform(0.88, 0.99),
                   sp_shift=rng.normal(0, 25), hub_shift=rng.normal(0, 8), group_scale=np.exp(rng.normal(0, 0.1, len(wg))))
    rw = float(np.exp(rng.normal(0, 0.25)))
    wt = wt * rw
    if rng.random() < 0.4:
        lim = rng.uniform(0.5, 0.9) * rw * g.wcap.sum(0)
        days = rng.random(n // 24 + 1) < 0.2
        wt = np.where(np.repeat(days, 24)[:n], np.minimum(wt, lim), wt)
    g.yw = wt
    g.wk = 1.0
    nom = R.wind_mw(sub, wg, g.wcap)
    g.wk = float(wt.sum() / max(nom.sum(), 1e-6) * np.exp(rng.normal(0, CFG['w_drift'])))
    # solar truth
    sds = rng.uniform(*CFG['s_sd'])
    cm = np.exp(ar1(rng, n, rng.uniform(0.8, 0.95), sds * 0.6)[None] + ar1(rng, n, rng.uniform(0.8, 0.95), sds * 0.8, (n_pts,))
                + rng.normal(0, 0.04, (n_pts, 1)))
    subs = {k: g.W[k] for k in ('pcf_track', 'pcf_fixed')}
    st = R.solar_mw(subs, sg, g.scap, cfmult=cm, group_scale=np.exp(rng.normal(0, 0.08, len(sg))))
    # reporting scale: v1 log-uniform 0.12-1.2 (median 0.38); v2 half near 1 (high-CF fleets like CISO were rare in v1)
    if CFG['drift'] < 1 and rng.random() < 0.5:
        rs = float(np.exp(rng.normal(0, 0.15)))
    else:
        rs = float(np.exp(rng.uniform(np.log(0.12), np.log(1.2))))
    g.ys = st * rs
    nom = R.solar_mw(subs, sg, g.scap)
    g.sk = float(g.ys.sum() / max(nom.sum(), 1e-6) * np.exp(rng.normal(0, CFG['s_drift'])))
    g.has_solar = True
    g.finalize()
    return g


def synth_fc(rng, n_pts, ba=None):
    """Synthetic day-ahead forecast errors over the 38 h after issue."""
    L = 38
    lead = (np.arange(L) + 14) / 38.0
    if TFC is not None and ba in TFC and rng.random() < 0.7:
        e = TFC[ba]; ok = np.where(np.isfinite(e).all(1))[0]
        T = e[ok[rng.integers(len(ok))]] * rng.uniform(0.8, 1.3)
    else:
        sT = rng.uniform(0.8, 2.2)
        T = ar1(rng, L, 0.93, sT) * (0.6 + 0.4 * lead) + rng.normal(0, 0.5)
    rh = ar1(rng, L, 0.9, rng.uniform(0.03, 0.12))
    sg_ = rng.uniform(0.1, 0.45)
    ghi = np.exp(ar1(rng, L, 0.85, sg_ * 0.7)[None] + ar1(rng, L, 0.8, sg_ * 0.6, (n_pts,)))
    sv = rng.uniform(0.08, 0.25)
    v = np.exp(ar1(rng, L, 0.92, sv * 0.8)[None] + ar1(rng, L, 0.85, sv * 0.5, (n_pts,)))
    return dict(T=T, rh=rh, ghi=ghi, v=v)


KEYS = ('xd', 'yd', 'xw', 'yw', 'xs', 'ys', 'pw', 'ps', 'night')


def bank(seed, n_regions, days_per=40, p_fc=0.5, out=None):
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in KEYS}; meta = []
    t0 = time.time()
    for r in range(n_regions):
        g = sim_region(rng)
        n = len(g.wx['T'])
        cand = np.arange(182 + 400, n - 24 - 1)
        cand = cand[(cand + g.off + LST[g.ba]) % 24 == 0]          # LST midnight
        cand = cand[HOURS.year[g.off + cand] == g.year]
        for i0 in rng.choice(cand, size=min(days_per, len(cand)), replace=False):
            fc = synth_fc(rng, g.W['pws50'].shape[0], g.ba) if rng.random() < p_fc else None
            s = g.sample(int(i0), fc)
            if not np.isfinite(s['yd']).all(): continue
            for k in KEYS:
                acc[k].append(s[k])
            meta.append((r, g.ba, g.year, int(i0 + g.off), fc is not None))
        if r % 20 == 0:
            print(f'region {r}/{n_regions} samples {len(meta)} {time.time() - t0:.0f}s', flush=True)
    arr = {k: np.stack(v).astype(np.float16 if k.startswith('x') else np.float32) for k, v in acc.items()}
    arr['meta_region'] = np.array([m[0] for m in meta]); arr['meta_fc'] = np.array([m[4] for m in meta])
    arr['meta_ba'] = np.array([m[1] for m in meta]); arr['meta_hour'] = np.array([m[3] for m in meta])
    if out: np.savez_compressed(out, **arr)
    return arr


if __name__ == '__main__':
    # python3 sim.py <name> <seed> <n_regions> [days_per]
    name, seed, nreg = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    dp = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    os.makedirs(f'{DATA}/banks', exist_ok=True)
    bank(seed, nreg, dp, out=f'{DATA}/banks/{name}.npz')
