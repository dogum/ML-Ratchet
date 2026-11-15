"""Wind and solar fleet physics for one BA (real fleet or a randomised one), vectorised over hours.

Wind (per farm group: weather point x hub height x specific power):
  shear      alpha = ln(ws50/ws10)/ln(50/10) from the point's own 10 m and 50 m winds (clipped 0-0.6)
  hub wind   v = ws50 (hub/50)^alpha
  density    moist-air rho(T, p, RH) (gridsim.renewables.air_density); IEC density-equivalent v_eq = v (rho/1.225)^(1/3)
  farm curve gridsim pitch-regulated turbine curve from specific power (W/m2), Gaussian spread across the farm
             (tabulated here on a 0.05 m/s grid for speed)
  losses     wake, availability, electrical (defaults 10 %, 97 %, 2 % as GridSim), cold shutdown < -30 C, cut-out 25 m/s
Solar: per weather point pvlib capacity factors (wxprep.py: single-axis tracking DC/AC 1.3, fixed 25 deg DC/AC 1.2) x
  the online MW of each type at that point.
Capacity online follows each generator's EIA-860 online / retirement dates (month resolution).
"""
import numpy as np, pandas as pd
from common import *
from gridsim.renewables import farm_curve, air_density

VG = np.arange(0, 40.0001, 0.05)
SPG = np.arange(150, 476, 25.0)
_TAB = None


def table():
    global _TAB
    if _TAB is None:
        _TAB = np.stack([farm_curve(VG, sp) for sp in SPG])      # [n_sp, n_v]
    return _TAB


def farm_cf(v_eq, sp):
    """Farm power curve (0-1, before losses) at density-equivalent speed v_eq for specific power sp (interp in both)."""
    t = table()
    j = np.clip(np.searchsorted(SPG, sp) - 1, 0, len(SPG) - 2)
    w = np.clip((sp - SPG[j]) / (SPG[j + 1] - SPG[j]), 0, 1)
    x = np.clip(v_eq / 0.05, 0, len(VG) - 1.001)
    i = x.astype(int); f = x - i
    a = t[j, i] * (1 - f) + t[j, i + 1] * f
    b = t[j + 1, i] * (1 - f) + t[j + 1, i + 1] * f
    return a * (1 - w) + b * w


def online_matrix(fl, hours=HOURS):
    """[n_farms, n_hours] bool: online and not retired."""
    tn = hours.tz_localize(None).values
    on = np.ones((len(fl), len(hours)), bool)
    for k, (o, r) in enumerate(zip(fl.online.values, fl.retired.values)):
        if not pd.isna(o): on[k] &= tn >= o
        if not pd.isna(r): on[k] &= tn < r
    return on


def wind_groups(ba):
    fl = pd.read_parquet(f'{DATA}/fleet/{ba}_wind.parquet')
    fl['hub_r'] = (fl.hub / 10).round() * 10
    fl['sp_r'] = (fl.sp / 25).round() * 25
    on = online_matrix(fl)
    keys = fl.groupby(['pt', 'hub_r', 'sp_r']).indices
    G = []
    cap = []
    for (pt, hub, sp), ix in keys.items():
        G.append((int(pt), float(hub), float(sp)))
        cap.append((fl.MW.values[ix, None] * on[ix]).sum(0))
    return np.array(G), np.array(cap, np.float32)          # groups [G,3], capacity [G, NH]




def wind_mw(W, groups, cap, sl=slice(None), vmult=None, wake=0.10, avail=0.97, elec=0.02, sp_shift=0.0,
            hub_shift=0.0, group_scale=None):
    """Total wind MW over hours `sl` and the per-group CF. groups [G,3] (pt, hub, sp); cap [G, NH]."""
    ws10 = W['pws10'][:, sl].astype(float); ws50 = W['pws50'][:, sl].astype(float)
    if vmult is not None:
        ws10 = ws10 * vmult; ws50 = ws50 * vmult
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = np.clip(np.nan_to_num(np.log(np.maximum(ws50, 0.1) / np.maximum(ws10, 0.1)) / np.log(5.0), nan=1 / 7), 0, 0.6)
    rho = air_density(W['pT'][:, sl], W['pps'][:, sl] * 10.0, rh=W['pRH'][:, sl] / 100.0)
    tot = np.zeros(ws50.shape[1])
    for g, (pt, hub, sp) in enumerate(groups):
        pt = int(pt)
        v = ws50[pt] * ((hub + hub_shift) / 50.0) ** alpha[pt]
        veq = v * (rho[pt] / 1.225) ** (1 / 3)
        cf = farm_cf(veq, sp + sp_shift) * (1 - wake) * avail * (1 - elec)
        cf = np.where(W['pT'][pt, sl] < -30, 0.0, cf)
        s = 1.0 if group_scale is None else group_scale[g]
        tot += cf * cap[g, sl] * s
    return tot


def solar_groups(ba):
    fl = pd.read_parquet(f'{DATA}/fleet/{ba}_solar.parquet')
    on = online_matrix(fl)
    keys = fl.groupby(['pt', 'track']).indices
    G, cap = [], []
    for (pt, tr), ix in keys.items():
        G.append((int(pt), int(tr)))
        cap.append((fl.MW.values[ix, None] * on[ix]).sum(0))
    return np.array(G), np.array(cap, np.float32)


def solar_mw(W, groups, cap, sl=slice(None), cfmult=None, track_share=None, group_scale=None):
    """Total solar MW: sum over (point, tracking) groups of CF x online MW. cfmult [n_pts, len] cloud error (truth);
    track_share overrides each group's tracking type with a mix (simulated fleets)."""
    tot = 0.0
    for g, (pt, tr) in enumerate(groups):
        pt = int(pt)
        if track_share is None:
            cf = (W['pcf_track'] if tr else W['pcf_fixed'])[pt, sl].astype(float)
        else:
            cf = track_share * W['pcf_track'][pt, sl] + (1 - track_share) * W['pcf_fixed'][pt, sl]
        if cfmult is not None:
            cf = np.minimum(cf * cfmult[pt], 1.0)
        s = 1.0 if group_scale is None else group_scale[g]
        tot = tot + cf * cap[g, sl] * s
    return np.asarray(tot, float)


def real_priors(ba):
    """Nominal-physics wind and solar MW for the real fleet, 2018-2025 (no calibration)."""
    W = load_bundle(ba)
    gw, cw = wind_groups(ba); gs_, cs = solar_groups(ba)
    return dict(wind=wind_mw(W, gw, cw), wind_cap=cw.sum(0), solar=solar_mw(W, gs_, cs), solar_cap=cs.sum(0))


if __name__ == '__main__':
    import sys, os
    os.makedirs(f'{DATA}/prior', exist_ok=True)
    for ba in (sys.argv[1:] or BAS):
        r = real_priors(ba)
        np.savez_compressed(f'{DATA}/prior/{ba}_renew.npz', **{k: v.astype(np.float32) for k, v in r.items()})
        d = gd.eia930(ba, columns=['ng_WND', 'ng_SUN']).reindex(HOURS)
        for y in range(2018, 2026):
            m = HOURS.year == y
            print(ba, y, f"wind real/phys {np.nanmean(d.ng_WND.values[m]) / r['wind'][m].mean():.2f} cap {r['wind_cap'][m].mean():.0f}"
                  f"  solar real/phys {np.nanmean(d.ng_SUN.values[m]) / max(r['solar'][m].mean(), 1e-6):.2f} cap {r['solar_cap'][m].mean():.0f}", flush=True)
