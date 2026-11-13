"""Parse EPA ALPHA map packages (.m files) into compact, normalised maps for roadsim.

Engine maps are stored as fuel POWER (W, = g/s x LHV of the test fuel) on a uniform grid of
(engine speed rad/s, torque per litre Nm/L), so they can be resized by displacement at constant BMEP
(the ALPHA resizing practice). Motor maps are stored as loss/P_max on a uniform grid of
(speed/w_max, torque/T_max).

Run:  python3 tools/build_maps.py   (writes roadsim/data/maps.npz and maps_meta.json)
"""
import os, re, json, glob
import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.join(HERE, '..', 'refdata', 'alpha', 'x')
OUT = os.path.join(HERE, '..', 'roadsim', 'data')
os.makedirs(OUT, exist_ok=True)


def mfield(txt, name):
    """Return numeric array for `name = [ ... ];` or `name = number;`"""
    m = re.search(re.escape(name) + r'\s*=\s*\[(.*?)\]\s*;', txt, re.S)
    if m:
        body = m.group(1)
        rows = [r for r in re.split(r'[;\n]', body) if r.strip()]
        vals = [[float(v) for v in r.replace(",", " ").split()] for r in rows]
        if all(len(v) == 1 for v in vals):
            return np.array([v[0] for v in vals])
        if len(vals) == 1:
            return np.array(vals[0])
        L = max(len(v) for v in vals)
        vals = [v for v in vals if len(v) == L]
        return np.array(vals)
    m = re.search(re.escape(name) + r'\s*=\s*([-0-9.eE+]+)\s*;', txt)
    if m:
        return float(m.group(1))
    return None


# id -> (folder glob, .m glob, LHV MJ/kg of test fuel, tech label, citation)
ENGINES = {
    'gas_na_toyota_a25a': ('2018-toyota-2.5l-a25a*', '7-*.m', 41.954, 'SI NA DI/PFI, 2018 Toyota 2.5L A25A-FKS (Tier 3 fuel)'),
    'gas_atk_toyota_tnga_hev': ('Toyota-2.5L-TNGA*', '7b-*.m', 43.3088, 'SI Atkinson hybrid engine, Toyota 2.5L TNGA prototype (2017 Vienna paper, octane-modified Tier 3)'),
    'gas_atk_toyota_tnga_hev_paper': ('Toyota-2.5L-TNGA*', '7a-*.m', 43.3088, 'SI Atkinson hybrid engine, Toyota 2.5L TNGA prototype (2017 Vienna paper image, as published)'),
    'gas_turbo_ford_27': ('2015-ford2.7l*', '7-*.m', 41.9, 'SI turbo GDI V6, 2015 Ford 2.7L EcoBoost (Tier 3)'),
    'gas_turbo_honda_15': ('2016-honda-1-5l*', '7-*.m', 41.81, 'SI turbo GDI I4, 2016 Honda 1.5L L15B7 (Tier 3)'),
    'gas_na_mazda_20': ('2014-mazda-2-0l*', '7-*.m', 42.887, 'SI NA high-CR I4, 2014 Mazda 2.0L SKYACTIV-G (Tier 2)'),
    'gas_v6_gm_43': ('2014-chevrolet-4-3l*', '7-*.m', 41.824, 'SI NA V6 with cylinder deactivation, 2014 GM 4.3L EcoTec3 (LEV III)'),
    'gas_v8_ford_73': ('GT-Power-Baseline-2020-Ford-7.3L*', '7-*.m', 41.787, 'SI NA V8 (GT-Power model), 2020 Ford 7.3L (Tier 3)'),
    'diesel_gm_30': ('GT-Power-2020-GM-3.0L-Duramax*', '7-*.m', 43.098, 'CI turbo I6 (GT-Power model), 2020 GM 3.0L Duramax'),
    'diesel_bmw_n57': ('2015-bmw-3-0-l-n57*', '7-*.m', 42.874, 'CI turbo I6, 2015 BMW 3.0L N57'),
}
MOTORS = {
    'ipm_generic_150': ('Generic-IPM-150kW*', 'Generic IPM 150 kW EDU (EPA ALPHA, 2023)'),
    'bolt_edu_150': ('2018-chevrolet-bolt*', '2018 Chevrolet Bolt 150 kW 400 V EDU (EPA ALPHA, SwRI test data)'),
    'prius_mg2_60': ('2010-Toyota-Prius-60kW*', '2010 Toyota Prius 60 kW MG2 (EPA ALPHA, ORNL test data)'),
    'prius_mg1_est': ('est-2010-toyota-prius*', 'Estimated 2010 Toyota Prius MG1 (EPA ALPHA, ORNL data)'),
    'sonata_p2_30': ('2011-Hyundai-Sonata-30kW*', '2011 Hyundai Sonata 30 kW P2 motor (EPA ALPHA, ORNL data)'),
}

NW, NT = 48, 48  # engine grid
meta = {'engines': {}, 'motors': {}}
arrays = {}

for eid, (fg, mg, lhv, label) in ENGINES.items():
    d = glob.glob(os.path.join(X, fg))[0]
    mf = glob.glob(os.path.join(d, mg))[0]
    t = open(mf, errors='ignore').read()
    disp = mfield(t, 'engine.displacement_L')
    ci = 'compression_ignition' in t
    ws = mfield(t, 'engine.fuel_map_speed_radps'); ts = mfield(t, 'engine.fuel_map_torque_Nm')
    fm = mfield(t, 'engine.fuel_map_gps')
    if fm.shape != (len(ts), len(ws)):
        fm = fm.T
    assert fm.shape == (len(ts), len(ws)), (eid, fm.shape, len(ts), len(ws))
    wft = mfield(t, 'engine.full_throttle_speed_radps'); tft = mfield(t, 'engine.full_throttle_torque_Nm')
    wct = mfield(t, 'engine.closed_throttle_speed_radps'); tct = mfield(t, 'engine.closed_throttle_torque_Nm')
    m = re.search(r'idle_(?:target_)?speed_radps\.table\s*=\s*\[\s*([-0-9.eE+]+)', t)
    idle = float(m.group(1)) if m else 650 * np.pi / 30
    # monotone axes
    ows = np.argsort(ws); ots = np.argsort(ts)
    ws, ts, fm = ws[ows], ts[ots], fm[np.ix_(ots, ows)]
    ws_u, iu = np.unique(ws, return_index=True); fm = fm[:, iu]
    ts_u, it = np.unique(ts, return_index=True); fm = fm[it, :]
    interp = RegularGridInterpolator((ts_u, ws_u), fm, bounds_error=False, fill_value=None)
    wmax = float(wft.max())
    tmax = float(tft.max())
    tmin = float(tct.min())
    wg = np.linspace(0, wmax, NW)
    tg = np.linspace(tmin * 1.05, tmax * 1.05, NT)
    TT, WW = np.meshgrid(tg, wg, indexing='ij')
    F = interp(np.stack([TT.ravel(), WW.ravel()], -1)).reshape(TT.shape)
    F = np.maximum(F, 0.0)
    P_fuel = F * lhv * 1e3  # W
    # peak brake efficiency on the WOT envelope region
    with np.errstate(divide='ignore', invalid='ignore'):
        eff = np.where(P_fuel > 0, TT * WW / P_fuel, 0)
    tw = np.interp(wg, wft, tft)
    eff = np.where(TT <= tw[None, :], eff, 0)
    pk = float(np.nanmax(eff[:, wg > 100]))
    pmax = float(np.max(np.interp(wg, wft, tft) * wg))
    key = 'eng_' + eid
    arrays[key + '_fuelW_per_L'] = (P_fuel / disp).astype(np.float32)   # W per litre, on (T/L, w) grid
    arrays[key + '_wft'] = wft.astype(np.float32); arrays[key + '_tft_per_L'] = (tft / disp).astype(np.float32)
    arrays[key + '_wct'] = wct.astype(np.float32); arrays[key + '_tct_per_L'] = (tct / disp).astype(np.float32)
    meta['engines'][eid] = dict(label=label, displacement_L=float(disp), fuel='diesel' if ci else 'gasoline',
                                lhv_MJkg_testfuel=lhv, w_max=wmax, t_grid_per_L=[float(tg[0] / disp), float(tg[-1] / disp), NT],
                                w_grid=[0.0, wmax, NW], idle_radps=idle, peak_bte=round(pk, 4),
                                rated_power_kW=round(pmax / 1e3, 1), peak_torque_Nm=round(tmax, 1),
                                source='US EPA NVFEL/NCAT ALPHA map package: ' + os.path.basename(d))
    print(f'{eid:28s} disp={disp:.2f}L Pmax={pmax/1e3:6.1f}kW Tmax={tmax:6.1f} peakBTE={pk:.3f} idle={idle*30/np.pi:.0f}rpm')

NWm, NTm = 41, 61
for mid, (fg, label) in MOTORS.items():
    d = glob.glob(os.path.join(X, fg))[0]
    mf = glob.glob(os.path.join(d, '7*.m'))[0]
    t = open(mf, errors='ignore').read()
    wmax = mfield(t, 'mg.max_speed_radps'); tmax = mfield(t, 'mg.max_torque_Nm'); pmax = mfield(t, 'mg.max_motor_power_W')
    ws = mfield(t, 'mg.electric_power_W.axis_1.breakpoints'); ts = mfield(t, 'mg.electric_power_W.axis_2.breakpoints')
    P = mfield(t, 'mg.electric_power_W.table')
    if P.ndim == 1:
        P = P.reshape(len(ws), len(ts))
    if P.shape != (len(ws), len(ts)):
        P = P.T
    if ws.ndim > 1: ws = ws.ravel()
    if ts.ndim > 1: ts = ts.ravel()
    wl = mfield(t, 'mg.positive_torque_limit_Nm.axis_1.breakpoints'); tl = mfield(t, 'mg.positive_torque_limit_Nm.table')
    wl = np.ravel(wl); tl = np.ravel(tl)
    if tl.size != wl.size:
        tl = tl[:wl.size]
    sel = wl >= 0
    wl, tl = wl[sel], tl[sel]
    W, T = np.meshgrid(ws, ts, indexing='ij')
    loss = P - W * T
    interp = RegularGridInterpolator((ws, ts), loss, bounds_error=False, fill_value=None)
    wg = np.linspace(0, 1, NWm) * wmax
    tg = np.linspace(-1, 1, NTm) * tmax
    WW, TT = np.meshgrid(wg, tg, indexing='ij')
    L = interp(np.stack([np.clip(WW, ws.min(), ws.max()).ravel(), np.clip(TT, ts.min(), ts.max()).ravel()], -1)).reshape(WW.shape)
    L = np.maximum(L, 0.0)
    # efficiency summary inside the envelope
    tlim = np.interp(wg, wl, tl)
    inside = (np.abs(TT) <= tlim[:, None]) & (WW > 0.1 * wmax) & (np.abs(TT) > 0.1 * tmax)
    pm = np.abs(WW * TT)
    eff_m = np.where(inside & (TT > 0), pm / (pm + L + 1e-9), np.nan)
    key = 'mot_' + mid
    arrays[key + '_loss_norm'] = (L / pmax).astype(np.float32)   # on (w/wmax in [0,1], T/Tmax in [-1,1])
    arrays[key + '_wlim_norm'] = (wl / wmax).astype(np.float32); arrays[key + '_tlim_norm'] = (tl / tmax).astype(np.float32)
    meta['motors'][mid] = dict(label=label, w_max=float(wmax), T_max=float(tmax), P_max_W=float(pmax), grid=[NWm, NTm],
                               peak_eff=round(float(np.nanmax(eff_m)), 4), median_eff_envelope=round(float(np.nanmedian(eff_m)), 4),
                               source='US EPA NVFEL/NCAT ALPHA e-machine map package: ' + os.path.basename(d))
    print(f'{mid:20s} wmax={wmax:.0f} Tmax={tmax:.0f} Pmax={pmax/1e3:.0f}kW peak_eff={np.nanmax(eff_m):.3f} med={np.nanmedian(eff_m):.3f}')

np.savez_compressed(os.path.join(OUT, 'maps.npz'), **arrays)
json.dump(meta, open(os.path.join(OUT, 'maps_meta.json'), 'w'), indent=1)
print('wrote', os.path.join(OUT, 'maps.npz'))
