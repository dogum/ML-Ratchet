"""behave(): BehavePlus-like point fire behaviour in SI units."""
import math

import numpy as np

from . import rothermel as R
from .crown import crown_fire, van_wagner_i0, r_active_critical
from .fuels import FM, _build_arrays, number_of
from .spotting import max_spot_distance_m, p_ignition
from .weather import U10_TO_U20


def behave(fuel, m1=6.0, m10=7.0, m100=8.0, mherb=60.0, mwoody=90.0, wind_speed=0.0, wind_height='10m',
           wind_dir=None, slope=0.0, aspect=0.0, waf=None, cc=0.0, ch=0.0, cbh=0.0, cbd=0.0, fmc=100.0,
           curing=None, wind_limit='none', lw='behaveplus', lw_max=8.0, crown='van_wagner',
           crown_spread='rothermel1991', cfb_method='van_wagner1993', elapsed_h=1.0, t_air=25.0, shade=0.5,
           cover_height=None, custom_fuels=None, english=False):
    """Point fire behaviour (Rothermel 1972 + Albini 1976, BehavePlus spread directions, crown fire, spotting).

    fuel: number (e.g. 102) or code ('GR2', 'FM10'); moistures in percent; wind_speed m/s at `wind_height`
    ('10m' contract wind, '20ft', or 'midflame'); wind_dir degrees the wind blows FROM (None = upslope wind);
    slope degrees; aspect degrees (direction the slope faces). waf: None = Andrews (2012) from fuel depth and canopy.
    Returns a dict (SI units; '_ft'/'_btu' keys hold the English values when english=True).
    """
    fm = FM if not custom_fuels else _build_arrays(custom_fuels)
    num = number_of(fuel)
    row = int(fm['lut'][num]) if num < len(fm['lut']) else -1
    out = dict(fuel=num, burnable=row >= 0)
    if row < 0:
        out.update(ros_mmin=0.0, fli_kwm=0.0, flame_length_m=0.0, fire_type=0)
        return out
    m = np.array([m1, m10, m100, m1, mherb, mwoody], float) / 100.0
    bd = np.zeros(R.NBUNDLE)
    R.bundle_one(fm['w0'][row], fm['sav'][row], fm['heat'][row], fm['depth'][row], fm['mx'][row],
                 fm['dynamic'][row], m, -1.0 if curing is None else float(curing), bd)
    # winds
    if wind_height == '10m':
        u20 = wind_speed * U10_TO_U20
    elif wind_height == '20ft':
        u20 = wind_speed
    elif wind_height == 'midflame':
        u20 = None
    else:
        raise ValueError(wind_height)
    if waf is None:
        w, sh = R.waf(np.array(fm['depth'][row]), np.array(cc), np.array(ch), np.array(cbh))
        waf_v = float(w)
        sheltered = bool(sh)
    else:
        waf_v = float(waf)
        sheltered = False
    if u20 is None:
        umid = wind_speed
        u20 = umid / waf_v if waf_v > 0 else 0.0
    else:
        umid = u20 * waf_v
    upslope = (aspect + 180.0) % 360.0
    wind_to = upslope if wind_dir is None else (wind_dir + 180.0) % 360.0
    wrel = math.radians(wind_to - upslope)
    wl = {'none': R.WL_NONE, 'rothermel': R.WL_ROTHERMEL, 'andrews2013': R.WL_ANDREWS}[wind_limit]
    lwm = {'behaveplus': R.LW_BEHAVE, 'anderson1983': R.LW_ANDERSON}[lw]
    c = np.zeros(8)
    tan_s = math.tan(math.radians(slope))
    R.combine(bd, umid * R.MS_TO_FTMIN, wrel, tan_s, wl, lwm, lw_max, c)
    rh_ft = c[0]
    e = c[4]
    z = c[3]
    rb_ft = rh_ft * (1 - e) / (1 + e)
    ha_btu = bd[R.B_HA]
    fli_btu = ha_btu * rh_ft / 60.0
    fl_ft = R.byram_flame_length_ft(fli_btu)
    ros = rh_ft * R.FTMIN_TO_MMIN
    rb = rb_ft * R.FTMIN_TO_MMIN
    L = (ros + rb) * elapsed_h * 60.0
    W = L / z
    rf = W / 2.0 / (elapsed_h * 60.0) if elapsed_h > 0 else 0.0
    a_, b_ = L / 2.0, W / 2.0
    perim = math.pi * (3 * (a_ + b_) - math.sqrt((3 * a_ + b_) * (a_ + 3 * b_))) if a_ > 0 else 0.0
    hpa = ha_btu * R.BTUFT2_TO_KJM2
    fli = fli_btu * R.BTUFTS_TO_KWM
    out.update(
        ros_mmin=ros, ros_back_mmin=rb, ros_flank_mmin=rf, direction_max_deg=(upslope + math.degrees(c[1])) % 360.0,
        lw_ratio=z, eccentricity=e, effective_wind_ms=c[2] * R.FTMIN_TO_MS, wind_limited=bool(c[7]),
        phi_w=c[5], phi_s=c[6], r0_mmin=bd[R.B_R0] * R.FTMIN_TO_MMIN,
        reaction_intensity_kwm2=bd[R.B_IR] * R.BTUFT2MIN_TO_KWM2,
        ir_dead_kwm2=bd[R.B_IRD] * R.BTUFT2MIN_TO_KWM2, ir_live_kwm2=bd[R.B_IRL] * R.BTUFT2MIN_TO_KWM2,
        hpa_kjm2=hpa, residence_time_min=bd[R.B_TR], fli_kwm=fli, flame_length_m=fl_ft * 0.3048,
        mx_live_pct=bd[R.B_MXL] * 100, mx_dead_pct=fm['mx'][row] * 100, sigma_ft=bd[R.B_SIGMA],
        packing_ratio=bd[R.B_BETA], relative_packing_ratio=bd[R.B_BRATIO], bulk_density_lbft3=bd[R.B_RHOB],
        bulk_density_kgm3=bd[R.B_RHOB] * 16.0185, curing=bd[R.B_CURED], char_dead_moisture_pct=bd[R.B_MD] * 100,
        char_live_moisture_pct=bd[R.B_ML] * 100, heat_sink_btuft3=bd[R.B_SINK], xi=bd[R.B_XI],
        waf=waf_v, sheltered=sheltered, u20_ms=u20, u_mid_ms=umid,
        wind_factor_C=bd[R.B_C], wind_factor_B=bd[R.B_B], wind_factor_E=bd[R.B_E],
        ellipse_length_m=L, ellipse_width_m=W, area_ha=math.pi * a_ * b_ / 1e4, perimeter_m=perim,
        fire_type=1 if ros > 0 else 0)
    if english:
        out.update(ros_ftmin=rh_ft, fli_btufts=fli_btu, flame_length_ft=fl_ft, hpa_btuft2=ha_btu,
                   ir_btuft2min=bd[R.B_IR], r0_ftmin=bd[R.B_R0], effective_wind_mph=c[2] / 88.0)
    # crown fire
    if crown not in (None, 'none', False) and cbd > 0 and ch > cbh:
        r10 = _fm10_ros(m, u20, wrel, tan_s, wl, lwm, lw_max, curing)
        r_sa = None
        if cfb_method == 'scott_reinhardt':
            r_sa = _r_sa(fm, row, m, wrel, tan_s, wl, lwm, lw_max, curing, waf_v, cbd, crown_spread, m1, cbh)
        cf = crown_fire(dict(ros_mmin=ros, hpa_kjm2=hpa, fli_kwm=fli), cbh, cbd, ch, fmc, u20_ms=u20,
                        u10_ms=u20 / U10_TO_U20, m1_pct=m1, r10_mmin=r10, method=crown_spread,
                        cfb_method=cfb_method, r_sa_mmin=r_sa)
        out['crown'] = cf
        out['fire_type'] = cf['fire_type']
        out['ros_final_mmin'] = cf['ros_mmin']
        out['fli_final_kwm'] = cf['fli_kwm']
        out['flame_length_final_m'] = cf['flame_length_m'] if cf['fire_type'] >= 2 else out['flame_length_m']
        out['torching_index_ms'], out['crowning_index_ms'] = _indices(fm, row, m, wrel, tan_s, wl, lwm, lw_max,
                                                                      curing, waf_v, cbh, cbd, fmc, crown_spread, m1)
    else:
        out['ros_final_mmin'] = ros
        out['fli_final_kwm'] = fli
        out['flame_length_final_m'] = out['flame_length_m']
    # spotting
    hc = cover_height if cover_height is not None else max(ch, 1.0)
    out['spot_max_surface_m'] = max_spot_distance_m(fli, u20, hc, cbd <= 0)
    if out['fire_type'] >= 2:
        out['spot_max_crown_m'] = max_spot_distance_m(out['fli_final_kwm'], u20, hc, False)
    out['p_ignition'] = p_ignition(m1 / 100.0, t_air, shade)
    return out


def _fm10_ros(m, u20, wrel, tan_s, wl, lwm, lw_max, curing):
    r10 = int(FM['lut'][10])
    bd = np.zeros(R.NBUNDLE)
    R.bundle_one(FM['w0'][r10], FM['sav'][r10], FM['heat'][r10], FM['depth'][r10], FM['mx'][r10],
                 FM['dynamic'][r10], m, -1.0 if curing is None else float(curing), bd)
    c = np.zeros(8)
    R.combine(bd, 0.4 * u20 * R.MS_TO_FTMIN, wrel, tan_s, wl, lwm, lw_max, c)
    return c[0] * R.FTMIN_TO_MMIN


def _surface_at(fm, row, m, u20, wrel, tan_s, wl, lwm, lw_max, curing, waf_v):
    bd = np.zeros(R.NBUNDLE)
    R.bundle_one(fm['w0'][row], fm['sav'][row], fm['heat'][row], fm['depth'][row], fm['mx'][row],
                 fm['dynamic'][row], m, -1.0 if curing is None else float(curing), bd)
    c = np.zeros(8)
    R.combine(bd, u20 * waf_v * R.MS_TO_FTMIN, wrel, tan_s, wl, lwm, lw_max, c)
    return c[0] * R.FTMIN_TO_MMIN, bd[R.B_HA] * R.BTUFT2_TO_KJM2


def _crown_active_ros(u20, m, wrel, tan_s, wl, lwm, lw_max, curing, cbd, crown_spread, m1):
    if crown_spread == 'cruz2005':
        from .crown import cruz2005_active
        return cruz2005_active(u20 / U10_TO_U20 * 3.6, cbd, m1)
    return 3.34 * _fm10_ros(m, u20, wrel, tan_s, wl, lwm, lw_max, curing)


def _bisect(f, lo, hi, it=60):
    flo = f(lo)
    if flo >= 0:
        return lo
    if f(hi) < 0:
        return np.inf
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if f(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return hi


def _indices(fm, row, m, wrel, tan_s, wl, lwm, lw_max, curing, waf_v, cbh, cbd, fmc, crown_spread, m1):
    """Torching index and crowning index (20-ft wind, m/s; Scott & Reinhardt 2001) by bisection."""
    i0 = van_wagner_i0(cbh, fmc)

    def ft(u):
        r, h = _surface_at(fm, row, m, u, wrel, tan_s, wl, lwm, lw_max, curing, waf_v)
        return h * r / 60.0 - i0
    ti = _bisect(ft, 0.0, 150.0)
    rac = r_active_critical(cbd)

    def fc(u):
        return _crown_active_ros(u, m, wrel, tan_s, wl, lwm, lw_max, curing, cbd, crown_spread, m1) - rac
    ci = _bisect(fc, 0.0, 150.0)
    return ti, ci


def _r_sa(fm, row, m, wrel, tan_s, wl, lwm, lw_max, curing, waf_v, cbd, crown_spread, m1, cbh):
    ci = _indices(fm, row, m, wrel, tan_s, wl, lwm, lw_max, curing, waf_v, cbh, cbd, 100.0, crown_spread, m1)[1]
    if not np.isfinite(ci):
        return np.inf
    return _surface_at(fm, row, m, ci, wrel, tan_s, wl, lwm, lw_max, curing, waf_v)[0]
