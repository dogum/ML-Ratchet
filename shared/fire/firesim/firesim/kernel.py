"""Minimum-travel-time (MTT) arrival-time solver (numba).

Fire spreads from burning cells to cells within a stencil (8, 16 or 32 neighbours) along straight segments. The time
to cross a segment is sum_j frac_j * d / R_j(theta), where the segment passes through cells j for a fraction frac_j of
its length d and R_j(theta) is the spread rate in cell j in the segment direction theta. R_j(theta) is the BehavePlus
ellipse ray speed from the rear focus, R_H (1 - e) / (1 - e cos(theta - az)), after crown-fire adjustment. Arrival
times are found with Dijkstra's algorithm (time-dependent: an edge is integrated through the hourly weather steps,
each with its own spread rates). This is Finney's (2002) MTT idea on a regular grid with a fixed stencil.

Per-cell state at a weather step k (cached for the latest step):
  0 head surface ROS (m/min, after adjustments)   1 heading azimuth (rad, clockwise from north)
  2 surface eccentricity                           3 heat per unit area HPA (kJ/m2)
  4 crown initiation intensity I0 (kW/m; inf if no canopy)   5 critical surface ROS for initiation (m/min)
  6 crown head ROS (m/min)                         7 crown eccentricity
  8 CAC (crown head ROS / (3/CBD))                 9 canopy fuel load (kg/m2)
  10 phi_w   11 phi_s   12 effective wind (m/s)    13 L/W surface   14 reaction intensity (kW/m2)
"""
import math

import numpy as np

from ._nb import njit
from . import rothermel as R
from .crown import van_wagner_i0, cruz2005_active
from .spotting import max_spot_distance_m, p_ignition

NS = 15
NLA = 8      # look-ahead cache slots (weather steps beyond the current one)
NLAS = 9     # state fields dir_ros needs (0-8), stored as float32 in the look-ahead cache
INF = 1e30


@njit(cache=True)
def _heap_push(hk, hv, size, key, val):
    n = size[0]
    if n >= hk.shape[0]:
        return False
    i = n
    hk[i] = key
    hv[i] = val
    size[0] = n + 1
    while i > 0:
        p = (i - 1) >> 1
        if hk[p] <= hk[i]:
            break
        tk = hk[p]
        hk[p] = hk[i]
        hk[i] = tk
        tv = hv[p]
        hv[p] = hv[i]
        hv[i] = tv
        i = p
    return True


@njit(cache=True)
def _heap_pop(hk, hv, size):
    n = size[0]
    key = hk[0]
    val = hv[0]
    n -= 1
    size[0] = n
    if n > 0:
        hk[0] = hk[n]
        hv[0] = hv[n]
        i = 0
        while True:
            l = 2 * i + 1
            if l >= n:
                break
            r = l + 1
            m = l
            if r < n and hk[r] < hk[l]:
                m = r
            if hk[i] <= hk[m]:
                break
            tk = hk[m]
            hk[m] = hk[i]
            hk[i] = tk
            tv = hv[m]
            hv[m] = hv[i]
            hv[i] = tv
            i = m
    return key, val


@njit(cache=True)
def cell_state(c, k, fuel_row, zone, slope_tan, upslope_az, waf, cbh, cbd, ch, adj,
               bundles, bundle10, u20_ftmin, wind_to, u10_kmh, m1_pct, fmc, crown_mode,
               wl_mode, lw_mode, lw_max, crown_ros_mult, st, tmp_bd):
    for i in range(NS):
        st[i] = 0.0
    st[4] = INF
    st[5] = INF
    st[13] = 1.0
    row = fuel_row[c]
    if row < 0:
        return
    z = zone[c]
    bd = bundles[k, z, row]
    out = tmp_bd
    u = waf[c] * u20_ftmin[k, z]
    wrel = wind_to[k, z] - upslope_az[c]
    R.combine(bd, u, wrel, slope_tan[c], wl_mode, lw_mode, lw_max, out)
    a = adj[c]
    st[0] = out[0] * R.FTMIN_TO_MMIN * a
    st[1] = upslope_az[c] + out[1]
    st[2] = out[4]
    st[3] = bd[R.B_HA] * R.BTUFT2_TO_KJM2
    st[10] = out[5]
    st[11] = out[6]
    st[12] = out[2] * R.FTMIN_TO_MS
    st[13] = out[3]
    st[14] = bd[R.B_IR] * R.BTUFT2MIN_TO_KWM2
    if crown_mode > 0 and cbd[c] > 0.0 and ch[c] > cbh[c] and st[0] > 0.0:
        i0 = van_wagner_i0(cbh[c], fmc)
        st[4] = i0
        st[5] = 60.0 * i0 / st[3] if st[3] > 0.0 else INF
        if crown_mode == 1:
            R.combine(bundle10[k, z], 0.4 * u20_ftmin[k, z], wrel, slope_tan[c], wl_mode, lw_mode, lw_max, out)
            rc = 3.34 * out[0] * R.FTMIN_TO_MMIN
        else:
            rc = cruz2005_active(u10_kmh[k, z], cbd[c], m1_pct[k, z])
        rc *= crown_ros_mult * a
        st[6] = rc
        z20 = 1.0 + 0.125 * u20_ftmin[k, z] / 88.0
        st[7] = math.sqrt(z20 * z20 - 1.0) / z20
        st[8] = rc * cbd[c] / 3.0
        st[9] = cbd[c] * (ch[c] - cbh[c])


@njit(cache=True)
def dir_ros(st, theta, crown_mode, res3):
    """Directional ROS (m/min) with crown adjustment. res3: [ros, fire_type, cfb]."""
    rh = st[0]
    if rh <= 0.0:
        res3[0] = 0.0
        res3[1] = 0.0
        res3[2] = 0.0
        return
    g = theta - st[1]
    e = st[2]
    rs = rh * (1.0 - e) / (1.0 - e * math.cos(g))
    res3[0] = rs
    res3[1] = 1.0
    res3[2] = 0.0
    if crown_mode == 0 or st[4] >= INF:
        return
    isurf = st[3] * rs / 60.0
    if isurf <= st[4]:
        return
    ec = st[7]
    rc = st[6] * (1.0 - ec) / (1.0 - ec * math.cos(g))
    cfb = 1.0 - math.exp(-0.23 * (rs - st[5]))
    if cfb < 0.0:
        cfb = 0.0
    if st[8] >= 1.0:
        res3[1] = 3.0
        res3[2] = 1.0
        res3[0] = rc if rc > rs else rs
    else:
        res3[1] = 2.0
        res3[2] = cfb
        if crown_mode == 2:
            rp = rc * math.exp(-st[8])
            res3[0] = rp if rp > rs else rs
        elif rc > rs:
            res3[0] = rs + cfb * (rc - rs)


@njit(cache=True)
def _load_state(cs, kk, k, fuel_row, zone, slope_tan, upslope_az, waf, cbh, cbd, ch, adj, bundles, bundle10,
                u20_ftmin, wind_to, u10_kmh, m1_pct, fmc, crown_mode, wl_mode, lw_mode, lw_max, crown_ros_mult,
                cache, cache_k, cache_la, cache_la_k, nla, out, st_full, tmp_bd):
    """Fields 0-8 of cell cs's state at weather step kk into out (what dir_ros reads). Step k (the popped cell's
    step) uses the float64 cache; any other step the float32 look-ahead cache, as the p1 segment code does."""
    if kk == k:
        if cache_k[cs] != k:
            cell_state(cs, k, fuel_row, zone, slope_tan, upslope_az, waf, cbh, cbd, ch, adj, bundles, bundle10,
                       u20_ftmin, wind_to, u10_kmh, m1_pct, fmc, crown_mode, wl_mode, lw_mode, lw_max,
                       crown_ros_mult, cache[cs], tmp_bd)
            cache_k[cs] = k
        for q_ in range(NLAS):
            out[q_] = cache[cs, q_]
    else:
        sl = kk % nla
        if cache_la_k[cs, sl] != kk:
            cell_state(cs, kk, fuel_row, zone, slope_tan, upslope_az, waf, cbh, cbd, ch, adj, bundles, bundle10,
                       u20_ftmin, wind_to, u10_kmh, m1_pct, fmc, crown_mode, wl_mode, lw_mode, lw_max,
                       crown_ros_mult, st_full, tmp_bd)
            for q_ in range(NLAS):
                cache_la[cs, sl, q_] = st_full[q_]
            cache_la_k[cs, sl] = kk
        for q_ in range(NLAS):
            out[q_] = cache_la[cs, sl, q_]


@njit(cache=True)
def _side_blocked(c, fuel_row, t_block, tt):
    return fuel_row[c] < 0 or t_block[c] <= tt


@njit(cache=True)
def run_mtt(ny, nx, res_m, fuel_row, zone, slope_tan, upslope_az, waf, cbh, cbd, ch, adj, t_block,
            bundles, bundle10, u20_ftmin, wind_to, u10_kmh, m1_pct, tair, fmc, dt_w,
            crown_mode, wl_mode, lw_mode, lw_max, crown_ros_mult,
            off_dy, off_dx, off_len, seg_start, seg_n, seg_dy, seg_dx, seg_frac, corner_a, corner_b,
            opp, nb_prev, nb_next, simplex, p2, corner_j,
            ign_cells, ign_times, t_end,
            spot_on, spot_seed, ember_rate, spot_med_frac, spot_sigma, spot_dir_sd, spot_p_est, spot_delay_h,
            spot_surface_fli, shade,
            ia_on, ia_start, ia_rate, ia_lw,
            max_spots):
    n = ny * nx
    arrival = np.full(n, np.inf)
    done = np.zeros(n, np.bool_)
    arr_dir = np.full(n, np.nan)
    ftype = np.zeros(n, np.int8)
    cfb_o = np.zeros(n, np.float32)
    rosd = np.zeros(n, np.float32)
    roshead = np.zeros(n, np.float32)
    hpa_o = np.zeros(n, np.float32)
    phiw_o = np.zeros(n, np.float32)
    phis_o = np.zeros(n, np.float32)
    lw_o = np.zeros(n, np.float32)
    i0_o = np.full(n, np.inf, np.float32)
    rc_o = np.zeros(n, np.float32)
    cfl_o = np.zeros(n, np.float32)
    src = np.zeros(n, np.int8)
    cap = n // 4 + 4096 + ign_cells.shape[0]       # grows by doubling when needed
    hk = np.empty(cap)
    hv = np.empty(cap, np.int64)
    size = np.zeros(1, np.int64)
    cache_k = np.full(n, -1, np.int64)
    cache = np.zeros((n, NS))
    # look-ahead cache for later weather steps (segment traversals that cross hour boundaries): slot kk % NLA
    nla = NLA if n <= 400000 else 4                 # fewer slots on big grids (memory)
    cache_la_k = np.full((n, nla), -1, np.int32)
    cache_la = np.zeros((n, nla, NLAS), np.float32)
    tmp_bd = np.zeros(8)
    st_tmp = np.zeros(NS)
    st_a = np.zeros(NS)      # p2 state buffers (segment walk; simplex v, u, w)
    st_b = np.zeros(NS)
    st_c = np.zeros(NS)
    st_full = np.zeros(NS)
    res3 = np.zeros(3)
    nk = bundles.shape[0]
    K = off_dy.shape[0]
    # spotting records: t_src, src, dst, dist_m, t_ign, ignited
    sp = np.zeros((max_spots, 6))
    nsp = 0
    if spot_on:
        np.random.seed(spot_seed)
    overflow = False
    for i in range(ign_cells.shape[0]):
        c = ign_cells[i]
        if fuel_row[c] >= 0 and ign_times[i] < arrival[c]:
            arrival[c] = ign_times[i]
            src[c] = 1
            _heap_push(hk, hv, size, ign_times[i], c)
    cell_area = res_m * res_m
    burned = 0
    contained_t = np.inf
    while size[0] > 0:
        if size[0] + K + 1024 >= hk.shape[0]:
            hk2 = np.empty(2 * hk.shape[0])
            hv2 = np.empty(2 * hk.shape[0], np.int64)
            hk2[:size[0]] = hk[:size[0]]
            hv2[:size[0]] = hv[:size[0]]
            hk = hk2
            hv = hv2
        t, c = _heap_pop(hk, hv, size)
        if done[c] or t > arrival[c]:
            continue
        if t > t_end:
            break
        if ia_on:
            # Fried & Fried (1996)-style containment: line built since ia_start vs perimeter of an ellipse of area A
            if t >= ia_start and burned > 0:   # an empty fire is never "contained" (PR #7 review, 21 Nov 2026)
                A = burned * cell_area
                per = math.pi * math.sqrt(2.0 * A / math.pi * (ia_lw + 1.0 / ia_lw))
                if ia_rate * (t - ia_start) >= per:
                    contained_t = t
                    break
        done[c] = True
        burned += 1
        k = int(t / dt_w)
        if k >= nk:
            k = nk - 1
        if k < 0:
            k = 0
        if cache_k[c] != k:
            cell_state(c, k, fuel_row, zone, slope_tan, upslope_az, waf, cbh, cbd, ch, adj, bundles, bundle10,
                       u20_ftmin, wind_to, u10_kmh, m1_pct, fmc, crown_mode, wl_mode, lw_mode, lw_max,
                       crown_ros_mult, cache[c], tmp_bd)
            cache_k[c] = k
        st = cache[c]
        # record behaviour at arrival
        th = arr_dir[c]
        if th != th:
            th = st[1]
        dir_ros(st, th, crown_mode, res3)
        rosd[c] = res3[0]
        ftype[c] = int(res3[1]) if st[0] > 0.0 else 0
        cfb_o[c] = res3[2]
        dir_ros(st, st[1], crown_mode, res3)
        roshead[c] = res3[0]
        hpa_o[c] = st[3]
        phiw_o[c] = st[10]
        phis_o[c] = st[11]
        lw_o[c] = st[13]
        i0_o[c] = st[4]
        rc_o[c] = st[6]
        cfl_o[c] = st[9]
        cy = c // nx
        cx = c - cy * nx
        # spotting from crowning (or very intense surface) cells
        if spot_on and ftype[c] >= 1:
            fli_src = 0.0
            nexp = 0.0
            if ftype[c] >= 2:
                fli_src = (st[3] + cfb_o[c] * st[9] * 18000.0) * roshead[c] / 60.0
                nexp = ember_rate * cell_area / 1e4 * (cfb_o[c] if cfb_o[c] > 0.05 else 0.05)
            elif spot_surface_fli > 0.0:
                fli_s = st[3] * roshead[c] / 60.0
                if fli_s > spot_surface_fli:
                    fli_src = fli_s
                    nexp = ember_rate * cell_area / 1e4 * 0.1
            if nexp > 0.0:
                z = zone[c]
                u20ms = u20_ftmin[k, z] * R.FTMIN_TO_MS
                dmax = max_spot_distance_m(fli_src, u20ms, ch[c] if ch[c] > 1.0 else 1.0, cbd[c] <= 0.0)
                if dmax > res_m:
                    ne = np.random.poisson(nexp)
                    for _e in range(ne):
                        d = dmax * spot_med_frac * math.exp(spot_sigma * np.random.standard_normal())
                        if d > dmax:
                            d = dmax * np.random.random()
                        if d < res_m:
                            continue
                        ang = wind_to[k, z] + spot_dir_sd * np.random.standard_normal()
                        lx = cx + int(round(d * math.sin(ang) / res_m))
                        ly = cy - int(round(d * math.cos(ang) / res_m))
                        if lx < 0 or ly < 0 or lx >= nx or ly >= ny:
                            continue
                        lc = ly * nx + lx
                        tfl = d / max(u20ms, 0.5) / 3600.0
                        t_ign = t + tfl + spot_delay_h
                        ok = 0.0
                        if fuel_row[lc] >= 0 and not done[lc] and t_ign < arrival[lc] and t_ign < t_block[lc]:
                            pig = p_ignition(m1_pct[k, zone[lc]] / 100.0, tair[k, zone[lc]], shade)
                            if np.random.random() < pig * spot_p_est:
                                ok = 1.0
                                arrival[lc] = t_ign
                                arr_dir[lc] = np.nan
                                src[lc] = 3
                                if not _heap_push(hk, hv, size, t_ign, lc):
                                    overflow = True
                        if nsp < max_spots:
                            sp[nsp, 0] = t
                            sp[nsp, 1] = c
                            sp[nsp, 2] = lc
                            sp[nsp, 3] = d
                            sp[nsp, 4] = t_ign
                            sp[nsp, 5] = ok
                            nsp += 1
        # relax neighbours
        for o in range(K):
            ny2 = cy + off_dy[o]
            nx2 = cx + off_dx[o]
            if ny2 < 0 or nx2 < 0 or ny2 >= ny or nx2 >= nx:
                continue
            c2 = ny2 * nx + nx2
            if done[c2] or fuel_row[c2] < 0:
                continue
            # corner rule (p1): a segment through a grid corner is blocked if both side cells cannot burn
            if not p2 and corner_a[o, 0] != -999:
                ca = (cy + corner_a[o, 0]) * nx + cx + corner_a[o, 1]
                cb = (cy + corner_b[o, 0]) * nx + cx + corner_b[o, 1]
                if fuel_row[ca] < 0 and fuel_row[cb] < 0:
                    continue
            theta = math.atan2(off_dx[o], -off_dy[o])
            dist = off_len[o] * res_m
            # integrate the segment through the weather steps
            tt = t
            p = 0.0
            kk = k
            blocked = False
            pruned = False
            if p2:
                # Physics p2 (22 Nov 2026): walk the segment cell by cell, each cell's part at its own spread rate,
                # stepping through the hourly weather. A cell the fire enters after its line is finished blocks the
                # step (U11: p1 checked every cell against the departure time), and a corner crossing is blocked when
                # both side cells are unburnable or lined by the time the fire reaches the corner (U1: p1 ignored
                # lines there). In constant weather the travel time equals p1's.
                s0 = seg_start[o]
                for s in range(s0, s0 + seg_n[o]):
                    cs = (cy + seg_dy[s]) * nx + cx + seg_dx[s]
                    if s > s0:
                        if fuel_row[cs] < 0 or t_block[cs] <= tt:
                            blocked = True
                            break
                        if s - s0 == corner_j[o]:
                            ca = (cy + corner_a[o, 0]) * nx + cx + corner_a[o, 1]
                            cb = (cy + corner_b[o, 0]) * nx + cx + corner_b[o, 1]
                            if _side_blocked(ca, fuel_row, t_block, tt) and _side_blocked(cb, fuel_row, t_block, tt):
                                blocked = True
                                break
                    rem = seg_frac[s]
                    while rem > 0.0:
                        _load_state(cs, kk, k, fuel_row, zone, slope_tan, upslope_az, waf, cbh, cbd, ch, adj,
                                    bundles, bundle10, u20_ftmin, wind_to, u10_kmh, m1_pct, fmc, crown_mode,
                                    wl_mode, lw_mode, lw_max, crown_ros_mult, cache, cache_k, cache_la, cache_la_k,
                                    nla, st_a, st_full, tmp_bd)
                        dir_ros(st_a, theta, crown_mode, res3)
                        r_ = res3[0]
                        tstep_end = (kk + 1) * dt_w if kk < nk - 1 else INF
                        if r_ <= 1e-9:
                            if tstep_end >= INF:
                                blocked = True
                                break
                            tt = tstep_end
                            kk += 1
                        else:
                            need = rem * dist / r_ / 60.0          # hours
                            if tt + need <= tstep_end:
                                tt = tt + need
                                rem = 0.0
                            else:
                                rem -= (tstep_end - tt) * 60.0 * r_ / dist
                                tt = tstep_end
                                kk += 1
                        if tt > t_end or tt >= arrival[c2]:
                            # no straight-segment candidate; the two-point update below may still reach c2 in time
                            pruned = True
                            break
                    if blocked or pruned:
                        break
            while not p2:
                slow = 0.0
                for s in range(seg_start[o], seg_start[o] + seg_n[o]):
                    cs = (cy + seg_dy[s]) * nx + cx + seg_dx[s]
                    if fuel_row[cs] < 0 or t_block[cs] <= tt:
                        blocked = True
                        break
                    if kk == k:
                        if cache_k[cs] != k:
                            cell_state(cs, k, fuel_row, zone, slope_tan, upslope_az, waf, cbh, cbd, ch, adj,
                                       bundles, bundle10, u20_ftmin, wind_to, u10_kmh, m1_pct, fmc, crown_mode,
                                       wl_mode, lw_mode, lw_max, crown_ros_mult, cache[cs], tmp_bd)
                            cache_k[cs] = k
                        dir_ros(cache[cs], theta, crown_mode, res3)
                    else:
                        sl = kk % nla
                        if cache_la_k[cs, sl] != kk:
                            cell_state(cs, kk, fuel_row, zone, slope_tan, upslope_az, waf, cbh, cbd, ch, adj,
                                       bundles, bundle10, u20_ftmin, wind_to, u10_kmh, m1_pct, fmc, crown_mode,
                                       wl_mode, lw_mode, lw_max, crown_ros_mult, st_tmp, tmp_bd)
                            for q_ in range(NLAS):
                                cache_la[cs, sl, q_] = st_tmp[q_]
                            cache_la_k[cs, sl] = kk
                        for q_ in range(NLAS):
                            st_tmp[q_] = cache_la[cs, sl, q_]
                        dir_ros(st_tmp, theta, crown_mode, res3)
                    if res3[0] <= 1e-9:
                        slow = INF
                        break
                    slow += seg_frac[s] / res3[0]
                if blocked:
                    break
                tstep_end = (kk + 1) * dt_w if kk < nk - 1 else INF
                if slow >= INF:
                    if tstep_end >= INF:
                        blocked = True
                        break
                    tt = tstep_end
                    kk += 1
                else:
                    need = (1.0 - p) * dist * slow / 60.0      # hours
                    if tt + need <= tstep_end:
                        tt = tt + need
                        break
                    p += (tstep_end - tt) * 60.0 / (dist * slow)
                    tt = tstep_end
                    kk += 1
                if tt > t_end:
                    blocked = True
                    break
                if tt >= arrival[c2]:
                    # exact pruning: this edge can no longer improve c2 (time only grows along the segment)
                    pruned = True
                    break
            if blocked:
                continue
            best = tt if not pruned else INF
            best_th = theta
            if simplex:
                # two-point (semi-Lagrangian) updates from the segment between this cell and an accepted angular
                # neighbour (relative to c2): removes most of the stencil's direction bias for elongated ellipses
                st2 = cache[c2]
                if cache_k[c2] != k:
                    cell_state(c2, k, fuel_row, zone, slope_tan, upslope_az, waf, cbh, cbd, ch, adj, bundles,
                               bundle10, u20_ftmin, wind_to, u10_kmh, m1_pct, fmc, crown_mode, wl_mode, lw_mode,
                               lw_max, crown_ros_mult, cache[c2], tmp_bd)
                    cache_k[c2] = k
                if st2[0] > 0.0:
                    ou = opp[o]
                    for side in range(2):
                        ow = nb_prev[ou] if side == 0 else nb_next[ou]
                        wy = ny2 + off_dy[ow]
                        wx = nx2 + off_dx[ow]
                        if wy < 0 or wx < 0 or wy >= ny or wx >= nx:
                            continue
                        cw = wy * nx + wx
                        if not done[cw] or fuel_row[cw] < 0:
                            continue
                        okw = True
                        for s_ in range(seg_start[opp[ow]], seg_start[opp[ow]] + seg_n[opp[ow]]):
                            if fuel_row[(wy + seg_dy[s_]) * nx + wx + seg_dx[s_]] < 0:
                                okw = False
                                break
                        if not okw:
                            continue
                        if cache_k[cw] != k:
                            cell_state(cw, k, fuel_row, zone, slope_tan, upslope_az, waf, cbh, cbd, ch, adj,
                                       bundles, bundle10, u20_ftmin, wind_to, u10_kmh, m1_pct, fmc, crown_mode,
                                       wl_mode, lw_mode, lw_max, crown_ros_mult, cache[cw], tmp_bd)
                            cache_k[cw] = k
                        tw = arrival[cw]
                        # vectors (east, north) in metres from u and from w to v
                        due = off_dx[o] * res_m
                        dun = -off_dy[o] * res_m
                        dwe = -off_dx[ow] * res_m
                        dwn = off_dy[ow] * res_m
                        de = due - dwe
                        dn = dun - dwn
                        L = math.sqrt(de * de + dn * dn)
                        if L <= 0.0:
                            continue
                        de /= L
                        dn /= L
                        e2 = st2[2]
                        cc = 1.0 / (st2[0] * (1.0 - e2))          # min per metre at the head
                        he = math.sin(st2[1])
                        hn = math.cos(st2[1])
                        kap = ((tw - t) * 60.0 / cc + e2 * (he * de + hn * dn) * L) / L
                        if kap <= -1.0 or kap >= 1.0:
                            continue
                        a_ = due * de + dun * dn
                        pe = due - a_ * de
                        pn = dun - a_ * dn
                        pp = math.sqrt(pe * pe + pn * pn)
                        q = kap * pp / math.sqrt(1.0 - kap * kap)
                        sv = (a_ - q) / L
                        if sv <= 1e-6 or sv >= 1.0 - 1e-6:
                            continue
                        ve = due - sv * L * de
                        vn = dun - sv * L * dn
                        dd = math.sqrt(ve * ve + vn * vn)
                        th2 = math.atan2(ve, vn)
                        if p2:
                            # U9 (p2): integrate the two-point path through the hourly weather from its start time,
                            # each step with that step's spread rates (p1 used the departure step's for the trip)
                            tt2 = (1.0 - sv) * t + sv * tw
                            kk2 = int(tt2 / dt_w)
                            if kk2 >= nk:
                                kk2 = nk - 1
                            if kk2 < 0:
                                kk2 = 0
                            rem2 = 1.0
                            okc = True
                            while True:
                                _load_state(c2, kk2, k, fuel_row, zone, slope_tan, upslope_az, waf, cbh, cbd, ch,
                                            adj, bundles, bundle10, u20_ftmin, wind_to, u10_kmh, m1_pct, fmc,
                                            crown_mode, wl_mode, lw_mode, lw_max, crown_ros_mult, cache, cache_k,
                                            cache_la, cache_la_k, nla, st_a, st_full, tmp_bd)
                                _load_state(c, kk2, k, fuel_row, zone, slope_tan, upslope_az, waf, cbh, cbd, ch,
                                            adj, bundles, bundle10, u20_ftmin, wind_to, u10_kmh, m1_pct, fmc,
                                            crown_mode, wl_mode, lw_mode, lw_max, crown_ros_mult, cache, cache_k,
                                            cache_la, cache_la_k, nla, st_b, st_full, tmp_bd)
                                _load_state(cw, kk2, k, fuel_row, zone, slope_tan, upslope_az, waf, cbh, cbd, ch,
                                            adj, bundles, bundle10, u20_ftmin, wind_to, u10_kmh, m1_pct, fmc,
                                            crown_mode, wl_mode, lw_mode, lw_max, crown_ros_mult, cache, cache_k,
                                            cache_la, cache_la_k, nla, st_c, st_full, tmp_bd)
                                dir_ros(st_a, th2, crown_mode, res3)
                                r_v = res3[0]
                                dir_ros(st_b, th2, crown_mode, res3)
                                r_u = res3[0]
                                dir_ros(st_c, th2, crown_mode, res3)
                                r_w = res3[0]
                                tstep_end = (kk2 + 1) * dt_w if kk2 < nk - 1 else INF
                                if r_v <= 1e-9 or r_u <= 1e-9 or r_w <= 1e-9:
                                    if tstep_end >= INF:
                                        okc = False
                                        break
                                    tt2 = tstep_end
                                    kk2 += 1
                                else:
                                    slow = 0.5 / r_v + 0.5 * ((1.0 - sv) / r_u + sv / r_w)
                                    need = rem2 * dd * slow / 60.0
                                    if tt2 + need <= tstep_end:
                                        tt2 = tt2 + need
                                        break
                                    rem2 -= (tstep_end - tt2) * 60.0 / (dd * slow)
                                    tt2 = tstep_end
                                    kk2 += 1
                                if tt2 >= best or tt2 > t_end or tt2 >= arrival[c2]:
                                    okc = False
                                    break
                            if not okc:
                                continue
                            # lines on either leg (u -> v, w -> v) finished before the arrival block the update, and
                            # so does a leg's corner with both sides unburnable or lined (U1/U11 for this update)
                            for leg in range(2):
                                ol = o if leg == 0 else opp[ow]
                                by = cy if leg == 0 else wy
                                bx = cx if leg == 0 else wx
                                for s_ in range(seg_start[ol] + 1, seg_start[ol] + seg_n[ol] - 1):
                                    if _side_blocked((by + seg_dy[s_]) * nx + bx + seg_dx[s_], fuel_row, t_block, tt2):
                                        okc = False
                                        break
                                if okc and corner_j[ol] >= 0:
                                    ca = (by + corner_a[ol, 0]) * nx + bx + corner_a[ol, 1]
                                    cb = (by + corner_b[ol, 0]) * nx + bx + corner_b[ol, 1]
                                    if _side_blocked(ca, fuel_row, t_block, tt2) and _side_blocked(cb, fuel_row,
                                                                                                    t_block, tt2):
                                        okc = False
                                if not okc:
                                    break
                            if not okc:
                                continue
                            cand = tt2
                        else:
                            dir_ros(st2, th2, crown_mode, res3)
                            r_v = res3[0]
                            dir_ros(cache[c], th2, crown_mode, res3)
                            r_u = res3[0]
                            dir_ros(cache[cw], th2, crown_mode, res3)
                            r_w = res3[0]
                            if r_v <= 1e-9 or r_u <= 1e-9 or r_w <= 1e-9:
                                continue
                            slow = 0.5 / r_v + 0.5 * ((1.0 - sv) / r_u + sv / r_w)
                            cand = (1.0 - sv) * t + sv * tw + dd * slow / 60.0
                        if cand < best:
                            best = cand
                            best_th = th2
            if best < arrival[c2] and best < t_block[c2] and best <= t_end:
                arrival[c2] = best
                arr_dir[c2] = best_th
                src[c2] = 2
                if not _heap_push(hk, hv, size, best, c2):
                    overflow = True
    for c in range(n):
        if not done[c]:
            arrival[c] = np.inf
    return (arrival, arr_dir, ftype, cfb_o, rosd, roshead, hpa_o, phiw_o, phis_o, lw_o, i0_o, rc_o, cfl_o, src,
            sp[:nsp], overflow, contained_t)
