"""Twin test on DEV events only: simulate each dev storm from its real IBTrACS track (positions, 1-min winds, R34 size)
into the same real buoy window, and compare the twin's swell with what the buoy recorded.
Storm sea state from wind: Hs = a_hs * Vmax[m/s] (a_hs = 0.2 by default, ~12 m for a 60 m/s hurricane), Tp = 4.4 sqrt(Hs)."""
import os, sys, json, numpy as np
import geo, swellsim as SS, events as EV, realdata as RD, model as M
TC = (np.arange(SS.NT) * SS.DT_H - SS.HALF_H + SS.DT_H / 2)

def storm_track(sid):
    import catalog
    for s in catalog.storms():
        if s['sid'] == sid: return s

def simulate_track(e, s, a_hs=0.2, fe=0.006, w_iso=0.6, r1_0=0.95, R_default=250e3):
    blat, blon = e['buoy_lat'], e['buoy_lon']; tc = e['tc_grid']
    t = np.array(s['t']); w = np.array(s['wind']); ok = w > 0; t, w = t[ok], w[ok]; la = np.array(s['lat'])[ok]; lo = np.array(s['lon'])[ok]; r34 = np.array(s['r34'])[ok]
    ts = np.arange(t[0], t[-1], 0.5); W = np.interp(ts, t, w) * 0.5144; lat = np.interp(ts, t, la); lon = np.degrees(np.unwrap(np.radians(lo))); lon = (np.interp(ts, t, lon) + 180) % 360 - 180
    R = np.interp(ts, t, np.where(r34 > 0, r34 * 1e3, R_default)); hs = a_hs * W; keep = hs > 1.5; ts, lat, lon, hs, R = ts[keep], lat[keep], lon[keep], hs[keep], R[keep]
    tp = np.maximum(4.4 * np.sqrt(hs), 4.0); th = np.arange(tc - SS.HALF_H, tc + SS.HALF_H, 1.0); nt = len(th)
    Esrc = SS.jonswap(SS.FFINE, hs, tp, 2.0); Eacc = np.zeros((nt, len(SS.FFINE)))
    hd = np.r_[geo.bearing(lat[:-1], lon[:-1], lat[1:], lon[1:]), 0]; hd[-1] = hd[-2] if len(hd) > 1 else 0
    for ob in [None] + list(np.arange(6) * 60.0):
        if ob is None: sla, slo = lat, lon
        else: sla, slo = geo.dest(lat, lon, ob, 0.6 * R)
        D = np.maximum(geo.dist(sla, slo, blat, blon), 1e5); thb = geo.bearing(sla, slo, blat, blon)
        rel = (np.radians(thb - hd) + np.pi) % (2 * np.pi) - np.pi; dirw = w_iso / (2 * np.pi) + (1 - w_iso) * (1 / (2 * np.pi)) * 2 * np.cos(rel / 2) ** 2
        spread = 2 * R / 7 * dirw / (geo.R_E * np.abs(np.sin(D / geo.R_E)) + 2 * R)
        k = np.argmax(hs); att = SS.attenuation(SS.FFINE, float(D[k]), hs[k] ** 2 / 16, tp[k], 2 * R[k], float(dirw[k]), fe)
        dep = Esrc * spread[:, None] * att[None] * 0.5; ta = ts[:, None] + D[:, None] / SS.cg(SS.FFINE)[None] / 3600
        it = np.floor(ta - th[0]).astype(int); okk = (it >= 0) & (it < nt); jf = np.broadcast_to(np.arange(len(SS.FFINE))[None], ta.shape)
        np.add.at(Eacc, (it[okk], jf[okk]), dep[okk])
    E = Eacc.reshape(nt, 34, SS.NSUB).mean(2); return SS.to_coarse(E, E, E)[0]

def ridge_energy(E, t0_rel, D):
    v = []
    for j, f in enumerate(SS.FGRID):
        if 0.045 <= f <= 0.085:
            ta = t0_rel + D / SS.cg(f) / 3600; i = np.searchsorted(TC, ta)
            if 1 < i < SS.NT - 2: v.append(np.nanmean(E[i - 2:i + 2, j]))
    return np.array(v)

if __name__ == '__main__':
    ev = EV.load('dev', None); out = []
    for e in ev:
        s = storm_track(e['sid']); (Er, Cr, Sr), _ = RD.window(e['buoy'], e['tc']); Et = simulate_track(e, s)
        rr, rt = ridge_energy(Er, e['t0_rel'], e['D']), ridge_energy(Et, e['t0_rel'], e['D'])
        bg = np.nanpercentile(Er, 25, axis=0)
        # where does the twin's swell peak in time at 0.06-0.07 Hz, and where does the real excess peak?
        j = np.argmin(np.abs(SS.FGRID - 0.065)); it_t = int(np.nanargmax(Et[:, j])) if Et[:, j].max() > 0 else -1
        ex = Er[:, j] - bg[j]; it_r = int(np.nanargmax(np.where(np.isfinite(ex), ex, -9)))
        out.append(dict(name=e['name'], buoy=e['buoy'], D=e['D'] / 1e3, visible=e['visible'], contrast=e['contrast'], twin_peak_065=float(Et[:, j].max()),
                        real_excess_065=float(np.nanmax(ex)), t_twin_peak=float(TC[it_t]), t_real_peak=float(TC[it_r]),
                        ratio_ridge=float(np.nanmedian((rr - bg[(SS.FGRID >= 0.045) & (SS.FGRID <= 0.085)][:len(rr)]) / np.maximum(rt, 1e-3))) if len(rr) and len(rt) else np.nan))
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in out[-1].items()}), flush=True)
    json.dump(out, open(os.path.join(EV.ROOT, 'twin_dev.json'), 'w'), indent=1)
