"""System 9068 (Kersey, Colorado) on SunDoctor's solar-time grid, with three physics answer keys:
off-tracking (from the 224 tracker-angle sensors vs the ideal backtracking angle), outages (one inverter module dark while
its 7 siblings produce), and snow (the whole fleet far below what its irradiance sensors see, on cold days)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, numpy as np, pandas as pd
sys.path.insert(0, _P.SRC)
import pvsim as S, tracker as TR, realdata as R, simbatch2 as SB2

SYS = dict(lat=40.3864, lon=-104.5512, tz='US/Mountain', axis_az=180.0, max_angle=60.0, gcr=0.35, backtrack=False, pdc_kw=4738.0 / 8, n_mod=8)   # trackers true-track (no backtracking): measured angles sit at +-60 deg at low sun

def geometry(days, gcr=SYS['gcr']):
    doy = days.dayofyear.values[:, None]; cz, el, az = S.sun(SYS['lat'], doy, S.TSOL[None])
    th = TR.rotation(cz, az, SYS['axis_az'], SYS['max_angle'], gcr, SYS['backtrack']); tilt, saz = TR.surface(th, SYS['axis_az']); ca = TR.cos_aoi(th, cz, az, SYS['axis_az'])
    g, dn, dh = S.ineichen(cz, doy, 3.0); b, sk, gn = S.haydavies(tilt, ca, cz, dn, dh, g, doy)
    return dict(cz=np.maximum(cz, 0), ca=np.maximum(ca, 0), poa_clear=b + sk + gn, el=el, az=az, theta=th, dni=dn, dhi=dh, ghi=g, doy=doy)

def load(cache=_P.WORK + '/data/sys9068_solar.npz'):
    try:
        z = np.load(cache, allow_pickle=False); days = pd.to_datetime(z['days'], unit='D')
        return days, {k: z[k] for k in z.files if k != 'days'}, geometry(days)
    except FileNotFoundError: pass
    z = np.load(_P.WORK + '/data/sys9068_15min.npz')
    t_local = pd.to_datetime(z['t'], unit='m') + pd.Timedelta(minutes=7.5)     # local clock with daylight saving; labels mark interval starts
    t_utc = t_local.tz_localize(SYS['tz'], ambiguous='NaT', nonexistent='NaT').tz_convert('UTC').tz_localize(None); ok = ~pd.isna(t_utc); tu = pd.DatetimeIndex(t_utc[ok])
    pac = np.where((z['pac'][ok] > -5) & (z['pac'][ok] < 600), z['pac'][ok], np.nan)
    days, P = R.to_solar_grid(tu, pac, SYS['lon']); _, G = R.to_solar_grid(tu, np.where((z['poa'][ok] > -5) & (z['poa'][ok] < 1600), z['poa'][ok], np.nan), SYS['lon'])
    _, T = R.to_solar_grid(tu, z['tamb'][ok], SYS['lon']); _, A = R.to_solar_grid(tu, z['ang'][ok], SYS['lon'])
    inv = np.array([1 if n.startswith('1') else 2 for n in z['tmod_of']])
    out = dict(P=(P / SYS['pdc_kw']).astype(np.float32), G=(G / 1000.0).astype(np.float32), T=T.astype(np.float32), A=A.astype(np.float16), inv=inv)
    np.savez_compressed(cache, days=days.values.astype('datetime64[D]').astype(np.int64), **out)
    return days, out, geometry(days)

def keys(days, d, geo):
    P, G, T, A, inv = d['P'], d['G'], d['T'], d['A'].astype(np.float32), d['inv']
    up = geo['cz'] > 0.1; th = geo['theta']
    # irradiance-sensor health, month by month: a working sensor reaches clear-sky levels on some days each month.
    # At this site the sensors read about half the true irradiance from May 2019 to Oct 2022 (and zero for much of 2022).
    with np.errstate(all='ignore'):
        r95 = np.nanpercentile(np.where(geo['cz'] > 0.3, G / np.maximum(geo['poa_clear'] / 1000, 0.05), np.nan), 95, axis=1)
    mon = pd.Series(r95, index=days).groupby([days.year, days.month]).transform(lambda x: x.quantile(0.9)).values
    sensor_ok = mon > 0.95
    # 1. off-tracking: share of each inverter's trackers more than 10 deg from the ideal angle, and the clear-sky energy it costs
    dev = np.abs(A - th[..., None]); off = (dev > 10) & up[..., None] & np.isfinite(A)
    kc = np.clip(G / np.maximum(geo['poa_clear'] / 1000, 0.05), 0, 1.3); kc = np.where(np.isfinite(kc) & sensor_ok[:, None], kc, 1.0)     # clear-sky weights where the sensor is missing or broken
    def poa_at(theta):
        tilt, saz = TR.surface(theta, SYS['axis_az']); ca = TR.cos_aoi(theta, geo['cz'], geo['az'], SYS['axis_az'])
        b, s, g = S.haydavies(tilt, ca, geo['cz'], geo['dni'], geo['dhi'], geo['ghi'], geo['doy']); return b + s + g
    ideal = poa_at(th); stuck_loss = {}
    for i in (1, 2):
        Ai = np.where(np.isfinite(A[..., inv == i]), A[..., inv == i], th[..., None])
        act = np.mean(np.stack([poa_at(Ai[..., k]) for k in range(Ai.shape[-1])], -1), -1)
        w = kc * up; den = np.sum(w * ideal, 1); stuck_loss[i] = np.where(den > 50, np.clip(1 - np.sum(w * act, 1) / np.maximum(den, 1e-6), 0, 1), np.nan)
    off_frac = {i: np.nanmean(np.where(up[..., None], off[..., inv == i], np.nan), (1, 2)) for i in (1, 2)}
    # 2. outages: one module dark (<5% of the fleet level) while the fleet produces
    ref = np.nanpercentile(P, 75, axis=2); dark = (P < 0.05 * ref[..., None]) & (ref[..., None] > 0.1)
    outage = np.nanmean(np.where((ref > 0.1)[..., None], dark, np.nan), 1)        # (days, 8) share of producing steps dark
    # on cold, dim days a dark module may be under snow its neighbours have shed: no outage answer on those days
    pcs = SB2.expected_power(geo['poa_clear'] / 1000, np.nan_to_num(T, nan=0), geo['ca'])
    with np.errstate(all='ignore'):
        fleet_cs = np.nanmedian(np.nansum(np.where(up[..., None], P, 0), 1), 1) / np.maximum(np.sum(np.where(up, pcs, 0), 1), 1e-6)
        tmean0 = np.nanmean(np.where(up, T, np.nan), 1)
    outage = np.where(((tmean0 < 3) & (fleet_cs < 0.6))[:, None], np.nan, outage)
    # 3. snow: the fleet far below what the irradiance sensors imply, on a cold day, most modules alike
    pe = SB2.expected_power(np.nan_to_num(np.maximum(G, 0)), np.nan_to_num(T, nan=0), geo['ca'])
    m = up & (np.nan_to_num(G) > 0.15)
    with np.errstate(all='ignore'):
        ratio = np.nansum(np.where(m[..., None], P, 0), 1) / np.maximum(np.nansum(np.where(m, pe, 0), 1)[:, None], 1e-6)   # (days, 8)
        fleet = np.nanmedian(ratio, 1); tmean = np.nanmean(np.where(up, T, np.nan), 1)
    alike = np.sum(ratio < 0.7, 1) >= 6
    # a snow day: the fleet far below the sensors, cold, most modules alike. The loss is then read module by module (some shed
    # their snow sooner), and a module with patchy data that day has no answer.
    covered = np.sum(np.where(m[..., None], np.isfinite(P), 0), 1) / np.maximum(m.sum(1), 1)[:, None]          # (days, 8)
    day = (fleet < 0.7) & (tmean < 3) & alike & (m.sum(1) >= 8) & sensor_ok
    snow = np.where(day[:, None], np.clip(1 - ratio / 0.95, 0, 1), 0.0)
    snow_known = (np.isfinite(fleet) & (m.sum(1) >= 8) & sensor_ok)[:, None] & (covered > 0.8)
    return dict(sensor_ok=sensor_ok, stuck_loss=stuck_loss, off_frac=off_frac, outage=outage, snow=snow, snow_known=snow_known, fleet_ratio=fleet, tmean=tmean)

if __name__ == '__main__':
    days, d, geo = load(); K = keys(days, d, geo)
    print('days', len(days), 'P', d['P'].shape)
    for i in (1, 2): print('inverter', i, 'days >5% off-tracking:', int(np.nansum(K['off_frac'][i] > 0.05)), 'days stuck loss >2%:', int(np.nansum(K['stuck_loss'][i] > 0.02)), 'mean loss %.4f' % np.nanmean(K['stuck_loss'][i]))
    print('module outage days (>50% dark):', (K['outage'] > 0.5).sum(0), 'days without an outage answer (cold and dim):', int(np.isnan(K['outage'][:, 0]).sum()))
    sd = (K['snow'] > 0).any(1); big = (K['snow'] > 0.2) & K['snow_known']
    print('snow days:', int(sd.sum()), 'module-days losing >20% to snow:', int(big.sum()), 'mean loss on those %.3f' % K['snow'][big].mean(), 'months', np.bincount(days[sd].month, minlength=13)[1:])
    print('sensor-healthy days:', int(K['sensor_ok'].sum()), 'of', len(days))
    np.savez_compressed(_P.WORK + '/data/sys9068_keys.npz', sensor_ok=K['sensor_ok'], stuck1=K['stuck_loss'][1], stuck2=K['stuck_loss'][2], off1=K['off_frac'][1], off2=K['off_frac'][2],
                        outage=K['outage'], snow=K['snow'], snow_known=K['snow_known'], fleet=K['fleet_ratio'], tmean=K['tmean'])
