"""Storm truth from IBTrACS v04r01 and buoy positions from the NDBC station table. Builds the list of candidate
(storm, buoy) swell events from storm tracks and geometry alone: no buoy measurement is used to pick the truth.

Truth definition (the same one the simulator labels): the storm 'origin' is the track point of peak USA_WIND (middle of the
plateau if the peak lasts several fixes); D = great-circle distance from that point to the buoy; t0 = its time;
dir = compass bearing from the buoy toward that point (the direction the swell comes FROM)."""
import os, re, json, numpy as np, pandas as pd
import geo
ROOT = geo.ROOT; G = 9.81
F_REF = 0.06           # Hz: windows are centred on the predicted arrival of this frequency, plus a seeded jitter
HALF_WIN_H = 120       # window is 240 h (10 days)
JITTER_H = 36

def cg(f): return G / (4 * np.pi * np.asarray(f, float))

def stations():
    S = {}
    for ln in open(os.path.join(ROOT, 'data/ndbc/station_table.txt')):
        if ln.startswith('#'): continue
        p = ln.split('|')
        m = re.match(r'\s*([\d.]+)\s*([NS])\s+([\d.]+)\s*([EW])', p[6]) if len(p) > 6 else None
        if m: S[p[0].strip().lower()] = (float(m.group(1)) * (1 if m.group(2) == 'N' else -1), float(m.group(3)) * (1 if m.group(4) == 'E' else -1), p[4].strip())
    return S

def years_available():
    av = {}
    for kind, fn in (('w', 'list_swden.txt'), ('d', 'list_swdir.txt'), ('j', 'list_swr1.txt')):
        for ln in open(os.path.join(ROOT, 'data/ndbc', fn)):
            m = re.match(r'([0-9a-z]+)([wdj])(\d{4})$', ln.strip())
            if m: av.setdefault(m.group(1), {}).setdefault(kind, set()).add(int(m.group(3)))
    return av

def storms(basins=('EP', 'NA', 'WP', 'SP'), y0=2005):
    cache = os.path.join(ROOT, 'data/ibtracs/storms.json')
    if os.path.exists(cache): return json.load(open(cache))
    out = []
    for b in basins:
        df = pd.read_csv(os.path.join(ROOT, f'data/ibtracs/ibtracs.{b}.csv'), skiprows=[1], low_memory=False,
                         usecols=['SID', 'SEASON', 'NAME', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'WMO_WIND', 'TRACK_TYPE', 'USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW'])
        df = df[(pd.to_numeric(df.SEASON, errors='coerce') >= y0) & (df.TRACK_TYPE == 'main')]
        for c in ('LAT', 'LON', 'USA_WIND', 'WMO_WIND', 'USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW'): df[c] = pd.to_numeric(df[c], errors='coerce')
        for sid, g in df.groupby('SID'):
            w = g.USA_WIND.values
            if np.all(np.isnan(w)): continue
            t = pd.to_datetime(g.ISO_TIME).values.astype('datetime64[s]').astype(np.int64) / 3600.0   # hours since 1970
            r34 = np.nanmean(g[['USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW']].values, 1) * 1.852   # km
            out.append(dict(sid=sid, name=str(g.NAME.iloc[0]), basin=b, season=int(g.SEASON.iloc[0]), t=t.tolist(), lat=g.LAT.values.tolist(),
                            lon=((g.LON.values + 180) % 360 - 180).tolist(), wind=np.nan_to_num(w, nan=-1).tolist(), r34=np.nan_to_num(r34, nan=-1).tolist()))
    json.dump(out, open(cache, 'w')); return out

def peak_point(s):
    w = np.array(s['wind']); m = w.max(); idx = np.where(w >= m)[0]; k = idx[len(idx) // 2]   # middle of the peak plateau
    return k, m

def candidates(min_wind=64, dmin=1000e3, dmax=9000e3, years=(2008, 2025)):
    S = stations(); av = years_available(); out = []
    buoys = {b: S[b] for b in av if b in S and 'w' in av[b] and 'd' in av[b] and 'j' in av[b]}
    for s in storms():
        k, m = peak_point(s); k, m = int(k), float(m)
        if m < min_wind or not (years[0] <= s['season'] <= years[1]): continue
        la, lo, t0 = s['lat'][k], s['lon'][k], s['t'][k]; yr = int(np.datetime64(int(t0), 'h').astype('datetime64[Y]').astype(int) + 1970)
        for b, (bla, blo, name) in buoys.items():
            if yr not in av[b]['w']: continue
            D = float(geo.dist(la, lo, bla, blo))
            if not (dmin <= D <= dmax): continue
            if not geo.ocean_path(la, lo, bla, blo): continue
            r34 = s['r34'][k]
            out.append(dict(sid=s['sid'], name=s['name'], basin=s['basin'], buoy=b, buoy_lat=bla, buoy_lon=blo, lat=float(la), lon=float(lo), t0=float(t0), wind=m, r34_km=float(r34),
                            D=D, dir=float(geo.bearing(bla, blo, la, lo)), year=yr, has_dir=bool(yr in av[b]['d'] and yr in av[b]['j'])))
    return out

if __name__ == '__main__':
    C = candidates(); print(len(C), 'candidate pairs')
    df = pd.DataFrame(C); print(df.groupby('buoy').size().sort_values(ascending=False).head(60).to_string())
    print(df.groupby('basin').size()); json.dump(C, open(os.path.join(ROOT, 'data/candidates_raw.json'), 'w'))
