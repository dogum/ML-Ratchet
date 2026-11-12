"""Historical tropical-cyclone tracks for the v3 simulator prior: IBTrACS v04r01 seasons 1985-2007 (never in dev or exam,
which use 2008-2025), storms that reached >= 64 kt (USA_WIND). Each track keeps 3-hourly (or 6-hourly) fixes of position,
1-min wind and mean R34. For every background buoy we precompute which tracks have their peak point 700-11,000 km away
over open water (the same geometry rule as the real candidates). Writes data/ibtracs/tracks_1985_2007.json."""
import os, json, numpy as np, pandas as pd
import catalog, geo, fetch
ROOT = catalog.ROOT; OUT = os.path.join(ROOT, 'data/ibtracs/tracks_1985_2007.json')

def build():
    out = []
    for b in ('EP', 'NA', 'WP', 'SP'):
        df = pd.read_csv(os.path.join(ROOT, f'data/ibtracs/ibtracs.{b}.csv'), skiprows=[1], low_memory=False,
                         usecols=['SID', 'SEASON', 'NAME', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'TRACK_TYPE', 'USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW'])
        df['SEASON'] = pd.to_numeric(df.SEASON, errors='coerce'); df = df[(df.SEASON >= 1985) & (df.SEASON <= 2007) & (df.TRACK_TYPE == 'main')]
        for c in ('LAT', 'LON', 'USA_WIND', 'USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW'): df[c] = pd.to_numeric(df[c], errors='coerce')
        for sid, g in df.groupby('SID'):
            w = g.USA_WIND.values
            if np.all(np.isnan(w)) or np.nanmax(w) < 64: continue
            t = pd.to_datetime(g.ISO_TIME).values.astype('datetime64[s]').astype(np.int64) / 3600.0
            r34 = np.nanmean(g[['USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW']].values, 1) * 1.852
            s = dict(sid=sid, name=str(g.NAME.iloc[0]), basin=b, season=int(g.SEASON.iloc[0]), t=np.round(t, 2).tolist(), lat=np.round(g.LAT.values, 2).tolist(),
                     lon=np.round((g.LON.values + 180) % 360 - 180, 2).tolist(), wind=np.nan_to_num(w, nan=-1).tolist(), r34=np.round(np.nan_to_num(r34, nan=-1), 0).tolist())
            k, m = catalog.peak_point(s); s['k'] = int(k); s['vmax'] = float(m); out.append(s)
    S = catalog.stations(); pairs = {}
    for b in fetch.BUOYS:
        if b not in S: continue
        bla, blo = S[b][:2]; pairs[b] = []
        for i, s in enumerate(out):
            la, lo = s['lat'][s['k']], s['lon'][s['k']]; D = float(geo.dist(la, lo, bla, blo))
            if 700e3 <= D <= 11000e3 and geo.ocean_path(la, lo, bla, blo): pairs[b].append(i)
        print(b, len(pairs[b]), flush=True)
    json.dump(dict(tracks=out, pairs=pairs), open(OUT, 'w')); print(len(out), 'tracks')

_T = None
def load():
    global _T
    if _T is None: _T = json.load(open(OUT))
    return _T

if __name__ == '__main__': build()
