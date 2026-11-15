"""EIA-930 solar quality flags (HANDOFF P0.6, from DemandCast's build) -> proc/checks/eia930_solar_qc.json

For every focus BA and year: the timestamp offset of the reported solar generation (ng_SUN) against the irradiance
at the BA's solar points (NASA POWER all-sky GHI, the `ghi_Wm2` column of proc/weather/<BA>.parquet), and the
negative night-time solar the BAs report.

    offset_h: the shift k in -3..+3 h that maximises corr(ng_SUN[t], GHI[t - k]); +1 = reported solar LAGS the sun by
              one hour, -1 = it LEADS by one hour (hour-beginning UTC on both sides).
    corr_at_0, corr_at_offset: the two correlations (daytime hours of the year).
    night_negative_share: share of hours with GHI = 0 and ng_SUN < 0; night_negative_MW_median: their median.

It is a data check, independent of any model: the offsets come from the reported series and the weather alone.
A consumer that needs aligned solar should shift by -offset_h. Loader: griddata.eia930_solar_qc().
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import json, os
import numpy as np, pandas as pd

PROC = f'{GRID_ROOT}/data/proc'
FOCUS = ['ERCO', 'CISO', 'PJM', 'MISO', 'SWPP', 'ISNE', 'NYIS', 'BPAT']
SHIFTS = range(-3, 4)


def main():
    out = {}
    for ba in FOCUS:
        e = pd.read_parquet(f'{PROC}/eia930/ba/{ba}.parquet', columns=['time_utc', 'ng_SUN']).set_index('time_utc')
        w = pd.read_parquet(f'{PROC}/weather/{ba}.parquet', columns=['time_utc', 'ghi_Wm2']).set_index('time_utc')
        e = e[~e.index.duplicated()]; w = w[~w.index.duplicated()]
        out[ba] = {}
        for y in sorted(set(e.index.year)):
            s = pd.to_numeric(e.ng_SUN[e.index.year == y], errors='coerce')
            if s.notna().sum() < 24 * 60 or s.max() <= 0:
                continue
            g = w.ghi_Wm2.reindex(pd.date_range(s.index.min() - pd.Timedelta(hours=3), s.index.max() + pd.Timedelta(hours=3),
                                                freq='h'))
            cors = {}
            for k in SHIFTS:
                gk = g.shift(k).reindex(s.index)          # GHI[t - k]
                m = s.notna() & gk.notna() & ((gk > 0) | (s > 0))
                cors[k] = float(np.corrcoef(s[m], gk[m])[0, 1]) if m.sum() > 100 else np.nan
            best = max(cors, key=lambda k: -np.inf if np.isnan(cors[k]) else cors[k])
            g0 = g.reindex(s.index)
            night = (g0 == 0) & s.notna()
            neg = night & (s < 0)
            out[ba][str(y)] = dict(offset_h=int(best), corr_at_0=round(cors[0], 4), corr_at_offset=round(cors[best], 4),
                                   night_negative_share=round(float(neg.sum() / max(night.sum(), 1)), 4),
                                   night_negative_MW_median=float(s[neg].median()) if neg.any() else 0.0,
                                   hours=int(s.notna().sum()))
        print(ba, {y: (v['offset_h'], v['corr_at_0'], v['corr_at_offset'], v['night_negative_share'])
                   for y, v in out[ba].items()}, flush=True)
    os.makedirs(f'{PROC}/checks', exist_ok=True)
    json.dump(dict(note=__doc__.split('\n\n')[1], bas=out), open(f'{PROC}/checks/eia930_solar_qc.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
