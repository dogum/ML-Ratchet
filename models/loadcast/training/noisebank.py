"""Real-noise bank from BDG2 DEV meters (never exam).

For every dev building-meter that is q_ok in both 2016 and 2017, and for each local year separately:
    e(t) = (y(t) - weather regression in-sample(t)) / mean_year(y)   (TOWT + daily-mean T + dewpoint + solar)
i.e. what a time-of-week + temperature model cannot explain: day-to-day occupancy, holidays and campus breaks,
events, meter noise, equipment cycling. Stored on a naive local-hour index (hours since local 2016-01-01 00:00) so
it can be injected into a simulated building at any site on the same calendar dates. Also stores each meter's
2017/2016 level ratio (same calendar days) and its missing-hour mask (real gap patterns).
-> data/noise_bank.npz
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_common as C

NL = 17544 + 24


def design(T, T24, dew, ghi, tow):
    """In-sample weather regression used to strip weather signal before keeping the residual as noise:
    TOWT (168 + 6) + piecewise T24 (6) + humid dewpoint + solar + occupied-hour x T (2)."""
    X1 = C.towt_X(T, tow)
    X2 = C.towt_X(T24, tow)[:, 168:]
    occ = ((tow % 24 >= 8) & (tow % 24 < 18) & (tow // 24 < 5)).astype(float)
    return np.column_stack([X1, X2, np.maximum(dew - 12, 0), ghi / 500, occ * T, occ * T24])


def main():
    q = C.bd.bdg2_quality(split='dev')
    q = q[q.meter.isin(C.METERS) & q.q_ok_2016 & q.q_ok_2017]
    E, M, B, R, G = [], [], [], [], []
    for m in C.METERS:
        qq = q[q.meter == m]
        if not len(qq): continue
        X = C.real_meter(m, list(qq.building_id))
        for _, r in qq.iterrows():
            b = r['building_id']; site = r['site_id']; cal = C.site_calendar(site)
            w, _ = C.site_weather(site)
            T = w['T_out'].interpolate(limit_direction='both').values
            T24 = pd.Series(T).rolling(24, min_periods=1).mean().values
            dew = np.minimum(w['dewpoint'].interpolate(limit_direction='both').values, T)
            ghi = w['GHI'].fillna(0).values
            y = X[b].values.astype(float)
            e = np.full(NL, np.nan, np.float32); g = np.zeros(NL, bool)
            ok_all = True
            for yr in (2016, 2017):
                sel = cal['year'] == yr
                ys = y[sel]; tow = (cal['dow'][sel] * 24 + cal['hour'][sel])
                mu = np.nanmean(ys)
                if not np.isfinite(mu) or mu <= 0: ok_all = False; break
                Xw = design(T[sel], T24[sel], dew[sel], ghi[sel], tow)
                ok = np.isfinite(ys)
                cf = np.linalg.solve(Xw[ok].T @ Xw[ok] + 1e-3 * np.eye(Xw.shape[1]), Xw[ok].T @ (ys[ok] / mu))
                res = ys / mu - Xw @ cf
                li = cal['lidx'][sel]
                e[li] = res; g[li] = ~np.isfinite(ys)
            if not ok_all: continue
            s16 = cal['year'] == 2016; s17 = cal['year'] == 2017
            # 2017/2016 ratio on matched day-of-year
            d = pd.DataFrame(dict(y=y, doy=cal['doy'], yr=cal['year']))
            a = d[d.yr == 2016].groupby('doy').y.mean(); bb = d[d.yr == 2017].groupby('doy').y.mean()
            jj = a.index.intersection(bb.index); ok = a[jj].notna() & bb[jj].notna()
            ratio = float(bb[jj][ok].sum() / max(a[jj][ok].sum(), 1e-9))
            E.append(e); M.append(m); B.append(b); R.append(ratio); G.append(g)
            print(b, m, 'resid sd %.3f  yoy %.2f' % (np.nanstd(e), ratio), flush=True)
    np.savez_compressed(f'{C.ROOT}/data/noise_bank.npz', E=np.array(E, np.float16), meter=np.array(M), bid=np.array(B),
                        ratio=np.array(R), gaps=np.array(G))
    print(len(E), 'series')


if __name__ == '__main__':
    main()
