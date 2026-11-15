"""Final exam (2023-2025): run ONCE per final model; never used for choosing anything.
python3 exam.py <ckpt>  -> data/exam/<ckpt>.json + data/exam/<ckpt>_pred.npz
Sigma scales (widen-only, one per output) are fitted on the full dev bank (2019, 2021)."""
import sys, os, json, numpy as np
from common import *
import evaluate as E

BANDS = [-40, -10, 0, 10, 20, 25, 30, 35, 50]


def mape(p, y, m):
    return float(np.mean(np.abs(p[m] - y[m]) / y[m]) * 100) if m.sum() else np.nan


def nmae(p, y, w, m):
    return float(np.sum((np.abs(p - y) * w)[m]) / max(np.sum((y * w)[m]), 1e-9) * 100) if m.sum() else np.nan


def cov(y, mu, sig, m, k):
    return float(np.mean((np.abs(y - mu) <= k * sig)[m])) if m.sum() else np.nan


def block(B, pred, sc, rows, wrows=None, srows=None):
    """All metrics for a set of rows (wrows / srows: rows used for wind / solar, default = rows)."""
    out = {}
    # demand: common hours where model + all baselines exist
    y = B['d_true']; mu, sig = pred['d']; prior = B['pd_total']
    p = prior * np.exp(mu)
    m = rows[:, None] & np.isfinite(y) & np.isfinite(B['df']) & np.isfinite(B['pers']) & np.isfinite(B['week']) & (y > 0)
    yl = np.log(np.where(m, y, 1) / prior)
    out['demand'] = dict(model=mape(p, y, m), ba_day_ahead=mape(B['df'], y, m), persistence_24h=mape(B['pers'], y, m),
                         same_hour_last_week=mape(B['week'], y, m), physics_prior=mape(prior, y, m),
                         cov68=cov(yl, mu, sig * sc['d'], m, 1.0), cov95=cov(yl, mu, sig * sc['d'], m, 1.96),
                         bias_pct=float(np.mean(((p - y) / y)[m]) * 100) if m.sum() else np.nan, hours=int(m.sum()))
    for k, key, cap, pers, clim, pk in (('w', 'yw', 'capw', 'w_pers', 'w_clim', 'pw'), ('s', 'ys', 'caps', 's_pers', 's_clim', 'ps')):
        y = B[key]; mu, sig = pred[k]
        rr = {'w': wrows, 's': srows}[k]
        rows_k = rows if rr is None else rr
        has = B['hasw' if k == 'w' else 'hass']
        m = rows_k[:, None] & has[:, None] & np.isfinite(y) & np.isfinite(B[pers])
        w = np.broadcast_to(B[cap][:, None], y.shape)
        md = m & (B['night'] < 0.5) if k == 's' else m
        out['wind' if k == 'w' else 'solar'] = dict(
            model=nmae(mu, y, w, m), physics_prior=nmae(B[pk], y, w, m), persistence_24h=nmae(B[pers], y, w, m),
            cf_climatology=nmae(B[clim], y, w, m), cov68=cov(y, mu, sig * sc[k], md, 1.0),
            cov95=cov(y, mu, sig * sc[k], md, 1.96), bias_pct=float(np.sum(((mu - y) * w)[m]) / max(np.sum((y * w)[m]), 1e-9) * 100) if m.sum() else np.nan,
            hours=int(m.sum()))
    return out


def temp_bands(B, pred, rows):
    y = B['d_true']; mu, _ = pred['d']; p = B['pd_total'] * np.exp(mu); T = B['T']
    m0 = rows[:, None] & np.isfinite(y) & np.isfinite(B['df']) & (y > 0)
    out = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        m = m0 & (T >= lo) & (T < hi)
        out.append(dict(band=f'{lo}..{hi} C', hours=int(m.sum()), model=mape(p, y, m), ba_day_ahead=mape(B['df'], y, m),
                        physics_prior=mape(B['pd_total'], y, m)))
    return out


def extremes(B, pred, rows, frac=0.02):
    """Hottest and coldest 2 % of exam days per BA (by daily max / min input temperature), actual-weather mode."""
    y = B['d_true']; mu, _ = pred['d']; p = B['pd_total'] * np.exp(mu)
    res = {}
    for ba in BAS:
        r = rows & (B['meta_ba'] == ba)
        ix = np.where(r)[0]
        if not len(ix): continue
        tmax = B['T'][ix].max(1); tmin = B['T'][ix].min(1)
        n = max(3, int(round(frac * len(ix))))
        for name, sel in (('hot', ix[np.argsort(-tmax)[:n]]), ('cold', ix[np.argsort(tmin)[:n]])):
            rr = np.zeros(len(rows), bool); rr[sel] = True
            m = rr[:, None] & np.isfinite(y) & np.isfinite(B['df']) & (y > 0)
            res.setdefault(ba, {})[name] = dict(days=int(n), model=mape(p, y, m), ba_day_ahead=mape(B['df'], y, m),
                                               physics_prior=mape(B['pd_total'], y, m),
                                               T=float(np.mean(tmax[np.argsort(-tmax)[:n]]) if name == 'hot' else np.mean(tmin[np.argsort(tmin)[:n]])))
    return res


def main(ckpt):
    nets = E.load(ckpt)
    Bd = dict(np.load(f'{DATA}/banks/real_dev.npz'))
    pd_ = E.predict(nets, Bd)
    sc = {k: E.sigma_scale(Bd, pd_, k) for k in ('d', 'w', 's')}
    B = dict(np.load(f'{DATA}/banks/real_exam.npz'))
    pred = E.predict(nets, B)
    yr = np.array([int(d[:4]) for d in B['meta_day']])
    out = dict(ckpt=ckpt, sigma_scale_dev=sc, table={}, overall={})
    fcb = B['meta_fc']
    # forecast mode: demand needs T+RH+GHI forecasts (bits 1|2|4); wind rows additionally need wind (8), solar GHI (4)
    fcmode = (fcb & 7) == 7
    for mode, mrows in (('actual', fcb == 0), ('forecast', fcmode)):
        if not mrows.any(): continue
        wr = (fcb & 8 > 0) & (fcb > 0) if mode == 'forecast' else mrows
        sr = (fcb & 4 > 0) & (fcb > 0) if mode == 'forecast' else mrows
        out['overall'][mode] = block(B, pred, sc, mrows, wr, sr)
        for ba in BAS:
            for y in list(EXAM_YEARS) + ['all']:
                sel = (B['meta_ba'] == ba) & ((yr == y) if y != 'all' else True)
                if (mrows & sel).any() or (wr & sel).any():
                    out['table'].setdefault(ba, {}).setdefault(str(y), {})[mode] = block(B, pred, sc, mrows & sel, wr & sel, sr & sel)
    # forecast vs actual on the same days (2024-2025 full forecasts)
    act = B['meta_fc'] == 0
    days_fc = set(zip(B['meta_ba'][fcmode], B['meta_day'][fcmode]))
    same = act & np.array([(b, d) in days_fc for b, d in zip(B['meta_ba'], B['meta_day'])])
    out['overall']['actual_same_days_as_forecast'] = block(B, pred, sc, same)
    out['temp_bands'] = temp_bands(B, pred, act)
    out['extremes'] = extremes(B, pred, act)
    os.makedirs(f'{DATA}/exam', exist_ok=True)
    jdump(out, f'{DATA}/exam/{ckpt}.json')
    np.savez_compressed(f'{DATA}/exam/{ckpt}_pred.npz', **{f'{k}_mu': v[0] for k, v in pred.items()},
                        **{f'{k}_sig': v[1] * sc[k] for k, v in pred.items()})
    o = out['overall']
    for mode in o:
        d, w, s = o[mode]['demand'], o[mode]['wind'], o[mode]['solar']
        print(f"{mode:30s} demand {d['model']:.2f}% (BA DA {d['ba_day_ahead']:.2f}, pers {d['persistence_24h']:.2f}, prior {d['physics_prior']:.2f}) cov {d['cov68']:.2f}/{d['cov95']:.2f}"
              f" | wind {w['model']:.1f}% (prior {w['physics_prior']:.1f}, pers {w['persistence_24h']:.1f}, clim {w['cf_climatology']:.1f}) cov {w['cov68']:.2f}"
              f" | solar {s['model']:.1f}% (prior {s['physics_prior']:.1f}, pers {s['persistence_24h']:.1f}, clim {s['cf_climatology']:.1f}) cov {s['cov68']:.2f}")
    for ba in BAS:
        t = out['table'][ba]['all']['actual']  # noqa
        print(f"  {ba}: demand {t['demand']['model']:.2f} vs DA {t['demand']['ba_day_ahead']:.2f} | wind {t['wind']['model']:.1f} vs prior {t['wind']['physics_prior']:.1f} | solar {t['solar']['model']:.1f} vs prior {t['solar']['physics_prior']:.1f}")


if __name__ == '__main__':
    main(sys.argv[1])
