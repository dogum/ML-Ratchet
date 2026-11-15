"""Statistics used to compare simulated and real price paths (calibration on dev only)."""
import numpy as np, pandas as pd
from battery import Battery
from strategies import pf_daily


def stats(rt, da, day, hour):
    rt = np.asarray(rt, float); da = np.asarray(da, float)
    b = Battery(1, 2, degradation=False)
    c, d = pf_daily(rt, day, b); pf = (rt * (d - c)).sum() / 1000 * 8760 / len(rt)
    g = pd.Series(rt).groupby(day); sp = g.max() - g.min()
    hs = pd.Series(rt).groupby(hour).median()
    return dict(mean=rt.mean(), median=np.median(rt), da_mean=np.nanmean(da), pf2h=pf, spread_med=sp.median(),
                spread_p90=sp.quantile(0.9), h100=int((rt > 100).sum()), h500=int((rt > 500).sum()),
                h1000=int((rt > 1000).sum()), neg=int((rt < 0).sum()),
                rtda_mae=float(np.nanmedian(np.abs(rt - da))),
                peak_hour=int(hs.idxmax()), shape_ratio=float(hs.max() / max(hs.min(), 1)))


def show(name, s):
    print(f"{name:14s} " + " ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
