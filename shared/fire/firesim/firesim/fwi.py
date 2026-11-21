"""Canadian Forest Fire Weather Index System (Van Wagner & Pickett 1985; Van Wagner 1987; equations as in Wang,
Anderson & Suddaby 2015, NOR-X-424). Daily noon inputs: T (C), RH (%), wind (km/h at 10 m), 24-h rain (mm).

Also hourly FFMC (Van Wagner 1977, the 'hourly FFMC' of the FBP system) for diurnal fine-fuel moisture.
"""
import math

import numpy as np
import pandas as pd

EL_DMC = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
FL_DC = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]


def ffmc(t, rh, ws, rain, ffmc0):
    mo = 147.2 * (101.0 - ffmc0) / (59.5 + ffmc0)
    if rain > 0.5:
        rf = rain - 0.5
        if mo > 150.0:
            mo = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf)) \
                + 0.0015 * (mo - 150.0) ** 2 * math.sqrt(rf)
        else:
            mo = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        mo = min(mo, 250.0)
    ed = 0.942 * rh ** 0.679 + 11.0 * math.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - t) * (1.0 - math.exp(-0.115 * rh))
    if mo > ed:
        ko = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * math.sqrt(ws) * (1.0 - (rh / 100.0) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * t)
        m = ed + (mo - ed) * 10.0 ** (-kd)
    else:
        ew = 0.618 * rh ** 0.753 + 10.0 * math.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - t) * (1.0 - math.exp(-0.115 * rh))
        if mo < ew:
            k1 = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * math.sqrt(ws) * (1.0 - ((100.0 - rh) / 100.0) ** 8)
            kw = k1 * 0.581 * math.exp(0.0365 * t)
            m = ew - (ew - mo) * 10.0 ** (-kw)
        else:
            m = mo
    return 59.5 * (250.0 - m) / (147.2 + m)


def dmc(t, rh, rain, dmc0, month):
    p0 = dmc0
    if rain > 1.5:
        re = 0.92 * rain - 1.27
        mo = 20.0 + math.exp(5.6348 - p0 / 43.43)
        if p0 <= 33.0:
            b = 100.0 / (0.5 + 0.3 * p0)
        elif p0 <= 65.0:
            b = 14.0 - 1.3 * math.log(p0)
        else:
            b = 6.2 * math.log(p0) - 17.2
        mr = mo + 1000.0 * re / (48.77 + b * re)
        pr = 244.72 - 43.43 * math.log(mr - 20.0)
        p0 = max(pr, 0.0)
    k = 1.894 * (t + 1.1) * (100.0 - rh) * EL_DMC[month - 1] * 1e-6 if t > -1.1 else 0.0
    return p0 + 100.0 * k


def dc(t, rain, dc0, month):
    d0 = dc0
    if rain > 2.8:
        rd = 0.83 * rain - 1.27
        qo = 800.0 * math.exp(-d0 / 400.0)
        qr = qo + 3.937 * rd
        dr = 400.0 * math.log(800.0 / qr)
        d0 = max(dr, 0.0)
    tt = t if t > -2.8 else -2.8
    v = 0.36 * (tt + 2.8) + FL_DC[month - 1]
    v = max(v, 0.0)
    return d0 + 0.5 * v


def isi(ffmc_v, ws):
    fw = math.exp(0.05039 * ws)
    m = 147.2 * (101.0 - ffmc_v) / (59.5 + ffmc_v)
    ff = 91.9 * math.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)
    return 0.208 * fw * ff


def bui(dmc_v, dc_v):
    if dmc_v <= 0.0 and dc_v <= 0.0:
        return 0.0
    if dmc_v <= 0.4 * dc_v:
        u = 0.8 * dmc_v * dc_v / (dmc_v + 0.4 * dc_v)
    else:
        u = dmc_v - (1.0 - 0.8 * dc_v / (dmc_v + 0.4 * dc_v)) * (0.92 + (0.0114 * dmc_v) ** 1.7)
    return max(u, 0.0)


def fwi(isi_v, bui_v):
    if bui_v <= 80.0:
        fd = 0.626 * bui_v ** 0.809 + 2.0
    else:
        fd = 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui_v))
    b = 0.1 * isi_v * fd
    if b > 1.0:
        return math.exp(2.72 * (0.434 * math.log(b)) ** 0.647)
    return b


def fwi_series(daily, ffmc0=85.0, dmc0=6.0, dc0=15.0):
    """Run the FWI system on a daily DataFrame (index dates; columns T, RH, wind_kmh, rain_24h).
    Returns columns FFMC, DMC, DC, ISI, BUI, FWI, DSR (daily severity rating 0.0272 FWI^1.77)."""
    rows = []
    f, p, d = ffmc0, dmc0, dc0
    for ts, r in daily.iterrows():
        mon = pd.Timestamp(ts).month
        rh = min(max(float(r['RH']), 0.0), 100.0)
        f = ffmc(float(r['T']), rh, float(r['wind_kmh']), float(r['rain_24h']), f)
        p = dmc(float(r['T']), rh, float(r['rain_24h']), p, mon)
        d = dc(float(r['T']), float(r['rain_24h']), d, mon)
        i = isi(f, float(r['wind_kmh']))
        b = bui(p, d)
        w = fwi(i, b)
        rows.append((f, p, d, i, b, w, 0.0272 * w ** 1.77))
    return pd.DataFrame(rows, index=daily.index, columns=['FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'DSR'])


def daily_from_hourly(w, lon=None):
    """Noon-LST daily inputs from hourly contract weather: T, RH, wind (km/h) at local noon; rain summed over the
    24 h ending at noon."""
    lon = lon if lon is not None else w.attrs.get('lon', -115.0)
    # local solar time = UTC + lon / 15 h (west is negative). PR #7 review, 21 Nov 2026: the sign was reversed, so
    # at lon = -100 'noon' was 05:20 UTC (local night).
    off = pd.Timedelta(hours=lon / 15.0)
    lst = w.index + off
    noon = w[(lst.hour == 12)]
    rain = w['precip'].rolling(24, min_periods=1).sum()
    out = pd.DataFrame(dict(T=noon['T'].to_numpy(), RH=noon['RH'].to_numpy(),
                            wind_kmh=noon['wind_speed'].to_numpy() * 3.6, rain_24h=rain.loc[noon.index].to_numpy()),
                       index=(noon.index + off).tz_localize(None).normalize())
    return out
