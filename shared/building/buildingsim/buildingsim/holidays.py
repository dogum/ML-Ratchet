"""Public-holiday calendars for US, Canada (Ontario), UK and Ireland (opt-in, `options={'holidays': ...}`).

Ported unchanged from LoadCast (`models/loadcast/training/lc_common.py`, v1.0): BDG2 has sites in the US, Ontario,
England and Ireland, and BuildingSim v1.0 knows US holidays only (`weather._holidays`). With
`options={'holidays': 'auto'}` the country follows the weather's time zone (Europe/London -> UK, Europe/Dublin -> IE,
America/Toronto -> CA, anything else -> US). For the US the v1.0 calendar is kept, as LoadCast's wrapper did, so
'auto' changes nothing at a US site.
"""
import functools

import numpy as np
import pandas as pd

COUNTRY = {'Europe/London': 'UK', 'Europe/Dublin': 'IE', 'America/Toronto': 'CA'}
COUNTRIES = ('US', 'CA', 'UK', 'IE')


def country_of_tz(tz):
    return COUNTRY.get(str(tz), 'US')


def _easter(y):
    a = y % 19; b = y // 100; c = y % 100; d = b // 4; e = b % 4; f = (b + 8) // 25; g = (b - f + 1) // 3  # noqa: E702
    h = (19 * a + b - d - g + 15) % 30; i = c // 4; k = c % 4; l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E702,E741
    m = (a + 11 * h + 22 * l) // 451; mo = (h + l - 7 * m + 114) // 31; da = ((h + l - 7 * m + 114) % 31) + 1  # noqa
    return pd.Timestamp(y, mo, da)


def _nth_dow(y, mo, dow, n):
    d = pd.Timestamp(y, mo, 1); off = (dow - d.dayofweek) % 7  # noqa: E702
    return d + pd.Timedelta(days=off + 7 * (n - 1))


def _last_dow(y, mo, dow):
    d = pd.Timestamp(y, mo, 1) + pd.offsets.MonthEnd(0); off = (d.dayofweek - dow) % 7  # noqa: E702
    return d - pd.Timedelta(days=off)


def _observed(d):      # weekend fixed-date holidays move to Monday (UK/IE/CA substitute days)
    return d + pd.Timedelta(days={5: 2, 6: 1}.get(d.dayofweek, 0))


@functools.lru_cache(maxsize=64)
def holiday_dates(country, year):
    y = year; D = []  # noqa: E702
    if country == 'US':
        D = [pd.Timestamp(y, 1, 1), _nth_dow(y, 1, 0, 3), _nth_dow(y, 2, 0, 3), _last_dow(y, 5, 0), pd.Timestamp(y, 7, 4),
             _nth_dow(y, 9, 0, 1), _nth_dow(y, 11, 3, 4), _nth_dow(y, 11, 3, 4) + pd.Timedelta(days=1),
             pd.Timestamp(y, 12, 24), pd.Timestamp(y, 12, 25), pd.Timestamp(y, 12, 31)]
    elif country == 'CA':
        e = _easter(y)
        vic = pd.Timestamp(y, 5, 24); vic = vic - pd.Timedelta(days=(vic.dayofweek - 0) % 7)  # noqa: E702
        D = [_observed(pd.Timestamp(y, 1, 1)), _nth_dow(y, 2, 0, 3), e - pd.Timedelta(days=2), vic,
             _observed(pd.Timestamp(y, 7, 1)), _nth_dow(y, 8, 0, 1), _nth_dow(y, 9, 0, 1), _nth_dow(y, 10, 0, 2),
             pd.Timestamp(y, 12, 24), pd.Timestamp(y, 12, 25), pd.Timestamp(y, 12, 26), pd.Timestamp(y, 12, 31)]
    elif country == 'UK':
        e = _easter(y)
        D = [_observed(pd.Timestamp(y, 1, 1)), e - pd.Timedelta(days=2), e + pd.Timedelta(days=1), _nth_dow(y, 5, 0, 1),
             _last_dow(y, 5, 0), _last_dow(y, 8, 0), pd.Timestamp(y, 12, 24), pd.Timestamp(y, 12, 25),
             pd.Timestamp(y, 12, 26), pd.Timestamp(y, 12, 27), pd.Timestamp(y, 12, 31)]
    elif country == 'IE':
        e = _easter(y)
        D = [_observed(pd.Timestamp(y, 1, 1)), _observed(pd.Timestamp(y, 3, 17)), e + pd.Timedelta(days=1),
             _nth_dow(y, 5, 0, 1), _nth_dow(y, 6, 0, 1), _nth_dow(y, 8, 0, 1), _last_dow(y, 10, 0),
             pd.Timestamp(y, 12, 24), pd.Timestamp(y, 12, 25), pd.Timestamp(y, 12, 26), pd.Timestamp(y, 12, 27)]
    return frozenset(d.normalize() for d in D)


def is_holiday(days, country):
    """days: DatetimeIndex of local dates (naive). -> bool array."""
    days = pd.DatetimeIndex(days).normalize()
    out = np.zeros(len(days), bool)
    for y in np.unique(days.year):
        hs = holiday_dates(country, int(y))
        out |= np.isin(days.values, np.array(sorted(hs), dtype='datetime64[ns]'))
    return out


def resolve(holidays, tz):
    """options['holidays'] -> country code, or None for the v1.0 US calendar."""
    if holidays is None:
        return None
    c = country_of_tz(tz) if holidays == 'auto' else str(holidays).upper()
    if c not in COUNTRIES:
        raise ValueError(f"holidays: 'auto' or one of {COUNTRIES}, got {holidays!r}")
    return None if c == 'US' else c
