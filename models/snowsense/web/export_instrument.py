"""Export SnowSense's weights, the three shipped exam stations and Python reference cases for the browser instrument.

    python web/export_instrument.py            (from models/snowsense, or any folder)

Writes site/public/instruments/snowsense/:
  weights.{json,bin}   the released GRU (float16) and the dev widen factors
  stations.{json,bin}  the three shipped exam stations (examples/*.csv), WY2016-2025 plus the August-September lead-in:
                       daily P, T_mean, T_min, T_max and their gap flags after the pipeline's own gap handling
                       (snowsense.clean), and the observed pillow SWE moved to end of day (SNOTEL reports start of day)
  reference.json       SnowSense, its bands and annual readings, and WaterSim's uncalibrated degree-day and energy-balance
                       snow routines, for recorded and what-if weather (warmer, more or less precipitation, the station
                       moved in elevation with or without the lapse rate, the biggest storm removed, a gauge outage)

The reference outputs use the float16-rounded weights and the float32 station values the browser reads, so the
comparison is exact in what it measures. The physics is WaterSim (shared/water/watersim) run snow-only from 1 August
with no snow, default parameters, Hargreaves radiation: the same routines as the exam's weather-only baselines, which
instead ran continuously from WY2010 with a one-year spin-up (on 1 August a SNOTEL pillow is almost always bare).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = Path(__file__).resolve().parents[1]
REPO = MODEL.parents[1]
for p in (MODEL, REPO, REPO / 'shared' / 'water' / 'watersim'):
    sys.path.insert(0, str(p))
from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402
import snowsense as S  # noqa: E402

L, WARM = S.L, S.WARM
WYS = list(range(2016, 2026))
STORM_DAYS = 3                       # "the biggest storm": the wettest 3 consecutive days, 1 Oct .. 30 Jun
GAP_DAYS = 30                        # a 30-day gauge outage
GAP_MONTHS = [None, 11, 12, 1, 2, 3, 4, 5]
LAPSE = -6.5                         # deg C per km, WaterSim's default TLAPSE
DD_DEFAULT = dict(TT=0.0, TTI=2.0, CFMAX=3.5, SFCF=1.0, CFR=0.05, CWH=0.1)
EB_DEFAULT = dict(TT=0.5, TTI=2.0, SFCF=1.0, CWH=0.05, ALB_MAX=0.85, ALB_MIN=0.45, ALB_DECAY=12.0, Z0=0.002,
                  TURB=1.0, GHEAT=0.173, CANOPY=1.0)
RECORDED = dict(warm=0.0, pscale=1.0, dz=0.0, lapse=True, drop=False, gap=0)
VARIANTS = [RECORDED,
            dict(RECORDED, warm=2.0),
            dict(RECORDED, warm=4.0, pscale=1.2),
            dict(RECORDED, pscale=0.7),
            dict(RECORDED, dz=500.0),
            dict(RECORDED, dz=-600.0, lapse=False),
            dict(RECORDED, drop=True),
            dict(RECORDED, gap=3),
            dict(RECORDED, warm=1.5, pscale=0.85, dz=300.0, drop=True, gap=5)]


def load_station(sid):
    """The pipeline's own reading of the CSV: gaps handled exactly as in training (snowsense.clean)."""
    df = pd.read_csv(MODEL / 'examples' / f'{sid}.csv', parse_dates=['date']).set_index('date').asfreq('D')
    c = S.clean(df)
    c['obs'] = df['SWE_obs'].shift(-1)          # SNOTEL SWE is start of day; the model gives end of day
    return c


def window(st, start, wy, s, elev, lat):
    """One water-year window with a what-if applied, from the float32 values the browser reads (float64 math)."""
    i0 = int((pd.Timestamp(f'{wy - 1}-08-01') - start).days)
    sl = slice(i0, i0 + L)
    P, Tm, Tn, Tx = (st[k][sl].astype(np.float64).copy() for k in ('P', 'T_mean', 'T_min', 'T_max'))
    mP, mT = st['mP'][sl].astype(np.float64).copy(), st['mT'][sl].astype(np.float64).copy()
    if s['drop']:
        a = storm_start(P)
        P[a:a + STORM_DAYS] = 0.0
    if s['gap']:
        a = gap_start(wy, s['gap'])
        b = a + GAP_DAYS
        P[a:b], mP[a:b], mT[a:b] = 0.0, 1.0, 1.0
        for v in (Tm, Tn, Tx):                  # linear between the days either side, as pandas interpolate does
            lo, hi = v[a - 1], v[b]
            v[a:b] = lo + (hi - lo) * np.arange(1, GAP_DAYS + 1) / (GAP_DAYS + 1)
        Tn[a:b] = np.minimum(Tn[a:b], Tm[a:b])
        Tx[a:b] = np.maximum(Tx[a:b], Tm[a:b])
    P = P * s['pscale']
    dT = s['warm'] + (LAPSE * s['dz'] / 1000.0 if s['lapse'] else 0.0)
    Tm, Tn, Tx = Tm + dT, Tn + dT, Tx + dT
    return dict(P=P, Tm=Tm, Tn=Tn, Tx=Tx, mP=mP, mT=mT, elev=elev + s['dz'], lat=lat,
                dates=pd.date_range(f'{wy - 1}-08-01', periods=L, freq='D'))


def storm_start(P):
    """First day of the wettest STORM_DAYS-day run between 1 Oct and 30 Jun (window days 61 .. 333)."""
    best, at = -1.0, WARM
    for a in range(WARM, WARM + 273 - STORM_DAYS + 1):
        t = float(P[a:a + STORM_DAYS].sum())
        if t > best + 1e-9:
            best, at = t, a
    return at


def gap_start(wy, k):
    m = GAP_MONTHS[k]
    y = wy - 1 if m >= 8 else wy
    return int((pd.Timestamp(f'{y}-{m:02d}-01') - pd.Timestamp(f'{wy - 1}-08-01')).days)


def nwy_days(wy):
    return int((pd.Timestamp(f'{wy}-10-01') - pd.Timestamp(f'{wy - 1}-10-01')).days)


def reading(model, w, wy):
    """Network forward, bands and annual products, exactly as SnowSense.predict does them for one window."""
    X = S.features(w['P'], w['Tm'], w['Tn'], w['Tx'], w['mP'], w['mT'], w['dates'], w['lat'], w['elev'])
    Q = model.forward(X[None].astype(np.float64))[0][WARM:WARM + nwy_days(wy)]
    k = model.k
    band = lambda j, kk=1.0: 100.0 * np.maximum(Q[:, 2] + kk * (Q[:, j] - Q[:, 2]), 0) ** 2
    mo = lambda s: (-1 if S.melt_out_idx(s) is None else S.melt_out_idx(s))
    med = band(2)
    r = dict(swe=med, lo68=band(1, k['daily']), hi68=band(3, k['daily']), lo90=band(0, k['daily']),
             hi90=band(4, k['daily']), peak=float(med.max()), peakDay=int(np.argmax(med)), melt=mo(med),
             peakLo68=float(band(1, k['peak']).max()), peakHi68=float(band(3, k['peak']).max()),
             peakLo90=float(band(0, k['peak']).max()), peakHi90=float(band(4, k['peak']).max()),
             meltLo68=mo(band(1, k['melt'])), meltHi68=mo(band(3, k['melt'])),
             meltLo90=mo(band(0, k['melt'])), meltHi90=mo(band(4, k['melt'])))
    return r


def physics(w, wy, snow):
    """WaterSim snow-only, one band at the station, default parameters, from 1 August with no snow."""
    import watersim as ws
    f = pd.DataFrame({'P': w['P'], 'T_mean': w['Tm'], 'T_min': w['Tn'], 'T_max': w['Tx']}, index=w['dates'])
    spec = {'id': 'station', 'area_km2': 1.0, 'lat': w['lat'], 'lon': -110.0, 'elev_mean_m': w['elev']}
    prm = DD_DEFAULT if snow == 'degree_day' else EB_DEFAULT
    o = ws.simulate(spec, f, f.index[0], f.index[-1], model='none', snow=snow, routing='none', pet='hargreaves',
                    params=dict(prm), spinup=0, labels=False)
    return o['ts']['SWE'].values[WARM:WARM + nwy_days(wy)]


def flat(r, eb, dd):
    """The parity vector; site/src/instruments/snowsense/parity.ts builds the same one."""
    head = [r[k] for k in ('peak', 'peakDay', 'melt', 'peakLo68', 'peakHi68', 'peakLo90', 'peakHi90',
                           'meltLo68', 'meltHi68', 'meltLo90', 'meltHi90')]
    return [float(v) for v in [*head, float(eb.max()), float(dd.max()), *r['swe'][::10], *r['lo68'][::30],
                               *r['hi90'][::30], *eb[::10], *dd[::10]]]


if __name__ == '__main__':
    out = instrument_dir('snowsense')
    model = S.SnowSense()
    raw = {k: v for k, v in np.load(MODEL / 'snowsense_weights.npz').items()}
    meta = dict(k=model.k, layers=model.layers, hidden=int(raw['gru.weight_hh_l0'].shape[1]),
                quantiles=[0.05, 0.16, 0.5, 0.84, 0.95], window_days=L, warmup_days=WARM)
    write_pack(out / 'weights', {k: v for k, v in raw.items() if k != 'k'}, dtype='f16', meta=meta)

    # The browser's weights: float16-rounded, used for every reference output below.
    browser = S.SnowSense()
    browser.w = {k: v.astype(np.float64) for k, v in roundtrip({k: v for k, v in raw.items() if k != 'k'}).items()}

    info = json.loads((MODEL / 'examples' / 'stations.json').read_text(encoding='utf-8'))
    summary = json.loads((MODEL / 'training' / 'final_summary.json').read_text(encoding='utf-8'))
    arrays, stations = {}, []
    for i, (sid, s) in enumerate(info.items()):
        c = load_station(sid)
        start = c.index[0]
        for k in ('P', 'T_mean', 'T_min', 'T_max', 'mP', 'mT', 'obs'):
            arrays[f'{k}{i}'] = c[k].values.astype(np.float32)
        stations.append(dict(id=sid, name=s['name'], state=s['state'], region=s['region'], lat=s['lat'],
                             elev_m=s['elev_m'], start=str(start.date()), days=len(c), wys=WYS))

        # The float64 path above must be SnowSense.predict itself on recorded weather (original weights, CSV values).
        st64 = {k: c[k].values for k in ('P', 'T_mean', 'T_min', 'T_max', 'mP', 'mT')}
        df = pd.read_csv(MODEL / 'examples' / f'{sid}.csv', parse_dates=['date']).set_index('date').asfreq('D')
        pred = model.predict(df, s['lat'], s['elev_m'], wys=[2019])['daily']['swe'].values
        mine = reading(model, window(st64, start, 2019, RECORDED, s['elev_m'], s['lat']), 2019)['swe']
        assert np.abs(pred - mine).max() < 1e-9, (sid, np.abs(pred - mine).max())
    write_pack(out / 'stations', arrays, dtype='f32', meta=dict(
        stations=stations, storm_days=STORM_DAYS, gap_days=GAP_DAYS, gap_months=GAP_MONTHS, lapse=LAPSE,
        dd_default=DD_DEFAULT, eb_default=EB_DEFAULT,
        exam=dict(n=summary['net']['n'], nse=summary['net']['nse'], eb=summary['eb_uncal']['nse'],
                  dd=summary['dd_uncal']['nse'], clim=summary['clim']['nse'], dd_cal=summary['dd_cal']['nse'])))

    reference = []
    rng = np.random.default_rng(20261118)
    for i, s in enumerate(stations):
        st = {k: arrays[f'{k}{i}'] for k in ('P', 'T_mean', 'T_min', 'T_max', 'mP', 'mT')}
        start = pd.Timestamp(s['start'])
        for wy in sorted(rng.choice(WYS, 2, replace=False).tolist()):
            for v in VARIANTS:
                w = window(st, start, wy, v, s['elev_m'], s['lat'])
                r = reading(browser, w, wy)
                eb, dd = physics(w, wy, 'energy_balance'), physics(w, wy, 'degree_day')
                reference.append(dict(example=i, variant=f"{s['id']} WY{wy} " + json.dumps(v, separators=(',', ':')),
                                      station=i, wy=wy, scenario=v, outputs=flat(r, eb, dd)))
    for stale in ('examples.json', 'examples.bin'):
        (out / stale).unlink(missing_ok=True)
    write_json(out / 'reference.json', dict(cases=reference, tolerance=1e-4))
    size = sum(p.stat().st_size for p in out.iterdir())
    print(f'{len(stations)} stations, {len(reference)} parity cases, {size / 1e6:.2f} MB in {out}')
