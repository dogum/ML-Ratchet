"""Dynamometer drive schedules -> trip format (1 Hz, SI, zero grade unless the schedule defines grade).
Sources: EPA dynamometer-drive-schedules page (txt), 40 CFR 1037 App. A/D via eCFR API, FASTSim 2.1.5 package resources."""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, re, json, glob, numpy as np, pandas as pd
ROOT = _VEH + '/data'
RAW = f'{ROOT}/raw/cycles'; OUT = f'{ROOT}/proc/cycles'; os.makedirs(OUT, exist_ok=True)
MPH = 0.44704


def read_txt(fn):
    raw = open(fn, 'rb').read()
    txt = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('latin1')
    head = ' '.join(txt.splitlines()[:4]).lower()
    unit = 'kmh' if ('km/h' in head or 'kph' in head or 'km/hr' in head) else 'mph'
    rows = []
    for line in txt.splitlines():
        p = re.split(r'[\s,]+', line.strip())
        if len(p) >= 2:
            try: rows.append((float(p[0]), float(p[1])))
            except ValueError: pass
    a = np.array(rows)
    return a[:, 0], a[:, 1] * (MPH if unit == 'mph' else 1 / 3.6), unit


def cfr_table(fn):
    t = open(fn).read()
    rows = re.findall(r'<TR>\s*<TD[^>]*>([^<]*)</TD>\s*<TD[^>]*>([^<]*)</TD>\s*</TR>', t)
    a = [(float(x.replace(',', '')), float(y.replace('−', '-').replace('−', '-'))) for x, y in rows if re.match(r'^[\d.,]+$', x.strip())]
    return np.array(a)


def to_1hz(t, v):
    t = np.asarray(t, float).copy()
    # some EPA files concatenate segments whose time restarts (e.g. JP 10-15): unwrap
    for i in range(1, len(t)):
        if t[i] <= t[i - 1] - 0.5:
            t[i:] += t[i - 1] + 1 - t[i]
    t = t - t[0]
    g = np.arange(0, int(np.floor(t[-1])) + 1, dtype=float)
    return g, np.interp(g, t, v)


def save(name, t, v, grade=None, T=25.0, meta=None):
    n = len(t)
    grade = np.zeros(n) if grade is None else grade
    s = np.r_[0, np.cumsum(0.5 * (v[1:] + v[:-1]))]
    elev = np.r_[0, np.cumsum(np.diff(s) * grade[1:])]
    rho = 101325 / (287.058 * (T + 273.15))
    nan = np.full(n, np.nan, np.float32)
    arr = dict(t=t, v=np.clip(v, 0, None), grade=grade, elev=elev, lat=nan, lon=nan, heading=nan,
               T_amb=np.full(n, T), wind_speed=np.zeros(n), wind_dir=np.zeros(n), rho_air=np.full(n, rho),
               surface=np.zeros(n, np.int8), fuel_rate=nan, engine_rpm=nan, batt_current=nan, batt_voltage=nan,
               soc=nan, aux_power=nan)
    arr = {k: (a.astype(np.float32) if k != 'surface' else a) for k, a in arr.items()}
    np.savez_compressed(f'{OUT}/{name}.npz', **arr)
    m = dict(trip_id=name, vehicle_id=None, source='cycle', date=None, distance_m=float(s[-1]), duration_s=float(n),
             T_amb_mean=T, split='train_noise', v_mean=float(s[-1] / n), v_max=float(v.max()), idle_frac=float(np.mean(v < 0.1)),
             notes='Dynamometer schedule; zero grade unless stated; nominal test temperature; no measured outputs.')
    m.update(meta or {})
    json.dump(m, open(f'{OUT}/{name}.json', 'w'), indent=1)
    return m


EPA_URL = 'https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules'
LD = [('UDDS', 'uddscol.txt', 25, 'light'), ('FTP', 'ftpcol.txt', 25, 'light'), ('HWFET', 'hwycol.txt', 25, 'light'),
      ('US06', 'us06col.txt', 25, 'light'), ('SC03', 'sc03col.txt', 35, 'light'), ('NYCC', 'epa-new-york-city-cycle.txt', 25, 'light'),
      ('LA92', 'la92dds.txt', 25, 'light'), ('LA92_short', 'la92shortdds.txt', 25, 'light'), ('IM240', 'im240col.txt', 25, 'light'),
      ('ECE_R15', 'ececol.txt', 25, 'light'), ('EUDC', 'eudccol.txt', 25, 'light'), ('EUDC_low', 'eudclcol.txt', 25, 'light'),
      ('JP_10_15', 'j1015col.txt', 25, 'light'), ('JP_10', 'jpn10col.txt', 25, 'light'), ('JP_15', 'jpn15col.txt', 25, 'light'),
      ('FTP_motorcycle_1B', 'ftpmc1b.txt', 25, 'motorcycle'), ('LA92_HD_class3', 'la92col.txt', 25, 'heavy')]
rows = []
for name, fn, T, duty in LD:
    t, v, unit = read_txt(f'{RAW}/{fn}')
    t, v = to_1hz(t, v)
    rows.append(save(name, t, v, T=T, meta=dict(duty=duty, source_url=EPA_URL, source_file=fn, unit_in_file=unit)))
# cold FTP (-7 C, 20 F) uses the FTP trace
t, v, _ = read_txt(f'{RAW}/ftpcol.txt'); t, v = to_1hz(t, v)
rows.append(save('FTP_cold_20F', t, v, T=-6.67, meta=dict(duty='light', source_url=EPA_URL, source_file='ftpcol.txt', notes='FTP trace at 20 F (-6.7 C) cold-CO test temperature.')))
# HD transient (40 CFR 1037 App. A)
a = cfr_table(f'{RAW}/cfr1037_appA.xml')
t, v = to_1hz(a[:, 0], a[:, 1] * MPH)
rows.append(save('HD_transient_1037A', t, v, meta=dict(duty='heavy', source_url='https://www.ecfr.gov/current/title-40/part-1037/appendix-Appendix%20A%20to%20Part%201037',
                                                   notes='40 CFR 1037 App. A Heavy-Duty Transient Test Cycle (the CARB HHDDT transient mode).')))
# GEM highway cruise 55 / 65 mph over the 40 CFR 1037 App. D grade profile (distance-based)
g = cfr_table(f'{RAW}/cfr1037_appD.xml')
for mph in (55, 65):
    vv = mph * MPH; L = g[-1, 0]
    t = np.arange(0, int(L / vv) + 1, dtype=float)
    grade = np.interp(t * vv, g[:, 0], g[:, 1] / 100.0)
    rows.append(save(f'HD_cruise_{mph}mph_1037D', t, np.full(len(t), vv), grade=grade,
                     meta=dict(duty='heavy', source_url='https://www.ecfr.gov/current/title-40/part-1037/appendix-Appendix%20D%20to%20Part%201037',
                               notes=f'Constant {mph} mph over the 40 CFR 1037 App. D grade profile (GEM highway cruise). Real tests start at speed.')))
t = np.arange(0, int(g[-1, 0] / (55 * MPH)) + 1, dtype=float)
rows.append(save('HD_cruise_55mph_flat', t, np.full(len(t), 55 * MPH), meta=dict(duty='heavy', source_url='https://www.ecfr.gov/current/title-40/part-1037',
                 notes='Constant 55 mph, zero grade, same length as the 1037 App. D profile (ICCT 55 mph flat test).')))
# FASTSim built-in cycles (Apache-2.0, NREL): WLTC class 3b, long-haul truck day with grade
for name, fn, duty in [('WLTC_3b', 'wltc_3b.csv', 'light'), ('FASTSim_long_haul', 'longHaulDriveCycle.csv', 'heavy')]:
    d = pd.read_csv(f'{RAW}/fastsim2_cycles/{fn}', encoding='utf-8-sig')
    t, v = to_1hz(d.cycSecs.values.astype(float), d.cycMps.values.astype(float))
    grade = np.interp(t, d.cycSecs.values - d.cycSecs.values[0], d.cycGrade.values.astype(float))
    rows.append(save(name, t, v, grade=grade, meta=dict(duty=duty, source_url='https://pypi.org/project/fastsim/2.1.5/',
                                                        source_file=f'fastsim/resources/cycles/{fn}',
                                                        notes='From FASTSim 2.1.5 package resources (Apache-2.0, NREL). Provenance of the long-haul trace is not documented in the package.')))
idx = pd.DataFrame(rows)
idx.to_parquet(f'{OUT}/index.parquet', index=False)
print(idx[['trip_id', 'duty', 'duration_s', 'distance_m', 'v_mean', 'v_max', 'idle_frac']].round(2).to_string())
