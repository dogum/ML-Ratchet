"""Phase 2 wrappers upstreamed into BuildingSim / BuildingData: each port reproduces the model's own wrapper.

    python checks/check_phase2.py            # all four models, one subprocess each (their training folders clash:
                                             # two have a common.py) -> checks/out/phase2_equivalence.json
    python checks/check_phase2.py loadcast   # one model -> checks/out/phase2_loadcast.json

The models' wrappers are imported read-only from models/<id>/training (v1.0 code). For each case the JSON gives the
largest absolute difference between the model's output and the port's (0.0 = identical). Weather: bundled TMYx
files, BDG2 dev site weather, RECS dev respondents, 2017 tariffs; ERCOT RTP needs GridData ($ML_RATCHET_GRID).
No exam unit is read.
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/building
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, BLDG_CODE)
from _common import write, BLDG_ROOT  # noqa: E402,F401
import buildingsim as bs  # noqa: E402
from buildingsim import phase2 as P  # noqa: E402

REPO = os.path.dirname(os.path.dirname(BLDG_CODE))
SOURCES = {'loadcast': ['lcsim.py', 'lc_common.py', 'calib.py'],
           'buildingdetective': ['wrappers.py', 'common.py', 'simone.py'],
           'faultfinder': ['simbank.py', 'noise.py', 'common.py'],
           'thermopilot': ['tp/twins.py', 'tp/env.py']}


def _train(model):
    d = os.path.join(REPO, 'models', model, 'training')
    if d not in sys.path:
        sys.path.insert(0, d)
    return d


def _src_state(model):
    files = [f'models/{model}/training/{f}' for f in SOURCES[model]]
    r = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *files], cwd=REPO)
    head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return dict(files=files, commit=head, unchanged_from_commit=r.returncode == 0)


def maxdiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return float('inf')
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float('inf')
    m = ~np.isnan(a)
    return float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0


def ts_diff(o1, o2, cols=None):
    """Largest abs difference over the float columns of out['ts'] (and the summary kWh_by_meter)."""
    t1, t2 = o1['ts'], o2['ts']
    cols = cols or [c for c in t1 if t1[c].dtype.kind == 'f']
    d = max(maxdiff(t1[c].values, t2[c].values) for c in cols)
    m1, m2 = o1['summary']['kWh_by_meter'], o2['summary']['kWh_by_meter']
    d = max(d, max(abs(m1[k] - m2.get(k, np.nan)) for k in m1))
    return d


def spec_equal(a, b):
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def tmy(city):
    return bs.read_epw(bs.epw_path(city))


# ------------------------------------------------------------------------------------------------- LoadCast
def loadcast():
    _train('loadcast')
    import lc_common as C
    import lcsim as L
    import calib as CAL
    import buildingdata as bd
    from buildingsim import holidays as H
    res = dict(source=_src_state('loadcast'))
    # 1. calendars
    nd = 0
    for c in ('US', 'CA', 'UK', 'IE'):
        for y in range(2015, 2021):
            nd += C.holiday_dates(c, y) != H.holiday_dates(c, y)
    res['holiday_calendars'] = dict(cases='4 countries x 2015-2020', differing=int(nd))
    # 2. simulate with the country calendar: LoadCast's monkeypatch vs options={'holidays': 'auto'}; Robin (UK),
    #    Easter 2016 (Good Friday 25 Mar, Easter Monday 28 Mar), a Robin dev building's metadata as an office
    w, m = C.site_weather('Robin')
    base = bd.bdg2_spec('Robin_lodging_Renea')
    spec = P.spec_from_meters(dict(base, primary_use='office'), ['electricity'], {})
    s0, e0 = '2016-03-14', '2016-04-04'
    bs.clear_caches()
    L.use_country(m['tz'])
    o_lc = bs.simulate(spec, (w, m), s0, e0, dt='1h', record_trend=False)
    L.use_country('America/New_York')            # back to the v1.0 calendar
    bs.clear_caches()
    o_up = bs.simulate(spec, (w, m), s0, e0, dt='1h', record_trend=False, options={'holidays': 'auto'})
    o_v1 = bs.simulate(spec, (w, m), s0, e0, dt='1h', record_trend=False)
    res['holiday_simulation'] = dict(
        case='Robin (Europe/London) office twin, 14 Mar - 4 Apr 2016, 1 h', max_abs_diff_vs_model=ts_diff(o_lc, o_up),
        holiday_days_flag_on=int(o_up['sim'].wx['cal']['holiday'].sum() // 24),
        holiday_days_v1=int(o_v1['sim'].wx['cal']['holiday'].sum() // 24),
        electricity_kWh=dict(model=o_lc['summary']['kWh_by_meter']['electricity'],
                             port=o_up['summary']['kWh_by_meter']['electricity'],
                             v1_calendar=o_v1['summary']['kWh_by_meter']['electricity']))
    # 3. spec maker over every dev building x its calibration grid
    meta = bd.bdg2_meta(split='dev')
    n = bad = 0
    for _, b in meta.iterrows():
        meters = C.meters_of(b['meters'])
        bspec = bd.bdg2_spec(b['id'])
        for kn in CAL.grid_for(meters) + [P.CAMPUS_KNOBS, dict(sys='keep', elec=1.3), {}]:
            n += 1
            bad += not spec_equal(L.make_spec(bspec, meters, kn), P.spec_from_meters(bspec, meters, kn))
    res['spec_from_meters'] = dict(cases=f'{n} (52 dev buildings x calib.grid_for knobs + campus/elec/none)',
                                   differing=int(bad))
    # 4. meter scaling
    rng = np.random.default_rng(1)
    x = {mt: rng.gamma(2, 10, 500) for mt in C.METERS}
    x['_sim_end_uses'] = None
    kn = {f'{c}_{mt}': float(rng.uniform(0, 3)) for mt in C.METERS[:4] for c in 'ab'}
    a, b2 = L.post(x, kn, 12345.0), P.scale_meters(x, kn, 12345.0)
    res['scale_meters'] = dict(max_abs_diff=max(maxdiff(a[k], b2[k]) for k in a), same_keys=sorted(a) == sorted(b2))
    # 5. campus knobs reproduce the dev summary's modes
    cs = json.load(open(os.path.join(REPO, 'models', 'loadcast', 'training', 'calib_summary.json')))
    res['campus_knobs'] = dict(CAMPUS_KNOBS=P.CAMPUS_KNOBS, dev_counts=cs['knobs'],
                               dev_meter_medians={k: (v['a_med'], v['b_W_m2_med']) for k, v in cs['meters'].items()},
                               DEV_METER_SCALE_equal=all(tuple(P.DEV_METER_SCALE[k]) == (v['a_med'], v['b_W_m2_med'])
                                                         for k, v in cs['meters'].items()))
    return res


# ------------------------------------------------------------------------------------------------- BuildingDetective
def detective():
    _train('buildingdetective')
    import wrappers as W
    res = dict(source=_src_state('buildingdetective'))
    cal = json.load(open(os.path.join(REPO, 'models', 'buildingdetective', 'training', 'calibration.json')))
    res['constants'] = dict(W1_equal=cal['intensity'] == P.W1_INTENSITY,
                            W2_equal=all(cal['campus'][k] == P.W2_CAMPUS[k] for k in P.W2_CAMPUS),
                            W3_equal=all(tuple(W.W3[k]) == tuple(P.W3_BASE[k]) if isinstance(W.W3[k], tuple)
                                         else W.W3[k] == P.W3_BASE[k] for k in P.W3_BASE))
    # W1 on every use type
    bad = 0
    uses = ['office', 'education', 'lodging', 'retail', 'healthcare', 'assembly', 'warehouse', 'residential_single',
            'residential_multi', 'food', 'lab', 'other']
    specs = {u: bs.generate_buildings(2, seed=7, primary_use=u) for u in uses}
    for u, ss in specs.items():
        for s in ss:
            f1, f2 = bs.fill_spec(s), bs.fill_spec(s)
            W.apply_intensity(f1, cal['intensity'][u])
            P.apply_intensity(f2, P.W1_INTENSITY[u])
            bad += not spec_equal(f1, f2)
    res['W1_apply_intensity'] = dict(cases=sum(len(v) for v in specs.values()), differing=int(bad))
    # W2: model = district_eff + '_eff_chw' then chilled-water meter rescaled (simone.py); port = district_eff_chw
    w = tmy('Chicago')
    rows = []
    for i, (u, kind) in enumerate([('office', 'both'), ('education', 'cool'), ('lab', 'heat'), ('healthcare', None),
                                   ('office', None)]):
        s = bs.generate_buildings(1, seed=40 + i, primary_use=u, system='vav_chiller_boiler')[0]
        f1, f2 = bs.fill_spec(s), bs.fill_spec(s)
        f1, k1 = W.make_campus(f1, np.random.default_rng(100 + i), cal, kind=kind)
        f2, k2 = P.make_campus(f2, np.random.default_rng(100 + i), kind=kind)
        o1 = bs.simulate(f1, w, '2019-07-01', '2019-07-15', dt='1h', record_trend=False)
        o2 = bs.simulate(f2, w, '2019-07-01', '2019-07-15', dt='1h', record_trend=False)
        r = f1['hvac']['district_eff'] / f1['hvac']['_eff_chw']
        W.campus_meters(o1['ts'], f1)
        o1['summary']['kWh_by_meter']['chilled_water'] *= r
        cols = [c for c in o1['ts'] if c.startswith('meter_')]
        c1, c2 = o1['summary']['kWh_by_meter']['chilled_water'], o2['summary']['kWh_by_meter']['chilled_water']
        rows.append(dict(use=u, kind=str(k1), kind_port=str(k2), district_eff=f2['hvac']['district_eff'],
                         district_eff_chw=f2['hvac']['district_eff_chw'],
                         chw_kWh_model=c1, chw_kWh_port=c2, rel_diff_chw=abs(c1 - c2) / max(c1, 1e-9),
                         max_abs_diff_meters_kW=ts_diff(o1, o2, cols), ledger_ok_port=bool(o2['ledger']['ok']),
                         ledger_closure_port=o2['ledger']['closure_rel']))
    res['W2_make_campus'] = dict(case='generated VAV buildings -> campus, Chicago TMYx 1-15 Jul, 1 h', rows=rows)
    # W3
    rng = np.random.default_rng(5)
    diffs = []
    for i in range(20):
        mt = {'elec': rng.gamma(3, 50, 300), 'fuel': rng.gamma(2, 5, 300), 'dheat': rng.gamma(1, 20, 300),
              'dcool': rng.gamma(2, 30, 300)}
        if i % 4 == 0:
            mt['dheat'] = np.zeros(300)
        a = W.w3_base_loads(mt, np.random.default_rng(i), commercial=i % 3 > 0)
        b = P.flat_base_loads(mt, np.random.default_rng(i), commercial=i % 3 > 0)
        names = {'elec': 'electricity', 'fuel': 'gas', 'dheat': 'steam', 'dcool': 'chilled_water'}
        c = P.flat_base_loads({names[k]: v for k, v in mt.items()}, np.random.default_rng(i), commercial=i % 3 > 0)
        diffs.append(max(max(maxdiff(a[k], b[k]) for k in a), max(maxdiff(a[k], c[names[k]]) for k in a)))
    res['W3_flat_base_loads'] = dict(cases='20 random meter sets (Detective and BuildingSim channel names)',
                                     max_abs_diff=max(diffs))
    # label cleaning
    import common as DC
    Y = np.random.default_rng(3).normal(0, 1, (400, len(DC.REG))).astype(np.float32) * 20
    Y[::7, DC.REG.index('base_share')] = -8e12
    Yc = DC.clean_labels(Y)
    d = 0.0
    for i in range(len(Y)):
        y = P.clean_labels({k: Y[i, DC.REG.index(k)] for k in ('base_share', 'Tb_heat', 'Tb_cool')})
        for k in y:
            d = max(d, maxdiff([Yc[i, DC.REG.index(k)]], [y[k]]))
    res['clean_labels'] = dict(cases='400 label rows, every 7th base_share = -8e12', max_abs_diff=d)
    return res


# ------------------------------------------------------------------------------------------------- FaultFinder
def faultfinder():
    _train('faultfinder')
    import common as FC
    import simbank as SB
    import noise as NZ
    res = dict(source=_src_state('faultfinder'))
    w = tmy('Atlanta')
    tz = w[1]['tz']
    spec = bs.generate_buildings(1, seed=3, primary_use='office', system='packaged_rtu')[0]
    vav = bs.generate_buildings(1, seed=4, primary_use='office', system='vav_chiller_boiler')[0]
    day = pd.Timestamp('2019-07-10')
    s0 = (day - pd.Timedelta(days=1)).tz_localize(tz).tz_convert('UTC').tz_localize(None)
    e0 = (day + pd.Timedelta(days=2)).tz_localize(tz).tz_convert('UTC').tz_localize(None)
    fs = str(day.tz_localize(tz).tz_convert('UTC').tz_localize(None))
    h = {}
    for nm, sp in (('rtu', spec), ('vav', vav)):
        h[nm] = bs.simulate(sp, w, str(s0), str(e0), dt='15min')
    # log_commands
    rows = []
    cases = [('rtu', dict(type='oa_damper_stuck', position=1.0)), ('vav', dict(type='cooling_valve_stuck', position=0.3)),
             ('vav', dict(type='heating_valve_stuck', position=1.0)), ('vav', dict(type='cooling_valve_leak', leak=0.4)),
             ('rtu', dict(type='oa_damper_leak', leak=0.25)), ('vav', dict(type='coil_fouling', severity=0.5))]
    for j, (nm, f) in enumerate(cases):
        sp = spec if nm == 'rtu' else vav
        fo = bs.simulate(sp, w, str(s0), str(e0), dt='15min', faults=[dict(f, start=fs)])
        for seed in (j, 100 + j):
            xh, xf = FC.to_day(FC.sim_raw(h[nm], tz), day), FC.to_day(FC.sim_raw(fo, tz), day)
            ff = SB.log_commands(xh, xf, f, np.random.default_rng(seed))
            tf = P.log_commands(h[nm]['trend'], fo['trend'], f, np.random.default_rng(seed))
            up = FC.to_day(FC.sim_raw(dict(trend=tf, sim=fo['sim']), tz), day)
            changed = int(np.sum(np.nan_to_num(ff - xf) != 0))
            rows.append(dict(system=nm, fault=f['type'], seed=seed, bins_changed_by_model=changed,
                             max_abs_diff=maxdiff(ff, up)))
    res['log_commands'] = dict(case='Atlanta TMYx 10 Jul, 15 min, fault from midnight', rows=rows,
                               max_abs_diff=max(r['max_abs_diff'] for r in rows))
    # damper characteristic
    rng0 = np.random.default_rng(9)
    frac = rng0.uniform(0, 1, (50, 96)).astype(np.float32)
    frac[:, ::11] = np.nan
    d = 0.0
    for i in range(50):
        a = NZ.Noise.damper_command(frac[i], np.random.default_rng(i))
        b = P.damper_command(frac[i], np.random.default_rng(i))
        d = max(d, maxdiff(a, b))
    x = FC.to_day(FC.sim_raw(h['rtu'], tz), day)
    di = 0.0
    for i in range(20):
        a = NZ.Noise.damper_command(x[FC.RI['damper_oa']], np.random.default_rng(i))
        t2 = P.installed_damper(h['rtu']['trend'], np.random.default_rng(i))
        b = FC.to_day(FC.sim_raw(dict(trend=t2, sim=h['rtu']['sim']), tz), day)[FC.RI['damper_oa']]
        di = max(di, maxdiff(a, b))
    res['damper_command'] = dict(cases='50 random days + 20 draws on a simulated RTU day', max_abs_diff=d,
                                 max_abs_diff_trend=di)
    # static offsets (Noise.apply with only the offsets on)
    tmp = os.path.join(BLDG_ROOT, 'checks', 'out', '_phase2_noise.npz')
    np.savez(tmp, **{'res_' + c: np.zeros((1, FC.NT), np.float32) for c in NZ.RES_CH})
    d = {True: 0.0, False: 0.0}
    for big in (True, False):
        nz = NZ.Noise(tmp, offsets=True, residual=False, patterns=False, fanoff=False, damper_char=False, big_offsets=big)
        for i in range(20):
            a = nz.apply(x, np.random.default_rng(i))
            t2, _ = P.static_offsets(h['rtu']['trend'], np.random.default_rng(i), large=big)
            b = FC.to_day(FC.sim_raw(dict(trend=t2, sim=h['rtu']['sim']), tz), day)
            T = [FC.RI[c] for c in FC.TEMPS]
            d[big] = max(d[big], maxdiff(a[T], b[T]))
    os.remove(tmp)
    res['static_offsets'] = dict(cases='20 draws each, v4 (large) and v3 sizes', max_abs_diff_large=d[True],
                                 max_abs_diff_v3=d[False])
    # operator jitter
    rows = []
    for seed in (1, 2, 3):
        o1 = bs.simulate(spec, w, '2019-03-04', '2019-03-11', dt='15min', controller=SB.Operator(seed, tz))
        o2 = bs.simulate(spec, w, '2019-03-04', '2019-03-11', dt='15min', controller=P.OperatorJitter(seed, tz))
        o0 = bs.simulate(spec, w, '2019-03-04', '2019-03-11', dt='15min')
        rows.append(dict(seed=seed, max_abs_diff=ts_diff(o1, o2),
                         fan_hours_jitter=float(o2['trend']['fan_status'].sum() / 4),
                         fan_hours_none=float(o0['trend']['fan_status'].sum() / 4)))
    res['operator_jitter'] = dict(case='RTU office, Atlanta TMYx 4-11 Mar, 15 min', rows=rows)
    return res


# ------------------------------------------------------------------------------------------------- ThermoPilot
def thermopilot():
    _train('thermopilot')
    from tp import twins as TW, env as EN
    import buildingdata as bd
    from buildingsim import stock
    res = dict(source=_src_state('thermopilot'))
    x = bd.recs(split='dev')
    x = x[x.system.isin(list(TW.SUPPORTED_SYSTEMS))]
    rng = np.random.default_rng(20261117)
    idx = rng.choice(len(x), 24, replace=False)
    n = bad = none = 0
    for i in idx:
        row = x.iloc[int(i)]
        seed = int(rng.integers(1 << 30))
        a, b = TW.recs_twin(row, seed=seed), stock.recs_to_spec(row, seed=seed, stable=False)
        n += 1
        if a is None or b is None:
            bad += (a is None) != (b is None)
            none += a is None
            continue
        bad += not (spec_equal(a[0], b[0]) and a[1] == b[1])
    res['recs_twins'] = dict(cases=f'{n} RECS dev respondents', no_twin=int(none), differing=int(bad))
    # tariffs: every URDB tariff + a synthetic one over 2017 at Bull (America/Chicago), and the bill
    times = pd.date_range('2017-01-01 06:00', '2018-01-01 06:00', freq='h', tz='UTC', inclusive='left')
    tz = 'America/Chicago'
    d = 0.0
    load = np.random.default_rng(2).gamma(2, 2, len(times))
    ym = times.tz_convert(tz).year * 12 + times.tz_convert(tz).month
    bd_ = 0.0
    keys = list(bd.tariffs()) + [EN.synth_tariff(5, demand=True)]
    for k in keys:
        a, b = EN.price_arrays(k, times, tz), bd.tariff_arrays(k, times, tz)
        d = max(d, max(maxdiff(a[c], b[c]) for c in a))
        e1, e2 = EN.bill_arrays(a, load, ym.values), bd.bill_arrays(b, load, ym.values)
        bd_ = max(bd_, abs(e1[0] - e2[0]), abs(e1[1] - e2[1]))
    res['tariff_arrays'] = dict(cases=f'{len(keys)} tariffs (19 URDB + 1 synthetic), 2017 hourly, {tz}',
                                max_abs_diff=d, bill_max_abs_diff=bd_)
    # DST: tariff_prices(utc=True) equals the tz frame on every hour
    neq = nfr = 0
    for k in bd.tariffs():
        for z in sorted(set(bd.SITE_TZ.values())):
            for y in (2016, 2017, 2018, 2019):
                u = bd.tariff_prices(k, y, tz=z, utc=True)
                r = bd.tariff_prices(k, y, tz=z).reindex(u.index.tz_convert(z))
                nfr += 1
                neq += not np.array_equal(r.values, u.values)
    res['tariff_prices_dst'] = dict(cases=f'{nfr} frames (19 tariffs x 7 time zones x 2016-2019)', differing=neq)
    try:
        a, b = EN.rtp_prices(times), bd.rtp_prices(times)
        res['rtp_prices'] = dict(case='ERCOT HB_HOUSTON, 2017 weather hours -> 2019 prices', max_abs_diff=maxdiff(a, b),
                                 mean_usd_kWh=float(np.mean(b)))
    except Exception as e:  # noqa: BLE001
        res['rtp_prices'] = dict(skipped=f'GridData not available: {e!r}'[:200])
    return res


PARTS = dict(loadcast=loadcast, detective=detective, faultfinder=faultfinder, thermopilot=thermopilot)


def main(argv):
    if argv and argv[0] != 'merge':
        t0 = time.time()
        r = PARTS[argv[0]]()
        r['wall_s'] = time.time() - t0
        write(f'phase2_{argv[0]}.json', r)
        return
    if not argv:          # 'merge' only combines the parts already written
        procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), p]) for p in list(PARTS)[:3]]
        for p in procs:
            p.wait()
        subprocess.run([sys.executable, os.path.abspath(__file__), list(PARTS)[3]])
    out = {}
    for p in PARTS:
        f = os.path.join(BLDG_ROOT, 'checks', 'out', f'phase2_{p}.json')
        out[p] = json.load(open(f)) if os.path.exists(f) else dict(error='part failed (see its log)')
    write('phase2_equivalence.json', dict(what=__doc__.strip().splitlines()[0], buildingsim=bs.__version__,
                                          physics=bs.PHYSICS_VERSION, **out))


if __name__ == '__main__':
    main(sys.argv[1:])
