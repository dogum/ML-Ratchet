"""Bank v5 for r7_context (NOTEBOOK 2026-11-17, "r7 plan fixed before bank v5 is built").

simbank.py's buildings, cities, control priors and fault draws, with the dev-calibration wrappers from BuildingSim
1.0.1's `buildingsim.phase2` (OperatorJitter controller; log_commands on the faulted trend from the fault start, one
feedback draw per window). The fault starts on day 1 in p_fault_day1 (0.2) of windows, else day 5. Every row records
window id, day (1-11), run (0 healthy, 1 faulted), fault_day and a sample flag. Kept: healthy days 1-11 and faulted
days from the fault start. Sample rows (training / frozen targets) are v3's: faulted days, healthy days before the
fault and healthy days 10-11. The other healthy days are references only.

Usage: python simbank5.py <out.npz> <n_windows> <seed> <calib.json> [--worker j/n | --concat]
Shards of 100 windows go to <out>.parts/; worker j builds shards j, j+n, ...; --concat joins them.
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CI, sim_raw, to_day, hvac_energy  # noqa: E402
from simbank import WIN_DAYS, FAULT_DAY, AHU_SYSTEMS, weather, sample_fault, randomize_controls, effect  # noqa: E402
import buildingsim as bs  # noqa: E402
from buildingsim import phase2  # noqa: E402


def one_window(rng, specs_pool, calib, wid):
    spec = dict(specs_pool[rng.integers(len(specs_pool))])
    spec = json.loads(json.dumps(spec))
    spec = randomize_controls(spec, rng, calib)
    cities = list(bs.weather.EPW_CLIMATE_ZONE.keys())
    city = cities[rng.integers(len(cities))]
    w = weather(city)
    tz = w[1]['tz']
    doy = int(rng.integers(0, 365 - WIN_DAYS - 3))
    d0 = pd.Timestamp('2019-01-01') + pd.Timedelta(days=doy)          # first local day kept
    start = (d0 - pd.Timedelta(days=1)).tz_localize(tz).tz_convert('UTC').tz_localize(None)
    end = (d0 + pd.Timedelta(days=WIN_DAYS + 1)).tz_localize(tz).tz_convert('UTC').tz_localize(None)
    f, cls, par = sample_fault(rng, spec['system'])
    c = calib or {}
    shift = None
    if c.get('schedule_shift') and rng.random() < 0.06:
        f, cls, par = dict(type='schedule_shift'), 'schedule', 1.0
        shift = (['start', 'stop'][rng.integers(2)], int(rng.integers(8, 17)))
    op = c.get('operator')
    oseed = int(rng.integers(1 << 30)) if op else None
    fday = 1 if rng.random() < c.get('p_fault_day1', 0.2) else FAULT_DAY
    fstart = (d0 + pd.Timedelta(days=fday)).tz_localize(tz).tz_convert('UTC').tz_localize(None)
    fd = {k: v for k, v in f.items() if not k.startswith('_')}
    fd['start'] = str(fstart)
    if f['type'] == 'setpoint_override':
        from buildingsim.spec import fill_spec
        fs = fill_spec(json.loads(json.dumps(spec)))
        sp = fs['setpoints']
        fd['heat'] = sp['heat'] + f['_shift']
        fd['cool'] = max(sp['cool'] + f['_shift'], fd['heat'] + 1.0)
    ctl_h = ctl_f = None
    if op:
        ctl_h = phase2.OperatorJitter(oseed, tz, **op)
        ctl_f = phase2.OperatorJitter(oseed, tz, **op, shift=shift, shift_from=d0 + pd.Timedelta(days=fday))
    h = bs.simulate(spec, w, str(start), str(end), dt='15min', controller=ctl_h)
    if f['type'] == 'schedule_shift':
        fo = bs.simulate(spec, w, str(start), str(end), dt='15min', controller=ctl_f)
    else:
        fo = bs.simulate(spec, w, str(start), str(end), dt='15min', faults=[fd], controller=ctl_f)
    fo_log = fo
    if c.get('log_commands') and f['type'] != 'schedule_shift':
        tf = phase2.log_commands(h['trend'], fo['trend'], f, rng)       # one feedback draw per window
        idx = fo['trend'].index
        before = idx < pd.Timestamp(fstart).tz_localize('UTC') if idx.tz is not None else idx < pd.Timestamp(fstart)
        tf = tf.copy()
        tf.loc[before] = fo['trend'].loc[before]                        # the fault (and its logging) starts at fstart
        fo_log = dict(fo, trend=tf)
    rh, rf = sim_raw(h, tz), sim_raw(fo_log, tz)
    eh, ef = hvac_energy(h, tz), hvac_energy(fo, tz)
    rows = []
    label = fo['trend'].fault.iloc[-1] if f['type'] != 'schedule_shift' else f'schedule_shift:{shift}'
    for k in range(1, WIN_DAYS):
        day = d0 + pd.Timedelta(days=k)
        xh = to_day(rh, day)
        sel = (eh.index >= day) & (eh.index < day + pd.Timedelta(days=1))
        Eh = float(eh[sel].sum() * 0.25)
        meta = dict(system=spec['system'], use=spec['primary_use'], city=city, month=int(day.month),
                    weekday=int(day.day_of_week), area=float(spec.get('floor_area_m2') or 0), window=wid, day=k,
                    fault_day=fday)
        rows.append(dict(x=xh, cls=0, par=0.0, waste=0.0, eff=0.0, ftype='none', label='', E=Eh, run=0,
                         sample=(k < fday) or (k >= WIN_DAYS - 2), **meta))
        if k >= fday:
            xf = to_day(rf, day)
            Ef = float(ef[sel].sum() * 0.25)
            waste = (Ef - Eh) / max(Eh, 1.0)
            rows.append(dict(x=xf, cls=CI[cls], par=par, waste=waste, eff=effect(xh, xf), ftype=f['type'],
                             label=label, E=Ef, run=1, sample=True, **meta))
    return rows


def save_rows(rows, fn):
    if not rows:
        return
    d = dict(x=np.stack([r['x'] for r in rows]).astype(np.float16),
             cls=np.array([r['cls'] for r in rows], np.int16), par=np.array([r['par'] for r in rows], np.float32),
             waste=np.array([r['waste'] for r in rows], np.float32), eff=np.array([r['eff'] for r in rows], np.float32),
             E=np.array([r['E'] for r in rows], np.float32),
             ftype=np.array([r['ftype'] for r in rows]), label=np.array([r['label'] for r in rows]),
             system=np.array([r['system'] for r in rows]), use=np.array([r['use'] for r in rows]),
             city=np.array([r['city'] for r in rows]), month=np.array([r['month'] for r in rows], np.int8),
             weekday=np.array([r['weekday'] for r in rows], np.int8),
             window=np.array([r['window'] for r in rows], np.int32), day=np.array([r['day'] for r in rows], np.int8),
             run=np.array([r['run'] for r in rows], np.int8), sample=np.array([r['sample'] for r in rows], bool),
             fault_day=np.array([r['fault_day'] for r in rows], np.int8))
    np.savez_compressed(fn, **d)


def build(out, n_windows, seed, calib, worker=0, nworkers=1):
    parts = out + '.parts'
    os.makedirs(parts, exist_ok=True)
    specs = [s for s in bs.generate_buildings(max(400, n_windows // 4), seed=seed) if s['system'] in AHU_SYSTEMS]
    print('buildingsim', bs.__version__, 'specs', len(specs), flush=True)
    t0 = time.time()
    for si, p0 in enumerate(range(0, n_windows, 100)):
        if si % nworkers != worker:
            continue
        fn = f'{parts}/{p0:06d}.npz'
        if os.path.exists(fn):
            continue
        rows = []
        for i in range(p0, min(p0 + 100, n_windows)):
            rng = np.random.default_rng([seed, i])
            try:
                rows += one_window(rng, specs, calib, i)
            except Exception as e:  # noqa: BLE001  (a failed window is logged and skipped)
                print('window', i, 'failed:', repr(e)[:200], flush=True)
        save_rows(rows, fn + '.tmp.npz')
        os.replace(fn + '.tmp.npz', fn)
        print(f'shard {p0}: {min(p0 + 100, n_windows)} windows done, {time.time() - t0:.0f} s', flush=True)


def concat(out, n_windows):
    parts = out + '.parts'
    want = [f'{p0:06d}.npz' for p0 in range(0, n_windows, 100)]
    missing = [f for f in want if not os.path.exists(f'{parts}/{f}')]
    if missing:
        raise SystemExit(f'missing shards: {missing}')
    allp = [dict(np.load(f'{parts}/{f}', allow_pickle=True)) for f in want]
    cat = {k: np.concatenate([a[k] for a in allp]) for k in allp[0]}
    np.savez_compressed(out, **cat)
    print('wrote', out, len(cat['cls']), 'rows', len(np.unique(cat['window'])), 'windows', flush=True)


if __name__ == '__main__':
    out, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    calib = json.load(open(sys.argv[4]))
    if '--concat' in sys.argv:
        concat(out, n)
    else:
        j, nw = 0, 1
        if '--worker' in sys.argv:
            j, nw = (int(v) for v in sys.argv[sys.argv.index('--worker') + 1].split('/'))
        build(out, n, seed, calib, j, nw)
