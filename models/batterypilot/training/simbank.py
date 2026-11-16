"""Build banks of simulated price path-years (resumable; one npz per path).

    python3 simbank.py <bank_name> <n_paths> <seed0> [--ba ERCO] [--years 2018,2020,2022] [--sim v2]
                       [--calib auto|<file>] [--part k/n] [--prefix p]

GridSim physics comes from $GRIDSIM_PHYSICS (1.1 by default; 1.0 = as released).  --calib auto reads the scarcity
calibration of that physics version (calib_file()); --part k/n builds only the paths with index % n == k (parallel
builds of one bank give the same files as one build).

Scenario per path: a training-input year (real inputs, fleet of that year) with random demand growth, wind and
solar build-out, system storage, gas price level, retirements, ORDC design (X, cap) and jittered dev-calibrated
scarcity / noise parameters.  Scenarios with absurd scarcity (> 400 h above $1000 or mean > $300) are redrawn.
"""
import argparse, json, os, sys, time
import numpy as np
import _paths  # noqa: E402  (this folder, GridSim and GridData on sys.path; outputs to WORK)
from pricesim import Engine, load_inputs, fundamentals, prices, DEFAULTS, physics_version
from battery import Battery
from strategies import pf_daily
from features import rolling_scale, _rollmax

ROOT = _paths.WORK   # was the sandbox workspace grid/batterypilot


def calib_file(version=None):
    """BatteryPilot's dev calibration (calib.py) for a GridSim physics version.  1.0: WORK/data/calibration.json if
    present, else the release's (training/checks/calibration.json).  1.1: WORK/data/calibration_v1_1.json."""
    v = version or physics_version()
    if v == '1.0':
        fn = f'{ROOT}/data/calibration.json'
        return fn if os.path.exists(fn) else os.path.join(_paths.SRC, 'checks', 'calibration.json')
    return f'{ROOT}/data/calibration_v{v.replace(".", "_")}.json'


def calibrated(sim='v2', path=None):
    if sim == 'v1':     # GridSim energy prices only: no scarcity curve, no cold outages, no RT noise
        # (and no physics 1.2 RT scarcity: scar_scale 0, so a v1 bank is the same on 1.1 and 1.2)
        return dict(DEFAULTS, mu=0.0, sigma=1.0, X=-1e9, cold_fmax=0.0, rt_noise_sd=0.0, floor_share=0.8,
                    scar_scale=0.0)
    c = json.load(open(path or calib_file()))['calibrated']
    return dict(DEFAULTS, **c)


def sample_caiso(rng, ba, year, inp, cal):
    """CAISO-like family on the real CISO fleet of a training-input year.  Centred on the two settings checked
    against CAISO's dev months (Aug-Dec 2023): solar output x1.6 and fixed imports x0.6 reproduce the duck curve
    (simulated hourly medians 18-25 $/MWh midday vs 21-25 real, 59-61 evening vs 66-72 real)."""
    peak = float(inp.d.max())
    s = dict(ba=ba, year=year)
    s['demand_scale'] = float(rng.uniform(0.97, 1.1))
    s['wind_scale'] = float(rng.uniform(0.8, 1.5))
    share = float(rng.uniform(0.35, 0.8))
    s['solar_scale'] = float(max(1.0, share * peak / max(inp.solar.max(), 1.0)))
    s['imports_scale'] = float(rng.uniform(0.4, 0.95))
    s['storage_MW'] = float(rng.uniform(0, 0.3) * peak)
    s['storage_h'] = float(rng.choice([2.0, 4.0, 4.0]))
    s['gas_scale'] = float(np.exp(rng.uniform(np.log(0.5), np.log(1.8))))
    s['carbon'] = float(rng.uniform(10, 40))
    s['retire_frac'] = float(rng.uniform(0, 0.08))
    s['X'] = float(rng.choice([1000.0, 1500.0]))
    s['voll'] = s['cap'] = float(rng.choice([1000.0, 2000.0]))
    s['mu'] = float(0.5 * cal['mu'] + rng.normal(0, 300))       # half-size system: reserve scales halved
    s['sigma'] = float(0.5 * cal['sigma'] * rng.uniform(0.7, 1.3))
    s['rt_noise_sd'] = float(cal['rt_noise_sd'] * rng.uniform(0.6, 1.5))
    s['ramp_k'] = float(2 * cal['ramp_k'] * rng.uniform(0.5, 1.5))
    s['cold_fmax'] = 0.0
    s['seed'] = int(rng.integers(1 << 30))
    return s


def sample_scenario(rng, ba, year, inp, cal, sim):
    if ba == 'CISO':
        return sample_caiso(rng, ba, year, inp, cal)
    peak = float(inp.d.max())
    s = dict(ba=ba, year=year)
    s['demand_scale'] = float(rng.uniform(0.97, 1.22))
    s['wind_scale'] = float(rng.uniform(0.8, 1.6))
    share = float(rng.uniform(0.02, 0.45))
    s['solar_scale'] = float(max(1.0, share * peak / max(inp.solar.max(), 1.0)))
    s['storage_MW'] = float(rng.uniform(0, 0.2) * peak * s['demand_scale'])
    s['storage_h'] = float(rng.choice([1.0, 2.0, 2.0, 4.0]))
    s['gas_scale'] = float(np.exp(rng.uniform(np.log(0.6), np.log(1.8))))
    s['retire_frac'] = float(rng.uniform(0, 0.08))
    if sim != 'v1':
        s['X'] = float(rng.choice([2000.0, 3000.0]))
        vc = [(9000.0, 9000.0), (5000.0, 5000.0)][rng.integers(2)]
        s['voll'], s['cap'] = vc
        s['mu'] = float(cal['mu'] + rng.normal(0, 500))
        s['sigma'] = float(cal['sigma'] * rng.uniform(0.7, 1.3))
        s['rt_noise_sd'] = float(cal['rt_noise_sd'] * rng.uniform(0.6, 1.5))
        s['ramp_k'] = float(cal['ramp_k'] * rng.uniform(0.5, 1.5))
        s['cold_fmax'] = float(cal.get('cold_fmax', 0.0) * rng.uniform(0.2, 1.2))
    s['seed'] = int(rng.integers(1 << 30))
    if sim != 'v1' and physics_version() == '1.2':
        # v1.3: real years differ in unforecast scarcity more than one draw does (GridSim LIMITS I: 81-165 h a year
        # above $100 RT - DA on the training years): scale the start rate per path
        s['scar_scale'] = float(rng.uniform(0.6, 1.5))
    return s


def make_path(eng_cache, inputs, s, cal):
    key = (s['ba'], s['year'], round(s['retire_frac'], 3))
    if key not in eng_cache:
        eng_cache.clear()
        eng_cache[key] = Engine(s['ba'], s['year'], retire_frac=s['retire_frac'], seed=s['seed'])
    eng = eng_cache[key]
    inp = inputs[(s['ba'], s['year'])]
    F = fundamentals(eng, inp, s, cal, seed=s['seed'])
    o = prices(F, cal, s, seed=s['seed'])
    return o


def finish(o):
    """Arrays stored per path (+ per-day PF values used to normalise training windows)."""
    rt = o.rt.values.astype(np.float64); da = o.da.values.astype(np.float64)
    day = o.day.values
    out = dict(rt=rt.astype(np.float32), da=da.astype(np.float32), df=o.df.values.astype(np.float32),
               T=o['T'].values.astype(np.float32), hour=o.hour.values.astype(np.int16),
               day=day.astype(np.int32), dow=o.dow.values.astype(np.int16), doy=o.doy.values.astype(np.int16))
    out['ps'] = rolling_scale(da).astype(np.float32)
    out['dmax'] = _rollmax(o.df.values, 720).astype(np.float32)
    dl = np.unique(day); out['day_list'] = dl
    for d in (1, 2, 4):
        b = Battery(1.0, d, degradation=False)
        c, dis = pf_daily(rt, day, b)
        v = rt * (dis - c)
        out[f'pf{d}'] = np.array([v[day == x].sum() for x in dl], np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bank'); ap.add_argument('n', type=int); ap.add_argument('seed0', type=int)
    ap.add_argument('--ba', default='ERCO'); ap.add_argument('--years', default='2018,2020,2022')
    ap.add_argument('--sim', default='v2'); ap.add_argument('--calib', default='auto')
    ap.add_argument('--part', default='0/1'); ap.add_argument('--prefix', default='p')
    a = ap.parse_args()
    k, nparts = (int(x) for x in a.part.split('/'))
    cfile = None if a.sim == 'v1' else (calib_file() if a.calib == 'auto' else a.calib)
    od = f'{ROOT}/data/banks/{a.bank}'; os.makedirs(od, exist_ok=True)
    years = [int(y) for y in a.years.split(',')]
    cal = calibrated(a.sim, cfile)
    inputs = {}
    eng_cache = {}
    t0 = time.time()
    for i in range(a.n):
        fn = f'{od}/{a.prefix}{i:04d}.npz'
        if os.path.exists(fn) or i % nparts != k:
            continue
        rng = np.random.default_rng(a.seed0 * 100003 + i)
        year = years[i % len(years)]
        if (a.ba, year) not in inputs:
            inputs[(a.ba, year)] = load_inputs(a.ba, year)
        for tries in range(6):
            s = sample_scenario(rng, a.ba, year, inputs[(a.ba, year)], cal, a.sim)
            o = make_path(eng_cache, inputs, s, cal)
            h1k = int((o.rt > 1000).sum()); mean = float(o.rt.mean())
            if h1k <= 400 and mean <= 300:
                break
        out = finish(o)
        s.update(h1000=h1k, mean_rt=mean, mean_da=float(o.da.mean()), tries=tries, physics=physics_version(),
                 coal_adder=next(iter(eng_cache.values())).coal_adder,
                 calib=os.path.basename(cfile) if cfile else 'none',
                 **({'scar_fit': cal.get('scar_fit')} if physics_version() == '1.2' else {}),
                 pf2_kWyr=float(out['pf2'].sum() / 1000))
        np.savez(fn + '.tmp.npz', meta=json.dumps(s), **out)
        os.replace(fn + '.tmp.npz', fn)
        print(i, year, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items() if k not in ('ba',)},
              '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
