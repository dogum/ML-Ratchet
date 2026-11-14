"""Bank: self-committed (price-taker) and cogeneration units in simulated price backgrounds, through the emulators.

Price backgrounds:
  'gs'    : the system price of a GridSim bank job (generated fleet, unit-level MILP, real train-year inputs; 92 days)
  'ercot' : real ERCOT day-ahead hub-average prices of a TRAIN-INPUT year (2018, 2020, 2022) + Henry Hub daily
            (a 'real background' in the recipe's sense; never dev or exam years)
Also converts the GridSim bank's own unit-level units (mode 'gs') through the same emulators.

usage: python bank_dp.py <version> <first_shard> <n_shards> [units_per_shard]
"""
from common import *
import json, glob, time
import numpy as np, pandas as pd
from simunits import random_unit, TECHS
from selfcommit import schedule, outages
from emulate import EMU, draw_reporting, record, context, fuel_fn
from pdcore import truth
import calib

TECH_P = dict(coal_steam=0.2, gas_cc=0.3, gas_ct=0.3, gas_steam=0.2)
EFOR = dict(coal_steam=(0.03, 0.12), gas_cc=(0.02, 0.08), gas_ct=(0.02, 0.08), gas_steam=(0.03, 0.15))
PLANNED = dict(coal_steam=(0.0, 0.10), gas_cc=(0.0, 0.08), gas_ct=(0.0, 0.05), gas_steam=(0.0, 0.12))


def ercot_background(year):
    from realdata import prices
    assert year in (2018, 2020, 2022)
    p = prices('2018-01-01', '2023-01-01')
    p = p[p.index.year == year]
    return dict(key=f'ercot{year}', t=epoch_s(p.index), price=p.price.ffill().bfill().values,
                gas=p.gas.values, utc_off=-6)


UTC_OFF = dict(ERCO=-6, CISO=-8, PJM=-5, MISO=-6, SWPP=-6, ISNE=-5, NYIS=-5)


def gs_background(f):
    z = np.load(f)
    m = json.loads(str(z['meta']))
    t = fix_times(z['times'])
    days = (t - t[0]) // 86400
    gas = z['gas'][np.minimum(days, len(z['gas']) - 1)]
    return dict(key=os.path.basename(f)[:-4], t=t, price=z['price'].astype(float), gas=gas.astype(float),
                utc_off=UTC_OFF.get(m['ba'], -6), z=z, meta=m)


def make_unit(rng, tech, cal):
    u0 = random_unit(tech, rng)
    u = dict(tech=tech, pmax=u0['pmax_MW'], pmin=u0['pmin_MW'], hr_full=u0['hr_full'], beta=u0['beta'],
             curv=u0['curv'], vom=u0['vom'], start_heat=u0['start_heat_MMBtu_per_MW'],
             min_up=u0['min_up_h'], min_dn=u0['min_down_h'], ramp=u0['ramp_up_MW_per_min'] * 60)
    u['start_mult'] = float(np.exp(rng.normal(*cal['start_mult'][tech])))       # wear-and-tear on top of start fuel
    u['fc_noise'] = float(rng.uniform(*cal['fc_noise']))
    u['efor'] = float(rng.uniform(*EFOR[tech])); u['planned'] = float(rng.uniform(*PLANNED[tech]))
    u['kappa'] = float(rng.uniform(*cal['kappa'][tech]))
    return u


def simulate_unit(u, bg, rng, rep, cal, mode):
    T = len(bg['price'])
    out = outages(T, rng, u['efor'], u['planned'])
    if u['tech'] == 'coal_steam':
        fp = np.full(T, rng.uniform(1.4, 2.8))
    else:
        fp = bg['gas'] * np.exp(rng.normal(0, 0.1))
    if mode == 'cogen':
        lo_ = u['pmin'] / u['pmax']
        lvl = rng.uniform(lo_, 0.98) if cal.get('cogen_level') == 'wide' else rng.uniform(max(lo_, 0.5), 0.98)
        sig = rng.uniform(0.01, 0.08)
        e = rng.normal(0, sig, T); a = 0.0; P = np.empty(T)
        for t in range(T):
            a = 0.98 * a + e[t] * 0.2; P[t] = a
        Pd = np.clip(u['pmax'] * (lvl + P), u['pmin'], u['pmax'])
        on = ~out
        # short trips
        for _ in range(rng.poisson(T / 8760 * rng.uniform(*cal['cogen_trips']))):
            s = int(rng.integers(0, T)); on[s:s + int(rng.integers(2, 48))] = False
        return on, np.where(on, Pd, 0.0)
    days = (np.arange(T) // 24)
    nd = days.max() + 1
    pf = bg['price'] * np.exp(rng.normal(0, u['fc_noise'], nd)[days]) + rng.normal(0, 3 * u['fc_noise'], T) * 10
    F = fuel_fn(u)
    start_warm = u['start_heat'] * u['pmax'] * float(np.mean(fp)) * u['start_mult']
    # CHP units run as merchants still sell process steam: a credit lowers their effective marginal cost (v2)
    vom = u['vom'] - (rng.uniform(*cal['steam_credit']) if (rep.get('chp') and 'steam_credit' in cal) else 0.0)
    on, P = schedule(pf, fp, u['pmax'], u['pmin'], F, vom, start_warm, u['min_up'], u['min_dn'], u['ramp'],
                     outage=out, L=7, kappa=u.get('kappa', 0.0))
    return on, P


def save_shard(f, units):
    np.savez_compressed(f, units=np.array(json.dumps([{k: v for k, v in d.items() if k != 'rec'} for d in units])),
                        **{f'{k}_{i}': d['rec'][k] for i, d in enumerate(units) for k in ('op', 'P', 'HI', 'meas')})


def shard(version, sid, n_units):
    out = f'{DATA}/bank_{version}'
    os.makedirs(out, exist_ok=True)
    f = f'{out}/shard_{sid:04d}.npz'
    if os.path.exists(f):
        return
    cal = calib.get(version)
    rng = np.random.default_rng(10_000 + sid)
    gs_files = sorted(glob.glob(f'{DATA}/bank_gs/job_*.npz'))
    units = []
    t0 = time.time()
    # one background per shard
    use_gs = bool(gs_files) and rng.random() < cal['p_gs_background']
    gi = sid % len(gs_files) if cal['p_gs_background'] >= 1.0 else int(rng.integers(len(gs_files) or 1))
    bg = gs_background(gs_files[gi]) if use_gs else \
        ercot_background(int(rng.choice([2018, 2020, 2022])))
    tt = pd.to_datetime(bg['t'], unit='s')
    doy = tt.dayofyear.values; years = tt.year.values
    for k in range(n_units):
        tech = str(rng.choice(TECHS, p=[TECH_P[t] for t in TECHS]))
        u = make_unit(rng, tech, cal)
        rep = draw_reporting(tech, rng, cal['emu'])
        mode = 'cogen' if (rep['chp'] and rng.random() < cal['p_cogen_given_chp']) else 'merchant'
        on, Pd = simulate_unit(u, bg, rng, rep, cal, mode)
        rec = record(u, on, Pd, doy, rng, rep, cal['emu'])
        ctx = context(u, rec, years, rng, rep, cal['emu'])
        tr = truth(rec['op'], rec['P'], rec['HI'], rec['meas'])
        units.append(dict(u=u, rep=rep, mode=mode, ctx={**ctx, 'hr923': {str(a): b for a, b in ctx['hr923'].items()}},
                          truth={a: (b if not isinstance(b, float) or np.isfinite(b) else None) for a, b in tr.items()},
                          bg=bg['key'], t0=int(bg['t'][0]), utc_off=bg['utc_off'], rec=rec))
    # GridSim's own unit-level units of this background (central MILP commitment)
    if use_gs and cal['include_gs_units'] is True:
        z = bg['z']; specs = bg['meta']['specs']
        for j, sp in enumerate(specs):
            meta = dict(hr_full=sp.get('hr_full', 10.0), beta=sp.get('beta', 0.1), curv=sp.get('curv', 0.0))
            u = dict(tech=sp['tech'], pmax=sp['pmax'], pmin=sp['pmin'], start_heat=sp.get('start_heat', 5.0),
                     min_up=sp['ut'], min_dn=sp['dt'], ramp=sp['ru'], vom=sp.get('vom', 3.0), **meta)
            on = z['u'][:, j] > 0.5
            rep = draw_reporting(u['tech'], rng, cal['emu'])
            rec = record(u, on, z['p'][:, j], doy, rng, rep, cal['emu'])
            ctx = context(u, rec, years, rng, rep, cal['emu'])
            tr = truth(rec['op'], rec['P'], rec['HI'], rec['meas'])
            units.append(dict(u=u, rep=rep, mode='gs', ctx={**ctx, 'hr923': {str(a): b for a, b in ctx['hr923'].items()}},
                              truth={a: (b if not isinstance(b, float) or np.isfinite(b) else None) for a, b in tr.items()},
                              bg=bg['key'], t0=int(bg['t'][0]), utc_off=bg['utc_off'], rec=rec))
    save_shard(f, units)
    print(f'shard {sid}: {bg["key"]} {len(units)} units {time.time() - t0:.0f}s', flush=True)


def gs_shards(version):
    """Each GridSim bank job's own units once, through the version's emulators (shard name gs_<job>)."""
    cal = calib.get(version)
    out = f'{DATA}/bank_{version}'; os.makedirs(out, exist_ok=True)
    for gf in sorted(glob.glob(f'{DATA}/bank_gs/job_*.npz')):
        f = f'{out}/shard_gs_{os.path.basename(gf)[4:-4]}.npz'
        if os.path.exists(f):
            continue
        bg = gs_background(gf)
        rng = np.random.default_rng(int(os.path.basename(gf)[4:-4]) + 77)
        tt = pd.to_datetime(bg['t'], unit='s'); doy = tt.dayofyear.values; years = tt.year.values
        z = bg['z']; units = []
        for j, sp in enumerate(bg['meta']['specs']):
            meta = dict(hr_full=sp.get('hr_full', 10.0), beta=sp.get('beta', 0.1), curv=sp.get('curv', 0.0))
            u = dict(tech=sp['tech'], pmax=sp['pmax'], pmin=sp['pmin'], start_heat=sp.get('start_heat', 5.0),
                     min_up=sp['ut'], min_dn=sp['dt'], ramp=sp['ru'], vom=sp.get('vom', 3.0), **meta)
            rep = draw_reporting(u['tech'], rng, cal['emu'])
            rec = record(u, z['u'][:, j] > 0.5, z['p'][:, j], doy, rng, rep, cal['emu'])
            ctx = context(u, rec, years, rng, rep, cal['emu'])
            tr = truth(rec['op'], rec['P'], rec['HI'], rec['meas'])
            units.append(dict(u=u, rep=rep, mode='gs', ctx={**ctx, 'hr923': {str(a): b for a, b in ctx['hr923'].items()}},
                              truth={a: (b if not isinstance(b, float) or np.isfinite(b) else None) for a, b in tr.items()},
                              bg=bg['key'], t0=int(bg['t'][0]), utc_off=bg['utc_off'], rec=rec))
        save_shard(f, units)
        print('gs shard', f, len(units), flush=True)


if __name__ == '__main__':
    if sys.argv[2] == 'gs':
        gs_shards(sys.argv[1]); sys.exit()
    v, a, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    nu = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    for s in range(a, a + n):
        shard(v, s, nu)
