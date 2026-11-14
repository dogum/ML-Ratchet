"""Bank v0: GridSim unit-level MILP commitment of generated fleets, driven by real train-year inputs.

One job = one generated system x one 92-day window (random BA and train-input year).  Thermal units get
PlantDetective's wider random parameters (simunits.random_unit) on top of gridsim.generate_fleet; gas_ice units are
dropped (not a target, and 20 tiny integer units make the MILP slow).  Saves per job: system price, gas price, and per
coal/CC/CT/gas-steam unit its spec, hourly output, commitment, starts and fuel.

usage: python simbank_gs.py first_seed n_jobs
"""
from common import *
import time, json
import numpy as np, pandas as pd
import gridsim as gs
from siminputs import window
from simunits import random_unit, TECHS
import gswrap; gswrap.install()

OUT = f'{DATA}/bank_gs'
os.makedirs(OUT, exist_ok=True)


def job(seed):
    f = f'{OUT}/job_{seed:05d}.npz'
    if os.path.exists(f):
        return
    rng = np.random.default_rng(1000 + seed)
    peak = float(np.exp(rng.uniform(np.log(2500), np.log(9000))))
    fl = gs.generate_fleet(seed=seed, peak_MW=peak)
    fl = fl[fl.tech != 'gas_ice'].reset_index(drop=True)
    gas_k = float(np.exp(rng.normal(0, 0.25)))
    coal_p = float(rng.uniform(1.4, 2.8))
    for i in fl.index[fl.tech.isin(TECHS)]:
        u = random_unit(fl.at[i, 'tech'], rng, pmax=fl.at[i, 'pmax_MW'])
        for k in ('pmin_MW', 'ramp_up_MW_per_min', 'ramp_down_MW_per_min', 'min_up_h', 'min_down_h'):
            fl.at[i, k] = u[k]
        fl.at[i, 'heat_rate_curve'] = u['heat_rate_curve']
        fc = coal_p if fl.at[i, 'tech'] == 'coal_steam' else 3.0
        fl.at[i, 'start_cost_$'] = u['start_heat_MMBtu_per_MW'] * u['pmax_MW'] * fc
        fl.at[i, 'fuel_cost_$_per_MMBtu'] = fc
        fl.at[i, 'vom_$_per_MWh'] = u['vom']
        fl.at[i, 'pd_meta'] = json.dumps(dict(hr_full=u['hr_full'], beta=u['beta'], curv=u['curv'],
                                              start_heat=u['start_heat_MMBtu_per_MW'], vom=u['vom']))
    ba, year, w, gas = window(rng, 92)
    dem = w.d / w.d.max() * peak * rng.uniform(0.85, 1.0)
    t0 = time.time()
    r = gs.dispatch(fl, dem, wind_cf=w.wcf * rng.uniform(0.8, 1.0), solar_cf=w.scf * rng.uniform(0.8, 1.0),
                    cluster=False, outages='random', seed=seed, method='milp', prices={'NG': gas * gas_k},
                    lookahead_h=8, mip_gap=0.01, time_limit=30.0, must_run=())
    wall = time.time() - t0
    cl = r.cf.clusters
    keep = [j for j, c in enumerate(cl) if c['tech'] in TECHS]
    specs = []
    for j in keep:
        c = cl[j]
        row = fl[fl.id == c['ids'][0]].iloc[0]
        meta = json.loads(row['pd_meta']) if isinstance(row.get('pd_meta'), str) else {}
        specs.append(dict(id=c['ids'][0], tech=c['tech'], pmax=c['pmax'], pmin=c['pmin'], ru=c['ru'], rd=c['rd'],
                          ut=c['ut'], dt=c['dt'], start=c['start'], F0=c['F0'], Fmin=c['Fmin'],
                          widths=list(map(float, c['widths'])), ihr=list(map(float, c['ihr'])),
                          fuel_price=float(row['fuel_cost_$_per_MMBtu']), **meta))
    np.savez_compressed(f, times=epoch_s(w.index), price=r.price.astype(np.float32),
                        gas=(gas * gas_k).values.astype(np.float32), coal=np.float32(coal_p),
                        p=r.p[:, keep].astype(np.float32), u=r.u[:, keep].astype(np.float32),
                        su=r.su[:, keep].astype(np.float32), fuel=r.fuel_MMBtu[:, keep].astype(np.float32),
                        navail=r.navail[:, keep].astype(np.float32),
                        meta=json.dumps(dict(seed=seed, ba=ba, year=year, peak=peak, wall=wall, gas_k=gas_k,
                                             n_relaxed=int(r.n_days_relaxed), specs=specs)))
    print(f'job {seed}: {ba} {year} peak {peak:.0f} units {len(keep)} wall {wall:.0f}s relaxed {r.n_days_relaxed}',
          flush=True)


if __name__ == '__main__':
    a, n = int(sys.argv[1]), int(sys.argv[2])
    for s in range(a, a + n):
        try:
            job(s)
        except Exception as e:  # noqa
            print(f'job {s} failed: {e!r}', flush=True)
