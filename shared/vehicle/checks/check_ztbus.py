"""Check: a RoadSim twin of the ZTBus trolleybuses on every ZTBus full-set leg (VBZ Zurich buses 183 and 208).
Writes ztbus_twin.json (v1, --split all) or ztbus_twin__<preset>[__<split>].json.

Twin (one spec for every leg, nothing fitted): an 18.7 m articulated battery-trolleybus run as a RoadSim BEV bus:
curb mass 19,000 kg plus 70 kg per passenger (the leg's mean passenger count), CdA 6.0 m2, Crr 0.007, one 320 kW
traction machine, a 200 kWh pack (the trolley line supplies most of the energy; a large pack keeps battery limits out
of the comparison), class-default bus accessories and HVAC (automatic, on). Trip: the leg's own 1 Hz speed, GNSS grade,
ambient temperature, no wind. Start: cabin 21 C, pack at ambient.
Compared per leg (q_label_ok legs only): net electrical energy at the DC bus, simulated vs ZTBus elec_kWh (traction +
auxiliaries, regeneration negative, whether from the line or the pack). Error = sim / real - 1 (%).
The by-month split (data/proc/hd/ztbus_full_splits.json) is in force: 294 dev missions, 1,115 exam. The calibration
pass may look only at dev; exam is computed once for the final report.

usage: python check_ztbus.py [--options roadsim-physics-v2] [--split dev|exam|all] [--workers 14] [--limit N]
"""
import os, sys, json, time, argparse
from multiprocessing import Pool
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VEH = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(VEH, 'roadsim')); sys.path.insert(0, VEH)
import _opts

CURB_KG = 19000.0
PAX_KG = 70.0


def bus_spec(passengers):
    return {'id': 'ztbus_twin', 'class': 'bus', 'architecture': 'bev', 'mass_kg': CURB_KG,
            'payload_kg': PAX_KG * float(max(passengers, 0.0)), 'CdA_m2': 6.0, 'Crr': 0.007, 'Crr_speed_coef': 0.002,
            'wheel_radius_m': 0.5, 'motors': [{'role': 'traction', 'max_power_kW': 320, 'eff_map': 'ipm_generic_150', 'max_rpm': 6000}],
            'battery': {'chemistry': 'LFP', 'capacity_kWh': 200, 'nominal_V': 650, 'soc_window': [0.05, 0.98]},
            'aux': {'base_W': 4000, 'HVAC': {'enabled': True}}}


def run_leg(job):
    import realdata as rd
    import roadsim
    row, opt = job['row'], job['opt']
    try:
        meta, a = rd.load_trip(row['trip_id'])
        trip = {k: np.nan_to_num(a[k].astype(float)) for k in ('t', 'v', 'grade', 'elev', 'heading', 'T_amb', 'rho_air')}
        trip['T_amb'] = np.where(np.isfinite(a['T_amb']), a['T_amb'], np.nanmean(a['T_amb'])).astype(float)
        trip['rho_air'] = np.where(np.isfinite(a['rho_air']) & (a['rho_air'] > 0.5), a['rho_air'], 1.2).astype(float)
        trip['wind_speed'] = np.zeros(len(trip['v'])); trip['wind_dir'] = np.zeros(len(trip['v']))
        trip['surface'] = np.zeros(len(trip['v']), int)
        T0 = float(np.nanmean(a['T_amb']))
        o = roadsim.simulate(bus_spec(row['passengers_mean']), trip, init_soc=0.6, options=opt, check=False, record=True,
                             init_temps=dict(cabin=21.0, battery=T0))
    except Exception as e:
        return dict(trip_id=row['trip_id'], error=f'{type(e).__name__}: {e}'[:200])
    E = float(np.sum(np.asarray(o['batt_current'], float) * np.asarray(o['batt_voltage'], float)) / 3.6e6)
    real = float(row['elec_kWh'])
    still = np.asarray(o['v']) < 0.05
    return dict(trip_id=row['trip_id'], split=row['split'], bus=row['vehicle_id'], T=T0, season=row['season'],
                km=float(row['distance_m']) / 1000, passengers=float(row['passengers_mean']), real_kWh=real, sim_kWh=E,
                err_pct=100 * (E / real - 1) if abs(real) > 1e-6 else None,
                sim_hvac_kWh=float(np.sum(o['hvac_power']) / 3.6e6), closure=float(o['ledger']['closure_rel']),
                trace_rms=float(o['summary']['trace_rms_mps']), still_frac=float(still.mean()))


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return None
    return dict(n=int(len(x)), median=round(float(np.median(x)), 2), p25=round(float(np.percentile(x, 25)), 2),
                p75=round(float(np.percentile(x, 75)), 2), mean_abs=round(float(np.mean(np.abs(x))), 2))


def summarize(rows):
    d = pd.DataFrame(rows)
    out = {}
    for split, g in d.groupby('split'):
        o = {'all': stats(g.err_pct), 'kWh_per_km': dict(sim=round(float(g.sim_kWh.sum() / g.km.sum()), 3),
                                                     real=round(float(g.real_kWh.sum() / g.km.sum()), 3))}
        o['by_T'] = {b: stats(g.loc[(g['T'] >= lo) & (g['T'] < hi), 'err_pct']) for b, lo, hi in
                     (('<0C', -40, 0), ('0-10C', 0, 10), ('10-20C', 10, 20), ('20-25C', 20, 25), ('>=25C', 25, 60))}
        o['by_season'] = {s: stats(gg.err_pct) for s, gg in g.groupby('season')}
        o['by_bus'] = {b: stats(gg.err_pct) for b, gg in g.groupby('bus')}
        out[split] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='dev', choices=['dev', 'exam', 'all'])
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--limit', type=int, default=None)
    _opts.init()
    a = ap.parse_args()
    import realdata as rd
    import roadsim
    t0 = time.time()
    legs = rd.ztbus_full(None if a.split == 'all' else a.split, label_ok=True)
    rows = legs.to_dict('records')
    if a.limit:
        rows = rows[:a.limit]
    with Pool(a.workers) as p:
        res = p.map(run_leg, [dict(row=r, opt=_opts.get()) for r in rows], chunksize=2)
    errors = [r for r in res if 'error' in r]
    res = [r for r in res if 'error' not in r]
    summ = summarize(res)
    out = dict(check='RoadSim twin of the ZTBus trolleybuses on their own legs', method=__doc__.strip(),
               options=_opts.get(), options_name=_opts.name(), physics_default=roadsim.PHYSICS_VERSION, split_run=a.split,
               n_legs=len(res), n_errors=len(errors), errors=errors[:20], summary=summ,
               ledger_worst_closure=max((r['closure'] for r in res), default=None), legs=res, wall_s=round(time.time() - t0, 1),
               note='exam is computed for the final report only; the calibration pass looks only at dev',
               attribution='ZTBus: F. Widmer, A. Ritter, C. H. Onder, ETH Zurich, doi:10.3929/ethz-b-000626723 (CC BY 4.0)')
    fn = _opts.out('ztbus_twin.json')
    if a.split != 'all':
        fn = fn.replace('.json', f'__{a.split}.json')
    json.dump(out, open(os.path.join(HERE, fn), 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    for split, o in summ.items():
        s = o['all'] or {}
        print(f"{split:5s} legs {s.get('n')} median {s.get('median')} IQR [{s.get('p25')}, {s.get('p75')}] kWh/km sim/real "
              f"{o['kWh_per_km']['sim']}/{o['kWh_per_km']['real']}  by T " + ' '.join(f"{k} {(v or {}).get('median')}" for k, v in o['by_T'].items()))
    print(f'{len(res)} legs, {len(errors)} errors, {out["wall_s"]} s -> {fn}')


if __name__ == '__main__':
    main()
