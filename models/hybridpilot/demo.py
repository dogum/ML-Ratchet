"""Demo: three real VED exam trips (Ann Arbor speed, lidar grade, airport weather), each driven by a RoadSim twin of
the car that drove it, under HybridPilot, RoadSim's A-ECMS and the rule-based strategy. Runs in a few seconds.
The logged fuel is printed for reference only (see MODEL_CARD: the simulator reads ~30 % lower than VED's
MAF-based fuel on real trips)."""
import json, os, time
import numpy as np
from pipeline import compare, load_example, HERE

t0 = time.time()
for ex in json.load(open(os.path.join(HERE, 'examples', 'index.json'))):
    spec, trip = load_example(ex['trip'])
    phev = ex['arch'].startswith('phev')
    soc0 = float(np.nanmax([trip['soc'][np.isfinite(trip['soc'])][0]])) if phev and np.isfinite(trip['soc']).any() else None
    soc0 = soc0 / 100 if soc0 and soc0 > 1.5 else soc0
    print(f"\n{ex['trip']}  {ex['arch']}  ({spec['meta'].get('likely_models')})  {ex['distance_km']:.1f} km, {ex['T_amb']:.0f} C"
          + (f", start SOC {soc0:.2f} (logged)" if soc0 else ''))
    print(f"  logged (VED, MAF estimate): fuel {np.nansum(trip['fuel_rate']):.0f} g")
    for r in compare(spec, trip, init_soc=soc0, warm=True):
        print(f"  {r['strategy']:16s} fuel {r['fuel_g']:7.1f} g   battery out {r['batt_kWh_out']:+.3f} kWh   "
              f"SOC {r['soc_start']:.3f} -> {r['soc_end']:.3f}   engine starts {r['engine_starts']:3d}")
print(f"\n{time.time() - t0:.1f} s")
