"""TripDetective demo: read a few real VED exam cars (never used for choosing) from their trip logs alone.
usage: python3 demo.py        (numpy + scipy only; runs in a few seconds)"""
import glob, json, os, time
import numpy as np
import tripdetective as TD

HERE = os.path.dirname(os.path.abspath(__file__))


def load_example(path):
    z = np.load(path)
    info = json.loads(str(z['info']))
    trips = []
    for k in range(len(info['trips'])):
        trips.append({key[len(f'trip{k}_'):]: z[key] for key in z.files if key.startswith(f'trip{k}_')})
    return info, trips


if __name__ == '__main__':
    td = TD.TripDetective()
    for p in sorted(glob.glob(os.path.join(HERE, 'examples', 'VED_*.npz'))):
        info, trips = load_example(p)
        est = info['trips'][0]['fuel_source'] != 'obd_fuel_rate_Lph'
        t0 = time.time()
        one = td.read_trip(trips[0], fuel_estimated=est)
        car = td.read_car(trips, fuel_estimated=est)
        dt = time.time() - t0
        pub = info['published']
        print(f"\n{info['vehicle_id']}  published: {pub['ved_type']}, weight class {pub['weight_class_lb']} lb "
              f"(curb ~{(pub['weight_class_lb'] or 0) * 0.4536:.0f} kg, +-113 kg), engine {pub['displacement_L']} L  [{dt:.2f} s]")
        for name, r in [('1 trip ', one), (f"{len(trips)} trips", car)]:
            lo, hi = r['mass_kg_1sigma']
            disp = '' if r['displacement_L'] is None else f", engine {r['displacement_L']:.1f} L ({r['displacement_L_1sigma'][0]:.1f}-{r['displacement_L_1sigma'][1]:.1f})"
            print(f"  {name}: {max(r['type_probs'], key=r['type_probs'].get)} ({r['architecture']}, p={r['arch_probs'][r['architecture']]:.2f}), "
                  f"curb {r['curb_kg']:.0f} kg (1 sigma {lo - 136:.0f}-{hi - 136:.0f}){disp}, CdA {r['CdA_m2']:.2f} m2, Crr {r['Crr']:.4f}")
        led = car['ledger_kJ_per_km'][0]
        print('  ledger of trip 1 (kJ/km, simulated-truth-trained, not real-graded):', ', '.join(f"{k} {v[0]:.0f}" for k, v in led.items()))

    # a city bus (v2.1): every leg of one ZTBus exam bus-day, read together (the reading follows passengers from leg to
    # leg within a day, not from day to day); truth convention 19,000 kg + 70 kg per passenger
    z = np.load(os.path.join(HERE, 'examples', 'ZTBus_exam_mission.npz'))
    info = json.loads(str(z['info']))
    legs = [{key[len(f'leg{k}_'):]: z[key] for key in z.files if key.startswith(f'leg{k}_')} for k in range(len(info['legs']))]
    t0 = time.time()
    R = TD.TripDetectiveBus().read_mission(legs)
    print(f"\n{info['bus']}, one day ({info['season']}): {len(legs)} legs read together  [{time.time() - t0:.2f} s]")
    for m, r in zip(info['legs'], R):
        pax = m['passengers_mean']
        truth = '   (no passenger count)' if pax is None else f"   convention {19000 + 70 * pax:6.0f} kg ({pax:4.1f} passengers)"
        print(f"  {m['km']:6.2f} km {m['minutes']:5.1f} min: bus mass {r['mass_kg']:6.0f} kg (1 sigma +-{r['mass_kg'] - r['mass_kg_1sigma'][0]:.0f}),"
              f" network alone {r['network_kg']:6.0f} kg{truth}")
