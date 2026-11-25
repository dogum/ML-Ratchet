"""P0 candidate (bank recipe, physics unchanged): plug-ins with charge start with a cold engine ('pc' banks).

EnergyNet's bank (training/bank.py init_state) starts every road trip that begins moving, and 55-60 % of the others,
with a warm engine (85 C coolant, 90 C oil): a conventional-car rule. A plug-in with usable charge drives on its battery,
so its engine is cold; the warm coolant heats the cabin for free (RoadSim CALIBRATION.md H25). Recipe change: every
plug-in ROAD row (route source VED, synthetic or cycle on the road; dynamometer rows unchanged) that the bank started
warm and whose start charge fraction disp > 0.05 is re-run with engine and oil at the trip's starting ambient
temperature (driveline, axle, tyres, cabin and battery as drawn). Same vehicle, trip, start SOC and run seed; both the
raw and the plug-in-patch targets re-simulated. Every other row and every draw of the bank's generator unchanged
(replayed, as EnergyNet's bank_replay.py; spliced into a copy of the source bank). Physics stays roadsim-physics-v2.

python phevcold.py <src bank> [--check N]   -> round2/work/banks/<src>_pc/ (+ MANIFEST.json, phevcold.json)
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hddyno as HD  # noqa: E402   (replay + row simulation; imports the round-2 path layer)
import numpy as np  # noqa: E402
P, C = HD.P, HD.C

DISP_MIN = 0.05


def main(src, ncheck=0):
    base = src.split('_')[0]
    cfg = P.BANK_RECIPES[base]
    sd = P.bank_dir(src); dd = P.bank_dir(src + '_pc')
    os.makedirs(dd, exist_ok=True)
    B = dict(np.load(os.path.join(sd, 'rows.npz')))
    tv = np.load(os.path.join(sd, 'trace_v.npy'), mmap_mode='r'); tx = np.load(os.path.join(sd, 'trace_x.npy'), mmap_mode='r')
    fleet = json.load(open(os.path.join(sd, 'fleet.json')))
    n = len(B['vid']); off = B['trace_off']
    new = {}
    rep = {'src': src, 'changed': 0, 'plugin_rows': 0, 'checked': [], 'rows': []}
    nch = (cfg['n'] + HD.CH - 1) // HD.CH
    t0 = time.time()
    for ci in range(nch):
        rng = np.random.default_rng([cfg['bseed'], ci])
        V = fleet[ci * HD.CH:(ci + 1) * HD.CH]
        for vi, x in enumerate(V):
            vid = ci * 100 + vi
            draws = HD.replay_vehicle(x['spec'], cfg['K'], rng)
            if not x['spec']['architecture'].startswith('phev'):
                continue
            rows = np.where(B['vid'] == vid)[0]
            j = 0
            for d in draws:
                if j >= len(rows):
                    break
                r = rows[j]
                if abs(B['dur'][r] - d['dur']) > 0.5 or abs(B['dist'][r] - d['dist']) > 1e-3 * max(d['dist'], 1.0):
                    continue
                j += 1
                rep['plugin_rows'] += 1
                if d['dyno'] or d['cold'] or d['disp'] is None or d['disp'] <= DISP_MIN:
                    if not d['dyno'] and len(rep['checked']) < ncheck:
                        row = HD.sim_row(x, d, vid, False)
                        rep['checked'].append({'row': int(r), 'max_rel_diff_tg': float(np.max(np.abs(row['tg'] / B['tg'][r] - 1)))})
                    continue
                d2 = dict(d)
                T0 = float(np.asarray(d['tru']['T_amb'])[0])
                temps = dict(d['temps']); temps.update(engine=T0, oil=T0)
                d2['temps'] = temps
                try:
                    new[int(r)] = HD.sim_row(x, d2, vid, False)
                except Exception as e:  # noqa: BLE001
                    print('re-run failed, row kept', vid, e, flush=True); continue
                rep['changed'] += 1
                rep['rows'].append({'row': int(r), 'arch': x['spec']['architecture'], 'T0': T0, 'disp': d['disp'],
                                    'E_warm': float(B['tg'][r, 4]), 'E_cold': float(new[int(r)]['tg'][4])})
        print(f'{src} chunk {ci}: changed {rep["changed"]} of {rep["plugin_rows"]} plug-in rows, {time.time() - t0:.0f}s', flush=True)
    out = {k: v.copy() for k, v in B.items()}
    for r, row in new.items():
        for k, v in row.items():
            if not k.startswith('_') and k not in ('src', 'dyno', 'cold', 'warm'):
                out[k][r] = v
        assert len(row['_tv']) == B['trace_len'][r]            # same trip: the trace is unchanged
    np.savez(os.path.join(dd, 'rows.npz'), **out)
    np.save(os.path.join(dd, 'trace_v.npy'), np.asarray(tv)); np.save(os.path.join(dd, 'trace_x.npy'), np.asarray(tx))
    json.dump(fleet, open(os.path.join(dd, 'fleet.json'), 'w'), default=float)
    man = {'split': src + '_pc', 'recipe': dict(cfg, change='phevcold: plug-in road rows with charge start with the engine '
                                                  'and oil at ambient (round2/bank/phevcold.py)'),
           'physics': C.PHYSICS, 'n_rows': int(n), 'n_vehicles': len(fleet), 'changed_rows': rep['changed'], 'files': {}}
    for f in ('rows.npz', 'trace_v.npy', 'trace_x.npy', 'fleet.json'):
        p = os.path.join(dd, f)
        man['files'][f] = dict(bytes=os.path.getsize(p), sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest())
    json.dump(man, open(os.path.join(dd, 'MANIFEST.json'), 'w'), indent=1)
    e = np.array([x['E_cold'] / x['E_warm'] - 1 for x in rep['rows']]) if rep['rows'] else np.array([np.nan])
    t = np.array([x['T0'] for x in rep['rows']]) if rep['rows'] else np.array([np.nan])
    rep['median_cold_over_warm'] = float(np.median(e))
    rep['median_cold_over_warm_below_10C'] = float(np.median(e[t < 10])) if (t < 10).any() else None
    json.dump(rep, open(os.path.join(dd, 'phevcold.json'), 'w'), indent=1, default=float)
    print('done', src + '_pc', rep['changed'], 'changed; median E_cold / E_warm - 1', round(rep['median_cold_over_warm'], 4),
          'below 10 C', rep['median_cold_over_warm_below_10C'], 'checks', rep['checked'][:5], flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], int(a[a.index('--check') + 1]) if '--check' in a else 0)
