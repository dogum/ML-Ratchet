"""Rung v2-r15: generated conventional light-duty cars without engine idle-stop.

RoadSim's generator gives 30% of conventional light-duty cars idle-stop (roadsim/generator.py, idle_stop = U < 0.3),
whatever their model year; nothing in the inputs says which. Idle-stop takes 7% off a hot UDDS and 0.1% off a HWFET
(12 RoadSim dev twins). RoadSim's conventional calibration on dev EPA cars ran twins without idle-stop, and those twins
already match the dev tests (FTP +1.4%, HWY +1.4% fuel, checks/epa_pool__roadsim-physics-v2.json); the network reads
the same twins 5.5% low on the UDDS and 0.8% low on the HWFET (twin_feed_epa.py). This rung turns idle-stop off for
the generated conventional cars, re-simulates every row of those cars (the trips and start states replayed from the
bank's own random draws, bank_replay.py) and recomputes their window-sticker labels (fleet.lab_labels).

python bank_idlestop.py <source bank> <fleet> <seed> <trips per vehicle> <out bank> [workers]
  e.g. train2d train2 101 12 train2di     (seeds: train2 101, train2b 102, train2c 103, val2 777)"""
import _paths as _P
import sys, os, json, glob, time, copy
sys.path.insert(0, _P.SRC)
from common import *
import numpy as np
import bank as BK, physics as PHY, dynoproc as DP, bank_replay as RP, features as F

_LAB = {}


def affected(x):
    s = x['spec']
    return s['architecture'] == 'conv' and s['class'] not in HD and bool((s.get('engine') or {}).get('idle_stop'))


def no_idle(spec):
    s = copy.deepcopy(spec); s['engine']['idle_stop'] = False
    return s


def one(args):
    src, fleet, seed, K, out, cp, patched = args
    ci = int(cp[-7:-4])
    op = f'{EN}/data/bank_{out}/chunk_{ci:03d}.npz'
    if os.path.exists(op): return (ci, 0, 0, 0)
    import fleet as FL
    B = dict(np.load(cp))
    V = json.load(open(f'{EN}/data/fleet_{fleet}/chunk_{ci:03d}.json'))
    R = RP.replay_chunk(fleet, K, seed, ci)
    n_rows = n_veh = n_unmatched = 0
    for vi, x in enumerate(V):
        if not affected(x): continue
        vid = ci * 100 + vi; spec = no_idle(x['spec']); n_veh += 1
        lab = FL.lab_labels(spec) or {}
        rows = np.where(B['vid'] == vid)[0]
        draws = R[vid]; j = 0
        for i in rows:
            while j < len(draws) and not (abs(draws[j]['dur'] - B['dur'][i]) < 0.5 and abs(draws[j]['dist'] - B['dist'][i]) < 1e-3 * max(B['dist'][i], 1.0)):
                j += 1
            if j >= len(draws):
                n_unmatched += 1; break
            d = draws[j]; j += 1
            name = d['name'] if d['dyno'] else None
            if patched and name in DP.CYCLES:
                o = DP.run(spec, d['tru'], name, d['soc'], d['temps'], d['seed'])
            else:
                hv = d['dyno'] and name in ('SC03', 'FTP_cold_20F')
                o = PHY.sim(spec, d['tru'], dyno=d['dyno'], hvac=hv, init_soc=d['soc'], init_temps=d['temps'], record=True, rng=d['seed'])
            tg = PHY.targets(o, spec, d['tru']['T_amb'], d['dyno'])
            L = o['ledger']
            B['comp'][i] = np.array([L['aero'], L['rolling'], L['brake'], L['aux'], L['eng_loss'] + L['mot_loss'] + L['trans_loss'] + L['batt_loss'],
                                     L['grade'], L['kinetic_change'], L['fuel'], L['batt_chem_out']], np.float32) / 1e3
            B['tg'][i] = np.array(tg + tg, np.float32)
            B['trace_miss'][i] = float(o['summary']['trace_rms_mps'])
            B['lab'][i] = np.array([lab.get('label_city_fuel', np.nan), lab.get('label_hwy_fuel', np.nan), np.nan, np.nan], np.float32)
            n_rows += 1
    os.makedirs(os.path.dirname(op), exist_ok=True)
    np.savez_compressed(op + '.tmp.npz', **B); os.replace(op + '.tmp.npz', op)
    return (ci, n_veh, n_rows, n_unmatched)


def main(src, fleet, seed, K, out, workers=1):
    patched = src.endswith('d')            # a bank_dyno.py bank: SC03 / cold-FTP rows under the EPA test conditions
    jobs = [(src, fleet, seed, K, out, cp, patched) for cp in sorted(glob.glob(f'{EN}/data/bank_{src}/chunk_*.npz'))]
    t0 = time.time(); tot = np.zeros(3, int)
    if workers <= 1:
        res = map(one, jobs)
    else:
        from multiprocessing import Pool
        p = Pool(workers); res = p.imap_unordered(one, jobs)
    for ci, nv, nr, nu in res:
        tot += [nv, nr, nu]
        print('chunk', ci, 'vehicles', nv, 'rows', nr, 'unmatched', nu, round(time.time() - t0), flush=True)
    print('DONE', out, 'vehicles', tot[0], 'rows', tot[1], 'unmatched', tot[2], flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6]) if len(sys.argv) > 6 else 1)
