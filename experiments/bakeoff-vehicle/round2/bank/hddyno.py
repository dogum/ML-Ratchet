"""P0 candidate (bank recipe, not physics): heavy-duty trucks on the chassis dynamometer ('hddyno' banks).

EnergyNet's bank (training/bank.py draw_route) never runs a heavy-duty vehicle on a dynamometer: trucks are 80 %
synthetic road drives and 20 % test cycles driven ON THE ROAD (weather, on-road extra loads, tyre-temperature effects).
The ICCT truck records are chassis-dynamometer tests (dyno flag 1), a flag no simulated truck ever carries.
Recipe change: half of every truck_md / truck_hd row whose route was a test cycle driven on the road (one of the
dynamometer cycles: HD_transient_1037A, HD_cruise_55/65mph_1037D, HD_cruise_55mph_flat, LA92_HD_class3, UDDS, NYCC) is
re-run on the dynamometer instead: the raw cycle (its own grade) at 22 C, no wind, dry surface, observed = true
weather, physics.sim(dyno=True) (dyno spec: no on-road load, no tyre-temperature effects, HVAC off), warm start, the
row's own start SOC and run seed, physics.targets(dyno=True). The choice per eligible row comes from a separate
generator, default_rng([20261124, chunk]), drawn in row order. Every other row, and every draw of the bank's own
generator, is unchanged: this script REPLAYS the bank's draws (as EnergyNet's bank_replay.py) and splices the new
rows into a copy of the source bank. Physics stays roadsim-physics-v2.

python hddyno.py <src bank> [--check N]    -> round2/work/banks/<src>_hd/ (+ MANIFEST.json, hddyno.json report)
  --check N: also re-simulate N unconverted on-road cycle rows exactly as the bank did and compare with the stored
             targets (must be equal: proves the replay is exact).
"""
import copy
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths as P  # noqa: E402
import numpy as np  # noqa: E402
import config as C  # noqa: E402
import common as CM  # noqa: E402
import bank as BK  # noqa: E402
import physics as PHY  # noqa: E402
import wrappers  # noqa: E402
import features as F  # noqa: E402
import routes  # noqa: E402
from roadsim.vehicle import fill_spec  # noqa: E402
import build_bank as BB  # noqa: E402

TRUCKS = ('truck_md', 'truck_hd')
DYNO_CYC = ('HD_transient_1037A', 'HD_cruise_55mph_1037D', 'HD_cruise_65mph_1037D', 'HD_cruise_55mph_flat',
            'LA92_HD_class3', 'UDDS', 'NYCC')
P_CONVERT = 0.5
T_DYNO = 22.0
CH = 50


class RecDict(dict):
    last = None

    def __getitem__(self, k):
        RecDict.last = k
        return dict.__getitem__(self, k)


_cyc = BK.cycles()
BK._CYC = RecDict(_cyc)


def replay_vehicle(spec, K, rng):
    full = fill_spec(spec); cls = spec['class']
    out = []
    for k in range(K):
        RecDict.last = None
        tr, src, dyno, name = BK.draw_route(rng, cls)
        cyc = RecDict.last if src == 'cyc_road' else None
        obs, tru = BK.observed_and_true(tr, rng, dyno)
        temps, cold, soc, disp = BK.init_state(rng, spec, full, tru, src, dyno, name)
        seed = int(rng.integers(1 << 30))
        out.append(dict(obs=obs, tru=tru, src=src, dyno=dyno, name=name, cyc=cyc, temps=temps, cold=cold, soc=soc,
                        disp=disp, seed=seed, dur=float(len(obs['v'])), dist=float(F.prep(obs)['v'].sum())))
    return out


def dyno_trip(name):
    tr = BK._fill(dict(_cyc[name])); N = len(tr['v'])
    tr = dict(tr)
    tr.update(T_amb=np.full(N, T_DYNO, np.float32), wind_speed=np.zeros(N, np.float32), surface=np.zeros(N, np.int8),
              rho_air=routes.rho_air(np.full(N, T_DYNO), np.zeros(N)).astype(np.float32))
    return tr


def sim_row(x, d, vid, dyno):
    """one bank row (build_bank.rows_for_vehicle's fields) from a replayed draw d; dyno=True runs d['obs'] on the dyno."""
    spec = x['spec']; lab = x['labels'] or {}; my = x['model_year']
    sim_spec = x.get('spec_sim', spec)
    if x['idle_off']:
        lab = dict(lab); lab.update(x['labels_sim'])
    full = BK.fill_spec(spec); rec = BK.veh_record(spec, full)
    cls, arch = spec['class'], spec['architecture']
    patched = wrappers.phev_patch(spec) if arch.startswith('phev') else None
    obs, tru = d['obs'], d['tru']
    temps, cold, disp, soc = d['temps'], d['cold'], d['disp'], d['soc']
    out = PHY.sim(sim_spec, tru, dyno=dyno, hvac=False, init_soc=soc, init_temps=temps, record=True, rng=d['seed'])
    tg = PHY.targets(out, sim_spec, tru['T_amb'], dyno)
    if patched is not None:
        o2 = PHY.sim(patched, tru, dyno=dyno, hvac=False, init_soc=soc, init_temps=temps, record=True, rng=d['seed'])
        tg = tg + PHY.targets(o2, patched, tru['T_amb'], dyno)
    else:
        tg = tg + tg
    Pp = F.prep(obs)
    L = out['ledger']
    comp = [L['aero'], L['rolling'], L['brake'], L['aux'], L['eng_loss'] + L['mot_loss'] + L['trans_loss'] + L['batt_loss'],
            L['grade'], L['kinetic_change'], L['fuel'], L['batt_chem_out']]
    tv, tx = BB.trace_of(obs)
    return dict(R=F.route_vec(Pp).astype(np.float32),
                PH=F.variants(Pp, cls, rec['m_true'], rec['cda'], rec['crr'], rec['cv'], rec['m_rep']).astype(np.float32),
                comp=np.array(comp, np.float32) / 1e3, tg=np.array(tg, np.float32), my=float(my) if my else np.nan,
                soc=np.nan if disp is None else disp, cold=float(cold), warm=float(not cold), dyno=float(dyno),
                src=BK.SRC.index('dyno' if dyno else d['src']), dur=float(len(obs['v'])), dist=float(Pp['v'].sum()),
                lab=np.array([lab.get('label_city_fuel', np.nan), lab.get('label_hwy_fuel', np.nan),
                              lab.get('label_city_elec', np.nan), lab.get('label_hwy_elec', np.nan)], np.float32),
                trace_miss=float(out['summary']['trace_rms_mps']), vid=vid, cycle=-1, T_true=float(np.mean(tru['T_amb'])),
                idle_off=float(x['idle_off']), _tv=tv, _tx=tx, **rec)


def main(src, ncheck=0):
    base = src.split('_')[0]
    cfg = P.BANK_RECIPES[base]
    sd = P.bank_dir(src); dd = P.bank_dir(src + '_hd')
    os.makedirs(dd, exist_ok=True)
    B = dict(np.load(os.path.join(sd, 'rows.npz')))
    tv = np.load(os.path.join(sd, 'trace_v.npy'), mmap_mode='r'); tx = np.load(os.path.join(sd, 'trace_x.npy'), mmap_mode='r')
    fleet = json.load(open(os.path.join(sd, 'fleet.json')))
    n = len(B['vid']); off = B['trace_off']
    new = {}                                   # row index -> new row dict
    rep = {'src': src, 'converted': 0, 'eligible': 0, 'checked': [], 'by_cycle': {}, 'rows': []}
    nch = (cfg['n'] + CH - 1) // CH
    t0 = time.time()
    for ci in range(nch):
        rng = np.random.default_rng([cfg['bseed'], ci]); rng2 = np.random.default_rng([20261124, ci])
        V = fleet[ci * CH:(ci + 1) * CH]
        for vi, x in enumerate(V):
            vid = ci * 100 + vi
            draws = replay_vehicle(x['spec'], cfg['K'], rng)
            if x['spec']['class'] not in TRUCKS:
                continue
            rows = np.where(B['vid'] == vid)[0]
            j = 0
            for d in draws:                    # align draws with stored rows (a failed run left no row)
                if j >= len(rows):
                    break
                r = rows[j]
                if abs(B['dur'][r] - d['dur']) > 0.5 or abs(B['dist'][r] - d['dist']) > 1e-3 * max(d['dist'], 1.0):
                    continue
                j += 1
                if d['src'] != 'cyc_road' or d['cyc'] not in DYNO_CYC:
                    continue
                rep['eligible'] += 1
                conv = rng2.random() < P_CONVERT
                if not conv:
                    if len(rep['checked']) < ncheck:
                        row = sim_row(x, d, vid, False)
                        rep['checked'].append({'row': int(r), 'max_rel_diff_tg': float(np.max(np.abs(row['tg'] / B['tg'][r] - 1)))})
                    continue
                dd_ = dict(d)
                trip = dyno_trip(d['cyc'])
                dd_.update(obs=trip, tru={k: np.asarray(v).copy() for k, v in trip.items()}, cold=False,
                           temps=CM.warm_temps(T_DYNO))
                try:
                    new[int(r)] = sim_row(x, dd_, vid, True)
                except Exception as e:  # noqa: BLE001
                    print('dyno run failed, road row kept', vid, e, flush=True); continue
                rep['converted'] += 1
                rep['by_cycle'][d['cyc']] = rep['by_cycle'].get(d['cyc'], 0) + 1
                rep['rows'].append({'row': int(r), 'cyc': d['cyc'], 'cls': x['spec']['class'], 'arch': x['spec']['architecture'],
                                    'E_road': float(B['tg'][r, 0]), 'E_dyno': float(new[int(r)]['tg'][0])})
        print(f'{src} chunk {ci}: converted {rep["converted"]} of {rep["eligible"]} eligible, {time.time() - t0:.0f}s', flush=True)
    # splice
    out = {k: v.copy() for k, v in B.items() if k not in ('trace_off', 'trace_len')}
    for r, row in new.items():
        for k, v in row.items():
            if not k.startswith('_') and k in out:
                out[k][r] = v
    lens = np.diff(off).astype(np.int64)
    for r, row in new.items():
        lens[r] = len(row['_tv'])
    out['trace_len'] = lens.astype(np.int32); out['trace_off'] = np.concatenate([[0], np.cumsum(lens)])
    TV = np.empty(int(lens.sum()), np.float32); TX = np.empty((int(lens.sum()), 7), np.float16)
    o2 = out['trace_off']
    for r in range(n):
        if r in new:
            TV[o2[r]:o2[r + 1]] = new[r]['_tv']; TX[o2[r]:o2[r + 1]] = new[r]['_tx']
        else:
            TV[o2[r]:o2[r + 1]] = tv[off[r]:off[r + 1]]; TX[o2[r]:o2[r + 1]] = tx[off[r]:off[r + 1]]
    np.savez(os.path.join(dd, 'rows.npz'), **out)
    np.save(os.path.join(dd, 'trace_v.npy'), TV); np.save(os.path.join(dd, 'trace_x.npy'), TX)
    json.dump(fleet, open(os.path.join(dd, 'fleet.json'), 'w'), default=float)
    man = {'split': src + '_hd', 'recipe': dict(cfg, change='hddyno: half of the on-road test-cycle truck rows re-run on '
                                                  'the dynamometer (round2/bank/hddyno.py)'),
           'physics': C.PHYSICS, 'n_rows': int(n), 'n_vehicles': len(fleet), 'converted_rows': rep['converted'], 'files': {}}
    for f in ('rows.npz', 'trace_v.npy', 'trace_x.npy', 'fleet.json'):
        p = os.path.join(dd, f)
        man['files'][f] = dict(bytes=os.path.getsize(p), sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest())
    json.dump(man, open(os.path.join(dd, 'MANIFEST.json'), 'w'), indent=1)
    e = np.array([x['E_dyno'] / x['E_road'] - 1 for x in rep['rows']]) if rep['rows'] else np.array([np.nan])
    rep['median_dyno_over_road'] = float(np.median(e))
    json.dump(rep, open(os.path.join(dd, 'hddyno.json'), 'w'), indent=1, default=float)
    print('done', src + '_hd', rep['converted'], 'converted; median E_dyno / E_road - 1', round(rep['median_dyno_over_road'], 4),
          'checks', rep['checked'][:5], flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], int(a[a.index('--check') + 1]) if '--check' in a else 0)
