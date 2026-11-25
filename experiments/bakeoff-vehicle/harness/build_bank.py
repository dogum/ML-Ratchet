"""The frozen simulated bank: EnergyNet v2.1's recipe on RoadSim physics v2 (pinned), with compact traces.

python3 build_bank.py <val|train> <worker> <n_workers>     -> bank/<split>/chunk_XXX.npz   (resumable per chunk)
python3 build_bank.py finalize <val|train>                  -> bank/<split>/{rows.npz, trace_*.npy, fleet.json, MANIFEST}

EnergyNet v2.1 (rung r17) was trained on banks that took three passes (training/bank.py -> bank_dyno.py (r14) ->
bank_idlestop.py (r15)). This builds the end state in one pass with the same random draws:
  fleet     training/fleet.py: roadsim.generator donors 'v2', model year + era prior, road load / transmission /
            BEV motor / articulated-bus widening, simulated window-sticker labels; vehicle ids '<val2|train2>_CCC_JJ'
            (the id length seeds the HD reported-mass draw in bank.veh_record). Fleets: val2 400 vehicles seed 202,
            train2 2400 vehicles seed 201.
  r15       generated conventional light-duty cars with idle-stop get idle-stop off and their labels recomputed
            (bank_idlestop.py: no_idle + fleet.lab_labels).
  trips     training/bank.py vehicle_examples(): routes (VED train_noise 45 %, synthetic 35 %, dyno cycles 13 %,
            cycles on the road 7 % for light duty), observed vs true weather, start states, run seeds: the same
            generator per chunk (default_rng([bank seed, chunk])). Banks: val2 10 trips/vehicle seed 777, train2 12
            trips/vehicle seed 101.
  r14       SC03 and 20 F cold-FTP dyno draws re-run under the EPA test conditions (dynoproc.run) with bank_dyno.py's
            seeds (default_rng([20261115, chunk]), one per such row in row order), except rows of r15's cars, which
            bank_idlestop.py re-ran with the row's own seed.
Stored per trip, on top of EnergyNet's bank fields (R, PH, tg, comp, lab, vehicle fields, tiers' raw material): the
observed trip at 1 Hz (what the network sees): v float32; grade, elev - elev[0], T_amb, wind along / across the
heading, rho_air, surface float16 (features.prep's values), capped at config.TRACE_MAX_S (no trip is longer).
"""
import os
import sys
import json
import glob
import time
import copy
import hashlib
import numpy as np
import config as C
import common as CM
import fleet as FL
import bank as BK
import physics as PHY
import dynoproc as DP
import wrappers
import features as F

SPLITS = {'val': dict(fleet='val2', n=400, fseed=202, K=10, bseed=777),
          'train': dict(fleet='train2', n=2400, fseed=201, K=12, bseed=101)}
CH = 50


def affected(spec):
    return spec['architecture'] == 'conv' and spec['class'] not in CM.HD and bool((spec.get('engine') or {}).get('idle_stop'))


def no_idle(spec):
    s = copy.deepcopy(spec); s['engine']['idle_stop'] = False
    return s


def fleet_chunk(cfg, ci):
    """fleet.one_chunk, in memory -> list of {'spec', 'labels', 'model_year', 'spec_sim', 'labels_sim', 'idle_off'}"""
    rng = np.random.default_rng([cfg['fseed'], ci])
    res = []
    for j in range(min(CH, cfg['n'] - ci * CH)):
        for attempt in range(5):
            try:
                s, my = FL.make_vehicle(rng); lab = FL.lab_labels(s)
                break
            except Exception as e:
                print('retry', e, flush=True)
        s['id'] = f"{cfg['fleet']}_{ci:03d}_{j:02d}"
        x = {'spec': s, 'labels': lab, 'model_year': my, 'idle_off': affected(s)}
        if x['idle_off']:
            x['spec_sim'] = no_idle(s)
            l2 = FL.lab_labels(x['spec_sim']) or {}
            x['labels_sim'] = {'label_city_fuel': l2.get('label_city_fuel', np.nan), 'label_hwy_fuel': l2.get('label_hwy_fuel', np.nan)}
        res.append(x)
    return res


def trace_of(obs, cap=C.TRACE_MAX_S):
    """observed trip -> (v float32 (N,), X float16 (N, 7)): grade, elev rel., T_amb, wind head / cross, rho_air, surface.
    cap: stored length cap for the bank (no simulated trip is longer); real traces use cap=None (full length)."""
    P = F.prep(obs)
    N = P['N']
    e = obs.get('elev')
    e = np.zeros(N) if e is None else np.asarray(e, float)
    if np.isnan(e).all(): e = np.zeros(N)
    elif np.isnan(e).any():
        i = np.arange(N); ok = ~np.isnan(e); e = np.interp(i, i[ok], e[ok])
    X = np.stack([P['grade'], e - e[0], P['T'], P['wh'], P['wc'], P['rho'], P['surf'].astype(float)], 1)
    return P['v'].astype(np.float32)[:cap], X.astype(np.float16)[:cap]


def rows_for_vehicle(x, K, rng, rng_dyno, vid):
    spec = x['spec']; lab = x['labels'] or {}; my = x['model_year']
    sim_spec = x.get('spec_sim', spec)
    if x['idle_off']:
        lab = dict(lab); lab.update(x['labels_sim'])
    full = BK.fill_spec(spec); rec = BK.veh_record(spec, full)
    cls, arch = spec['class'], spec['architecture']
    patched = wrappers.phev_patch(spec) if arch.startswith('phev') else None
    rows = []
    for k in range(K):
        try:
            tr, src, dyno, name = BK.draw_route(rng, cls)
            obs, tru = BK.observed_and_true(tr, rng, dyno)
            temps, cold, soc, disp = BK.init_state(rng, spec, full, tru, src, dyno, name)
            hv = dyno and name in ('SC03', 'FTP_cold_20F')
            seed = int(rng.integers(1 << 30))
            if x['idle_off'] or hv:
                # EnergyNet's bank row exists only if the original run succeeded
                PHY.sim(spec, tru, dyno=dyno, hvac=hv, init_soc=soc, init_temps=temps, record=False, rng=seed)
            if hv:
                dseed = int(rng_dyno.integers(1 << 30))            # bank_dyno.py draws one per SC03 / cold-FTP row
                use = seed if x['idle_off'] else dseed             # bank_idlestop.py re-ran its cars with the row seed
                try:
                    out = DP.run(sim_spec, tru, name, soc, temps, use)
                    tg = PHY.targets(out, sim_spec, tru['T_amb'], True)
                    if patched is not None:
                        o2 = DP.run(patched, tru, name, soc, temps, use)
                        tg = tg + PHY.targets(o2, patched, tru['T_amb'], True)
                    else:
                        tg = tg + tg
                except Exception as ex:                              # bank_dyno.py kept the old row
                    print('dyno fail, old row kept', spec['id'], ex, flush=True)
                    out = PHY.sim(sim_spec, tru, dyno=dyno, hvac=hv, init_soc=soc, init_temps=temps, record=True, rng=seed)
                    tg = PHY.targets(out, sim_spec, tru['T_amb'], dyno)
                    if patched is not None:
                        o2 = PHY.sim(patched, tru, dyno=dyno, hvac=hv, init_soc=soc, init_temps=temps, record=True, rng=seed)
                        tg = tg + PHY.targets(o2, patched, tru['T_amb'], dyno)
                    else:
                        tg = tg + tg
            else:
                out = PHY.sim(sim_spec, tru, dyno=dyno, hvac=hv, init_soc=soc, init_temps=temps, record=True, rng=seed)
                tg = PHY.targets(out, sim_spec, tru['T_amb'], dyno)
                if patched is not None:
                    o2 = PHY.sim(patched, tru, dyno=dyno, hvac=hv, init_soc=soc, init_temps=temps, record=True, rng=seed)
                    tg = tg + PHY.targets(o2, patched, tru['T_amb'], dyno)
                else:
                    tg = tg + tg
        except Exception as ex:
            print('fail', spec['id'], ex, flush=True); continue
        P = F.prep(obs)
        Rv = F.route_vec(P)
        PH = F.variants(P, cls, rec['m_true'], rec['cda'], rec['crr'], rec['cv'], rec['m_rep'])
        L = out['ledger']
        comp = [L['aero'], L['rolling'], L['brake'], L['aux'], L['eng_loss'] + L['mot_loss'] + L['trans_loss'] + L['batt_loss'],
                L['grade'], L['kinetic_change'], L['fuel'], L['batt_chem_out']]
        tv, tx = trace_of(obs)
        rows.append(dict(R=Rv.astype(np.float32), PH=PH.astype(np.float32), comp=np.array(comp, np.float32) / 1e3,
                         tg=np.array(tg, np.float32), my=float(my) if my else np.nan,
                         soc=np.nan if disp is None else disp, cold=float(cold), warm=float(not cold), dyno=float(dyno),
                         src=BK.SRC.index(src), dur=float(len(obs['v'])), dist=float(P['v'].sum()),
                         lab=np.array([lab.get('label_city_fuel', np.nan), lab.get('label_hwy_fuel', np.nan),
                                       lab.get('label_city_elec', np.nan), lab.get('label_hwy_elec', np.nan)], np.float32),
                         trace_miss=float(out['summary']['trace_rms_mps']), vid=vid,
                         cycle=BK.LD_CYC.index(name) if (dyno and name in BK.LD_CYC) else -1,
                         T_true=float(np.mean(tru['T_amb'])), idle_off=float(x['idle_off']),
                         _tv=tv, _tx=tx, **rec))
    return rows


def one_chunk(split, ci):
    cfg = SPLITS[split]
    od = os.path.join(C.BANK, split); os.makedirs(od, exist_ok=True)
    op = os.path.join(od, f'chunk_{ci:03d}.npz')
    if os.path.exists(op):
        return
    t0 = time.time()
    V = fleet_chunk(cfg, ci); t1 = time.time()
    rng = np.random.default_rng([cfg['bseed'], ci]); rng_dyno = np.random.default_rng([20261115, ci])
    rows = []
    for vi, x in enumerate(V):
        rows += rows_for_vehicle(x, cfg['K'], rng, rng_dyno, ci * 100 + vi)
    D = {k: np.stack([r[k] for r in rows]) for k in rows[0] if not k.startswith('_')}
    D['trace_len'] = np.array([len(r['_tv']) for r in rows], np.int32)
    D['trace_v'] = np.concatenate([r['_tv'] for r in rows]); D['trace_x'] = np.concatenate([r['_tx'] for r in rows])
    np.savez(op + '.tmp.npz', **D)
    json.dump(V, open(os.path.join(od, f'fleet_{ci:03d}.json.tmp'), 'w'), default=float)
    os.replace(os.path.join(od, f'fleet_{ci:03d}.json.tmp'), os.path.join(od, f'fleet_{ci:03d}.json'))
    os.replace(op + '.tmp.npz', op)
    print(f'{split} chunk {ci} rows {len(rows)} fleet {t1 - t0:.0f}s trips {time.time() - t1:.0f}s', flush=True)


def finalize(split):
    od = os.path.join(C.BANK, split)
    cfg = SPLITS[split]; nch = (cfg['n'] + CH - 1) // CH
    fs = [os.path.join(od, f'chunk_{ci:03d}.npz') for ci in range(nch)]
    miss = [f for f in fs if not os.path.exists(f)]
    if miss:
        raise SystemExit(f'{len(miss)} chunks missing')
    parts = [dict(np.load(f)) for f in fs]
    keys = [k for k in parts[0] if not k.startswith('trace_')]
    rows = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    tl = np.concatenate([p['trace_len'] for p in parts]).astype(np.int64)
    rows['trace_len'] = tl.astype(np.int32)
    rows['trace_off'] = np.concatenate([[0], np.cumsum(tl)])
    np.savez(os.path.join(od, 'rows.npz'), **rows)
    np.save(os.path.join(od, 'trace_v.npy'), np.concatenate([p['trace_v'] for p in parts]))
    np.save(os.path.join(od, 'trace_x.npy'), np.concatenate([p['trace_x'] for p in parts]))
    fleet = []
    for ci in range(nch):
        fleet += json.load(open(os.path.join(od, f'fleet_{ci:03d}.json')))
    json.dump(fleet, open(os.path.join(od, 'fleet.json'), 'w'), default=float)
    man = {'split': split, 'recipe': cfg, 'physics': C.PHYSICS, 'n_rows': int(len(tl)), 'n_vehicles': len(fleet), 'files': {}}
    for f in ('rows.npz', 'trace_v.npy', 'trace_x.npy', 'fleet.json'):
        p = os.path.join(od, f)
        man['files'][f] = dict(bytes=os.path.getsize(p), sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest())
    json.dump(man, open(os.path.join(od, 'MANIFEST.json'), 'w'), indent=1)
    for f in fs:
        os.remove(f)
    for ci in range(nch):
        os.remove(os.path.join(od, f'fleet_{ci:03d}.json'))
    print('finalized', split, man['n_rows'], 'rows', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'finalize':
        finalize(sys.argv[2])
    else:
        split, w, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
        nch = (SPLITS[split]['n'] + CH - 1) // CH
        for ci in range(w, nch, nw):
            one_chunk(split, ci)
        print('WORKER_DONE', split, w, flush=True)
