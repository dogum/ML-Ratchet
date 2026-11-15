"""Rung v2-r14: re-simulate a bank's SC03 and 20 F cold-FTP dynamometer rows under the EPA test conditions
(dynoproc.py), everything else unchanged.

python bank_dyno.py <bank> <fleet> [workers]     e.g. train2 train2   -> data/bank_<bank>d/chunk_XXX.npz

The rows are found by what the network reads (a dynamometer draw at 35 C is SC03, at -6.7 C the 20 F cold FTP; no
other draw runs at those temperatures). Each keeps its vehicle (vid = fleet chunk x 100 + index), its trip, its start
state (cold soak or warm, start SOC) and so its inputs; only the simulated targets, energy components and trace error
change (new run seed). Rows the new run cannot simulate keep their old values and are counted."""
import _paths as _P
import sys, os, json, glob, time
sys.path.insert(0, _P.SRC)
from common import *
import numpy as np
from roadsim.vehicle import fill_spec
import bank as BK, physics as PHY, dynoproc as DP, wrappers, routes

T_OF = {'SC03': 35.0, 'FTP_cold_20F': -6.7}


def which(B):
    """-> array of cycle names ('' = untouched) for the bank rows."""
    T = B['R'][:, 76] * 20.0
    d = B['dyno'] > 0.5
    out = np.full(len(T), '', object)
    out[d & (np.abs(T - 35.0) < 0.05)] = 'SC03'
    out[d & (np.abs(T + 6.7) < 0.05)] = 'FTP_cold_20F'
    return out


def start_soc(full, arch, disp):
    b = full.get('battery') or {}
    lo, hi = (b.get('soc_window') or [0.2, 0.9])
    if arch == 'conv' or not np.isfinite(disp):
        return None
    if arch.startswith('phev') and disp == 0:
        return lo + 0.02
    return lo + disp * (hi - lo)


def one(args):
    bank, fleet, cp = args
    ci = int(cp[-7:-4])
    op = f'{EN}/data/bank_{bank}d/chunk_{ci:03d}.npz'
    if os.path.exists(op): return (ci, 0, 0)
    B = dict(np.load(cp))
    F_ = json.load(open(f'{EN}/data/fleet_{fleet}/chunk_{ci:03d}.json'))
    w = which(B); idx = np.where(w != '')[0]
    cy = BK.cycles(); nfail = 0
    rng = np.random.default_rng([20261115, ci])
    for i in idx:
        name = str(w[i]); vi = int(B['vid'][i]) - ci * 100
        spec = F_[vi]['spec']; arch = spec['architecture']; full = fill_spec(spec)
        T = T_OF[name]
        tr = BK._fill(cy[name]); N = len(tr['v'])
        tr = dict(tr); tr.update(T_amb=np.full(N, T, np.float32), wind_speed=np.zeros(N, np.float32), grade=np.zeros(N, np.float32),
                                 surface=np.zeros(N, np.int8), rho_air=routes.rho_air(np.full(N, T), np.zeros(N)).astype(np.float32))
        assert abs(float(B['dur'][i]) - N) < 0.5, (ci, i, name, B['dur'][i], N)
        cold = B['cold'][i] > 0.5
        temps = None if cold else warm_temps(T)
        soc = start_soc(full, arch, float(B['soc'][i]))
        seed = int(rng.integers(1 << 30))
        try:
            out = DP.run(spec, tr, name, soc, temps, seed)
            tg = PHY.targets(out, spec, tr['T_amb'], True)
            if arch.startswith('phev'):
                pt = wrappers.phev_patch(spec)
                o2 = DP.run(pt, tr, name, soc, temps, seed)
                tg = tg + PHY.targets(o2, pt, tr['T_amb'], True)
            else:
                tg = tg + tg
        except Exception as ex:
            print('fail', ci, i, spec.get('id'), ex, flush=True); nfail += 1; continue
        L = out['ledger']
        B['comp'][i] = np.array([L['aero'], L['rolling'], L['brake'], L['aux'], L['eng_loss'] + L['mot_loss'] + L['trans_loss'] + L['batt_loss'],
                                 L['grade'], L['kinetic_change'], L['fuel'], L['batt_chem_out']], np.float32) / 1e3
        B['tg'][i] = np.array(tg, np.float32)
        B['trace_miss'][i] = float(out['summary']['trace_rms_mps'])
    os.makedirs(os.path.dirname(op), exist_ok=True)
    np.savez_compressed(op + '.tmp.npz', **B); os.replace(op + '.tmp.npz', op)
    return (ci, len(idx), nfail)


def main(bank, fleet, workers=1):
    jobs = [(bank, fleet, cp) for cp in sorted(glob.glob(f'{EN}/data/bank_{bank}/chunk_*.npz'))]
    t0 = time.time(); tot = [0, 0]
    if workers <= 1:
        res = map(one, jobs)
    else:
        from multiprocessing import Pool
        p = Pool(workers); res = p.imap_unordered(one, jobs)
    for ci, n, nf in res:
        tot[0] += n; tot[1] += nf
        print('chunk', ci, 'rows', n, 'fail', nf, round(time.time() - t0), flush=True)
    print('DONE', bank, 'rows', tot[0], 'failed', tot[1], flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
