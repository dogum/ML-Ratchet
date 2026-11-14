"""Build an imitation/DAgger bank: run cases with the DP teacher labelling every traction decision.
usage: gen_bank.py ROUND N_CASES SEED ACT [NET.npz|-] [BETA] [extra json]
  ACT = teacher (DP acts, with exploration bursts) | mix (student acts, teacher with prob BETA)
  extra json: HybridPilot params for the acting student (Kp, onoff, ...), plus
     timed: true   -> teacher = DP with the physics' minimum on/off times (v2: 8 s / 3 s)
     teacher_start_pen: J  -> virtual start cost in the teacher's choice (drivability)
     fleet: 'gen' (default) | 'gen+twin'  (training fleet; see hp_env.train_fleet)
     ved_frac, cyc_frac: route mix
Shards of 25 cases, resumable; HP_WORKERS processes (default 5)."""
import sys, os, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from hp_env import *
from hp_policy import HybridPilot, MLP, make_stage, make_pt, teacher_solution
from roadsim import simulate
from roadsim.sim import normalize_trip

SHARD = 25


def run_one(job):
    c, ci, seed, act, net_path, beta, extra = job
    net = MLP.load(net_path) if net_path else None
    extra = dict(extra)
    timed = bool(extra.pop('timed', False))
    t_pen = float(extra.pop('teacher_start_pen', 0.0))
    for k in ('fleet', 'ved_frac', 'cyc_frac'):
        extra.pop(k, None)
    veh = get_vehicle(c['spec'])
    trip = normalize_trip(c['trip'])
    stage = make_stage(veh, make_pt(veh), trip)
    sol = teacher_solution(veh, stage, len(trip['v']), is_phev(veh.arch), timed=timed)
    rng = np.random.default_rng(seed * 100003 + ci)
    prm = dict(net=net, preview_s=c['preview'], nav=c['nav'], teacher=dict(sol=sol, start_pen=t_pen),
               act=act if net is not None else 'teacher', beta=beta, rng=rng, Kp=6.0,
               explore=0.012 if act == 'teacher' else 0.004, record=True)
    prm.update(extra)
    st = HybridPilot(**prm)
    try:
        simulate(c['spec'], c['trip'], strategy=st, init_soc=c['soc0'], init_temps=c['temps'], record=False, options=PHYS)
    except Exception as ex:
        return ci, None, repr(ex)
    L = st.log
    if not L:
        return ci, None, 'empty'
    return ci, dict(X=np.stack([l[0] for l in L]), s=np.array([l[1] for l in L], np.float32),
                    on=np.array([l[2] for l in L], np.float32), k=np.array([l[3] for l in L], np.int32),
                    arch=HYBRID_ARCHS.index(veh.arch), steps=len(c['trip']['v'])), None


def main():
    rnd, n, seed, act = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    net_path = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != '-' else None
    beta = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    extra = json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}
    outdir = f'{ROOT}/data/bank_{rnd}'; os.makedirs(outdir, exist_ok=True)
    json.dump(dict(round=rnd, n=n, seed=seed, act=act, net=net_path, beta=beta, extra=extra, physics=PHYS),
              open(f'{outdir}/config_s{seed}.json', 'w'), indent=1)
    fleet = train_fleet(extra.get('fleet', 'gen'))
    tr_v, _ = train_noise_partition()
    ved = ved_routes('train_noise', vehicles=tr_v)
    cases = train_cases(n, seed, fleet=fleet, ved=ved, ved_frac=extra.get('ved_frac', 0.45), cyc_frac=extra.get('cyc_frac', 0.2))
    todo = [s0 for s0 in range(0, n, SHARD) if not os.path.exists(f'{outdir}/s{seed}_{s0:05d}.npz')]
    jobs = [(cases[ci], ci, seed, act, net_path, beta, extra) for s0 in todo for ci in range(s0, min(s0 + SHARD, n))]
    res = {}
    t0 = time.time()
    with Pool(int(os.environ.get('HP_WORKERS', '5'))) as pool:
        for ci, r, err in pool.imap_unordered(run_one, jobs):
            if err:
                print('fail', ci, err, flush=True)
            res[ci] = r
            s0 = (ci // SHARD) * SHARD
            idx = list(range(s0, min(s0 + SHARD, n)))
            if all(i in res for i in idx):
                R = [res[i] for i in idx if res[i] is not None]
                fn = f'{outdir}/s{seed}_{s0:05d}.npz'
                np.savez_compressed(fn, X=np.concatenate([r['X'] for r in R]).astype(np.float32), s=np.concatenate([r['s'] for r in R]),
                                    on=np.concatenate([r['on'] for r in R]), k=np.concatenate([r['k'] for r in R]),
                                    case=np.concatenate([np.full(len(r['s']), i, np.int32) for i, r in zip([i for i in idx if res[i] is not None], R)]),
                                    arch=np.concatenate([np.full(len(r['s']), r['arch'], np.int8) for r in R]))
                print(f'{fn} {sum(len(r["s"]) for r in R)} samples, {sum(r["steps"] for r in R)} steps, {time.time() - t0:.0f}s', flush=True)
                for i in idx:
                    res.pop(i, None)


if __name__ == '__main__':
    main()
