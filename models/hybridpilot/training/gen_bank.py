"""Build an imitation/DAgger bank: run cases with the DP teacher labelling every traction decision.
usage: gen_bank.py ROUND N_CASES SEED ACT [NET.npz] [BETA]
  ACT = teacher (DP acts, with exploration bursts) | mix (student acts, teacher with prob BETA) """
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hp_env import *
from hp_policy import HybridPilot, MLP, make_stage, dp_backward, teacher_solution
from roadsim import simulate
from roadsim.powertrain import ARCH_CLASSES
from roadsim.sim import normalize_trip

def main():
    rnd, n, seed, act = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    net = MLP.load(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != '-' else None
    beta = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    extra = json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}
    outdir = f'{ROOT}/data/bank_{rnd}'; os.makedirs(outdir, exist_ok=True)
    fleet = gen_fleet(400, 'train')
    tr_v, _ = train_noise_partition()
    ved = ved_routes('train_noise', vehicles=tr_v)
    cases = train_cases(n, seed, fleet=fleet, ved=ved)
    shard = 25
    for s0 in range(0, n, shard):
        fn = f'{outdir}/s{seed}_{s0:05d}.npz'
        if os.path.exists(fn):
            continue
        X, S, ON, K, CID, AR = [], [], [], [], [], []
        t0 = time.time(); steps = 0
        for ci in range(s0, min(s0 + shard, n)):
            c = cases[ci]
            veh = get_vehicle(c['spec']); b = veh.battery
            trip = normalize_trip(c['trip'])
            pt = ARCH_CLASSES[veh.arch](veh)
            stage = make_stage(veh, pt, trip)
            sol = teacher_solution(veh, stage, len(trip['v']), is_phev(veh.arch))
            rng = np.random.default_rng(seed * 100003 + ci)
            prm = dict(net=net, preview_s=c['preview'], nav=c['nav'], teacher=dict(sol=sol), act=act if net is not None else 'teacher',
                       beta=beta, rng=rng, Kp=6.0, explore=0.012 if act == 'teacher' else 0.004, record=True)
            prm.update(extra)
            st = HybridPilot(**prm)
            try:
                o = simulate(c['spec'], c['trip'], strategy=st, init_soc=c['soc0'], init_temps=c['temps'], record=False, options='roadsim-physics-v1')
            except Exception as ex:
                print('fail', ci, ex, flush=True); continue
            L = st.log
            if not L: continue
            X.append(np.stack([l[0] for l in L])); S.append(np.array([l[1] for l in L], np.float32))
            ON.append(np.array([l[2] for l in L], np.float32)); K.append(np.array([l[3] for l in L], np.int32))
            CID.append(np.full(len(L), ci, np.int32)); AR.append(np.full(len(L), HYBRID_ARCHS.index(veh.arch), np.int8))
            steps += len(c['trip']['v'])
        np.savez_compressed(fn, X=np.concatenate(X).astype(np.float32), s=np.concatenate(S), on=np.concatenate(ON), k=np.concatenate(K),
                            case=np.concatenate(CID), arch=np.concatenate(AR))
        print(f'{fn} {sum(len(x) for x in X)} samples, {steps} steps, {time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    main()
