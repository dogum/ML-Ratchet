"""Network twin test on DEV only: each confirmed dev event's storm is simulated from its real IBTrACS track (twin.simulate_track)
and injected into 4 quiet background windows of the same buoy; the network reads twin and real. If the network makes the
same distance / time errors on the twins as on the real windows, the errors come from storm behaviour (moving, growing
storms) that the training prior lacks, not from the buoy data. Usage: python3 twin_net.py ckpt [a_hs]"""
import os, sys, json, numpy as np
import twin as TW, events as EV, realdata as RD, simbank as SB, evaluate as E, swellsim as SS, fetch

if __name__ == '__main__':
    ck = sys.argv[1]; a_hs = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    net = E.load_net(ck); ev = EV.load('dev', 'confirmed'); BG = SB.Backgrounds(); rng = np.random.default_rng(5)
    rows = []; Xr = EV.windows(ev); Pr = E.predict(net, Xr)
    for i, e in enumerate(ev):
        s = TW.storm_track(e['sid']); Et = TW.simulate_track(e, s, a_hs=a_hs)
        if e['buoy'] not in BG.B: continue
        d = BG.B[e['buoy']]; X = []
        for k in range(4):
            st = int(rng.choice(d['starts'])); sl = slice(st, st + SS.NT); Eb = d['E'][sl].astype(np.float32)
            Ex = Eb + np.round(Et * rng.chisquare(24, Et.shape) / 24, 2); X.append(np.stack([Ex, d['C'][sl].astype(np.float32), d['S'][sl].astype(np.float32)]))
        P = E.predict(net, np.stack(X))
        rows.append(dict(name=e['name'], buoy=e['buoy'], D=e['D'] / 1e3, t0=e['t0_rel'], real_D=float(Pr['D_km'][i]), real_t0=float(Pr['t0'][i]),
                         twin_D=float(np.median(P['D_km'])), twin_t0=float(np.median(P['t0']))))
        r = rows[-1]; print(f"{r['name'][:10]:10s} {r['buoy']} D {r['D']:5.0f} real {r['real_D']:5.0f} twin {r['twin_D']:5.0f} | t0 {r['t0']:6.1f} real {r['real_t0']:6.1f} twin {r['twin_t0']:6.1f}", flush=True)
    D = np.array([r['D'] for r in rows]); f = lambda k: np.log10(np.array([r[k] for r in rows]) / D)
    g = lambda k: np.array([r[k] for r in rows]) - np.array([r['t0'] for r in rows])
    summ = dict(ckpt=ck, a_hs=a_hs, n=len(rows), real_logD_bias=float(np.median(f('real_D'))), twin_logD_bias=float(np.median(f('twin_D'))),
                real_logD_mae=float(np.mean(np.abs(f('real_D')))), twin_logD_mae=float(np.mean(np.abs(f('twin_D')))),
                real_t0_bias=float(np.median(g('real_t0'))), twin_t0_bias=float(np.median(g('twin_t0'))), real_t0_mae=float(np.mean(np.abs(g('real_t0')))), twin_t0_mae=float(np.mean(np.abs(g('twin_t0')))),
                corr_logD_err=float(np.corrcoef(f('real_D'), f('twin_D'))[0, 1]), corr_t0_err=float(np.corrcoef(g('real_t0'), g('twin_t0'))[0, 1]))
    print(json.dumps(summ, indent=1)); out = os.path.join(E.ROOT, f'twin_net_{os.path.basename(ck).split(".")[0]}.json'); json.dump(dict(summary=summ, rows=rows), open(out, 'w'), indent=1)
