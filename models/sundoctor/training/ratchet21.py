"""SunDoctor v2.1 ladder (15 Nov 2026). Physics v2.1 adds multi-day inverter trips: an inverter trips at a random time
and stays dark for 2-14 days until it is reset (simbatch2.LONG_OUT). The v1/v2 simulators only switched an inverter off
for single days, and the outage probe (outage_probe.py) showed the release catches only 78% of simulated multi-day trips
with a sensor (fixed mounts) and 68% without one.

Because the physics changed, the frozen set changes (docs/method.md): val_set_v21 = 2,000 fortnights, seed 779, v2.1
physics (weather fronts as rung 8, LONG_OUT = 0.15), half with an irradiance sensor. Same objective and rule as the v2
ladder: the mean of the with-sensor and home halves of [mean skill over pf, frac, soiling, shading, clipping, outage,
snow, stuck + 0.5 step AUC + 0.5 dead-string counting accuracy]; keep if it rises by more than 0.01. Only the released
rung 8 survives as weights (the s3-s7 checkpoints were not kept), so it is the starting point; earlier rungs cannot be
re-scored. The outage probe and the simulated small-tracker-loss AUC are reported beside it, not used to decide.

    python ratchet21.py <weights.npz | ckpt.pt> "<note>"      logs to work/ratchet21.json
Real exams are scored separately (realexam2.py), once, after the release choice."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, time, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.MODEL)
import evaluate2 as E2, simbatch2 as SB, train2 as T, outage_probe as OP
VAL21 = _P.WORK + '/val_set_v21.npz'; LOG = _P.WORK + '/ratchet21.json'

def val_set21(n=2000, D=14):
    if os.path.exists(VAL21):
        z = np.load(VAL21); return z['X'], z['Y'], z['G'], {k[2:]: z[k] for k in z.files if k.startswith('m_')}
    SB.FRONTS, SB.LONG_OUT = True, 0.15; rng = np.random.default_rng(779); Xs, Ys, Gs, Ms = [], [], [], []
    for b in range(n // 100):
        X, Y, G, m = T.batch(rng, 100, D, home=np.arange(100) % 2 == 1); Xs.append(X.astype(np.float16)); Ys.append(Y); Gs.append(G); Ms.append(m)
    SB.FRONTS, SB.LONG_OUT = False, 0.0
    M = {k: np.concatenate([m[k] for m in Ms]) for k in ('n_strings', 'kind', 'rate', 'sat', 'tracker', 'home', 'cdte', 'snow_any', 'stuck_any')}
    np.savez_compressed(VAL21, X=np.concatenate(Xs), Y=np.concatenate(Ys), G=np.concatenate(Gs), **{'m_' + k: v for k, v in M.items()})
    return val_set21(n, D)

def predictor(path):
    if path.endswith('.npz'):
        import sundoctor
        net = sundoctor.SunDoctorNet(path); return net, (lambda _n, X: net.predict(np.asarray(X, np.float32)))
    net = E2.load(path); return net, E2.predict

def small_tracker_auc(P, X, Y, M):
    """simulated analogue of the Colorado row: tracker days with no outage and no snow, AUC of the stuck output for a
    tracker loss over 2%, and the same among days whose loss is 0-5%"""
    yi = lambda k: Y[:, SB.DAY_LABELS.index(k)]; valid = (X[:, 9].astype(np.float32).mean(-1) > 0.2) & M['tracker'][:, None]
    ok = valid & (yi('outage') < 0.2) & (yi('snow') == 0); st = yi('stuck')[ok]; ps = P['stuck'][ok]; out = {}
    out['auc_over_2pct'] = E2.auc(ps, (st > 0.02).astype(float)); sm = st < 0.05
    out['auc_2_5pct_vs_under_2pct'] = E2.auc(ps[sm], (st[sm] > 0.02).astype(float)); out['n_days'] = int(ok.sum()); out['n_2_5pct'] = int(((st > 0.02) & sm).sum())
    return out

def score(path):
    net, pf = predictor(path); X, Y, G, M = val_set21(); P = pf(net, X)
    r = {'poa': E2.metrics(P, X, Y, G, M, ~M['home']), 'home': E2.metrics(P, X, Y, G, M, M['home']), 'trackers': E2.metrics(P, X, Y, G, M, M['tracker'])}
    r['objective'] = 0.5 * (r['poa']['objective'] + r['home']['objective']); r['tracker_small_loss'] = small_tracker_auc(P, X, Y, M)
    Xp, Yp, Mp = OP.probe_set(600, 4242); r['outage_probe'] = OP.score(pf(net, Xp), Yp, Mp)[0]
    return r

if __name__ == '__main__':
    path, note = sys.argv[1], sys.argv[2]; t = time.time(); v = score(path)
    log = json.load(open(LOG)) if os.path.exists(LOG) else []; inc = next((r for r in reversed(log) if r['accepted']), None)
    rec = {'ckpt': os.path.basename(path), 'note': note, 'val': v, 'objective': v['objective'], 'eval_s': round(time.time() - t)}
    if inc is None: rec['accepted'] = True; rec['why'] = 'starting point (the v2.0 release, rung 8)'
    else: rec['accepted'] = bool(v['objective'] > inc['objective'] + 0.01); rec['why'] = f"objective {v['objective']:.3f} vs {inc['objective']:.3f}"; rec['vs'] = inc['ckpt']
    log.append(rec); json.dump(log, open(LOG, 'w'), indent=1)
    sk = lambda h: {k: round(v[h][k]['skill'], 3) for k in E2.SKILLS}
    print(f"{rec['ckpt']}: objective {v['objective']:.4f} (with sensor {v['poa']['objective']:.4f}, home {v['home']['objective']:.4f})")
    print('   skills with sensor', sk('poa')); print('   skills home       ', sk('home'))
    print('   small tracker losses', v['tracker_small_loss'])
    print('   outage probe', {k: (x['caught'], x['n']) for k, x in v['outage_probe'].items()})
    print('  ', 'ACCEPTED' if rec['accepted'] else 'rejected', '-', rec['why'])
