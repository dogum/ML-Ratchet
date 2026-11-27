"""Judge StarQuake rung r6 (27 Nov 2026) as h1 was judged (rules in NOTEBOOK.md, fixed before scoring): export r6 to numpy,
refit the hybrid's w and k on the fit set read by r6, regenerate val3 / val3neg from their seeds with r6, score the
objective against h1's stored readings; dev gate from the 18 dev stars at five lengths. Also the false-alarm rate on
val3hard (999 pieces of the 18 high-RMS noise_val dwarfs that screen.py dropped, no signal), for h1 and r6 (reported,
not used). Writes work/hybrid/r6_judgement.json. Run from training/ after run_r6.sh train."""
import _paths as _P  # portable paths (see _paths.py)
import os, sys, json, subprocess, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.MODEL)
H = _P.WORK + '/hybrid/'; REL = _P.WORK + '/release_r6/'; PY = sys.executable

def run(*a, env=None):
    e = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', SQ_WORKERS='3', **(env or {})); print('>>', ' '.join(a), env or '', flush=True)
    subprocess.run([PY, os.path.join(_P.SRC, 'hybrid.py'), *a], check=True, env=e)

if __name__ == '__main__':
    os.makedirs(REL, exist_ok=True); npz = REL + 'starquake_r6.npz'
    if not os.path.exists(npz):
        import export, snet as SN, starquake as R
        net = export.export(_P.WORK + '/ckpt/r6.pt', npz); z = np.load(_P.WORK + '/bank/sq6_v2n_a.npz'); i = np.arange(0, 5000, 250)
        P = SN.predict(net, z['x1'][i], z['x2'][i], z['s'][i]); m = R.Net(npz); Q = [m.forward(z['x1'][j].astype(np.float32), z['x2'][j].astype(np.float32), z['s'][j]) for j in i]
        print('numpy vs torch max diff', max(np.max(np.abs(np.array([q[k] for q in Q]) - P[k])) for k in ('ln_numax', 'ln_dnu', 'sig_numax', 'sig_dnu', 'p_det')), flush=True)
    N6 = dict(SQ_NET=npz)
    for name, args, env in (('fit_r6', ('sims', 'fit_r6', 'noise_train', '3000', '11'), N6), ('val3_r6', ('sims', 'val3_r6', 'noise_val', '5001', '12'), N6),
                            ('val3neg_r6', ('negs', 'val3neg_r6', 'noise_val', '2001', '13'), N6), ('val3hard', ('negs', 'val3hard', 'noise_val_hard', '999', '14'), {}),
                            ('val3hard_r6', ('negs', 'val3hard_r6', 'noise_val_hard', '999', '14'), N6), ('dev_r6', ('real', 'dev', 'dev_r6'), N6)):
        if not os.path.exists(H + name + '.npz'): run(*args, env=env)
    if not os.path.exists(H + 'params_r6.json'): run('fit', 'fit_r6', env=dict(SQ_PARAMS='params_r6.json'))
    import hybrid as HY
    p1 = json.load(open(H + 'params.json')); p6 = json.load(open(H + 'params_r6.json'))
    s1 = HY.score('val3', p1, 'val3neg')['hybrid']; s6 = HY.score('val3_r6', p6, 'val3neg_r6')['hybrid']
    d1 = HY.real_report('dev', p1); d6 = HY.real_report('dev_r6', p6)
    four = lambda d: [d['0']['numax']['hybrid'], d['0']['dnu']['hybrid'], d['90']['numax']['hybrid'], d['90']['dnu']['hybrid']]
    ratio = float(np.mean(np.array(four(d6)) / np.array(four(d1))))
    fa = {k: float(np.mean(np.load(H + k + '.npz')['r'][:, 4] > 0.5)) for k in ('val3hard', 'val3hard_r6', 'val3neg', 'val3neg_r6')}
    keep = bool(s6['objective'] > s1['objective'] + 0.01 and ratio <= 1.10)
    out = dict(h1=dict(objective=s1['objective'], score=s1, dev_four=four(d1)), r6=dict(objective=s6['objective'], score=s6, dev_four=four(d6), params=p6),
               dev_ratio=ratio, false_alarm=fa, decision='kept' if keep else 'rejected',
               why=f"objective {s6['objective']:.4f} vs {s1['objective']:.4f}" + ('' if s6['objective'] > s1['objective'] + 0.01 else ' (needs +0.01)') + f"; dev ratio {ratio:.3f}" + ('' if ratio <= 1.10 else ' (gate: at most 1.10)'))
    json.dump(out, open(H + 'r6_judgement.json', 'w'), indent=1); print(json.dumps({k: v for k, v in out.items() if k not in ('h1', 'r6')}, indent=1))
    print('h1', round(s1['objective'], 4), 'r6', round(s6['objective'], 4), 'dev four h1', np.round(four(d1), 4), 'r6', np.round(four(d6), 4))
