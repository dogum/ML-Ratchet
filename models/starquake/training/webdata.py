"""Build the page: weights as base64 float32, the 12 explorer stars' light curves (int16), two lab noise stars, exam
and ratchet summaries, and the Python readings the browser must reproduce. Writes web/starquake.html."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, base64, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.WORK + '/release')
import sq, realbank as RB, classical as C
import starquake as REL
D = _P.WORK + '/data/'; W = _P.LAB + '/'

def b64(a): return base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()

def lc16(y):
    m = np.isfinite(y); sc = float(np.nanmax(np.abs(y)) / 32000) or 1.0; q = np.where(m, np.round(np.where(m, y, 0) / sc), -32768).astype(np.int16)
    return dict(d=base64.b64encode(q.tobytes()).decode(), scale=sc), np.where(q == -32768, np.nan, q.astype(np.float64) * sc)

def main(texts):
    Y = json.load(open(D + 'yu2018.json')); G = json.load(open(D + 'gaia.json')); picks = json.load(open(D + 'web_picks.json'))
    negs = [int(k) for k in np.random.default_rng(11).choice(json.load(open(D + 'split.json'))['exam_neg'], 2, replace=False)]   # fixed random draw of 2 exam dwarfs
    picks = list(picks) + negs
    wz = np.load(_P.WORK + '/release/starquake_weights.npz'); meta = json.loads(str(wz['meta']))
    net = dict(meta=meta, W={k: dict(s=list(wz[k].shape), d=b64(wz[k].ravel())) for k in wz.files if k != 'meta'}); R = REL.StarQuake()
    stars = {}; lcs = {}; hero = int(picks[4])
    for k in picks:
        y = sq.read_npz(f'{D}lc/{k}.npz'); enc, yq = lc16(y); lcs[str(k)] = enc; st = {}; py = {}; cl = {}; rng = np.random.default_rng(k)
        # recover each piece's start the same way realbank.pieces draws it
        for L in RB.LENS:
            if not L: continue
            n = int(L * 86400 / sq.DT)
            for _ in range(50):
                s = int(rng.integers(0, len(y) - n + 1))
                if np.isfinite(y[s:s + n]).mean() > 0.8: st[str(L)] = s; break
        for L in RB.LENS:
            seg = yq if not L else yq[st[str(L)]:st[str(L)] + int(L * 86400 / sq.DT)]
            r = R.read_ppm(seg); py[str(L)] = [r['numax'], r['dnu'], r['numax_err'], r['dnu_err'], r['p_osc']]
            F1, F2, s, (f, P), T = sq.prepare(seg); c = C.measure(f, P, F1, F2); cl[str(L)] = [c['numax'], c['dnu']]
        if k in negs:
            stars[str(k)] = dict(neg=True, kp=round(float(json.loads(str(np.load(f'{D}lc/{k}.npz')['hdr']))['KEPMAG']), 1), starts=st, py=py, cl=cl); print(k, py['0'], flush=True); continue
        y_ = Y[str(k)]; stars[str(k)] = dict(numax=y_['numax'], e_numax=y_['e_numax'], dnu=y_['dnu'], e_dnu=y_['e_dnu'], M=y_['M'], e_M=y_['e_M'], R=y_['R'], e_R=y_['e_R'], teff=y_['teff'], e_teff=y_['e_teff'],
                                             kp=y_['kp'], phase=y_['phase'], gaiaR=G.get(str(k), {}).get('R_flame'), starts=st, py=py, cl=cl)
        print(k, py['0'], flush=True)
    lab = []
    for k in texts['lab_noise']:
        y = sq.read_npz(f'{D}lc/{k}.npz'); enc, _ = lc16(y); lcs[str(k)] = enc; lab.append(dict(kic=int(k), kp=round(float(json.loads(str(np.load(f'{D}lc/{k}.npz')['hdr']))['KEPMAG']), 1)))
    E = json.load(open(_P.WORK + '/exam.json')); RL = json.load(open(_P.WORK + '/ratchet.json'))
    ladder = [dict(ckpt=r['ckpt'], note=r['note'], objective=r['objective'], dev=r['dev']['keys'], accepted=r['accepted'], why=r['why']) for r in RL]
    DATA = dict(net=net, stars=stars, lc=lcs, picks=[str(k) for k in picks], hero=str(hero), lab_noise=lab, exam=dict(lengths=E['lengths'], detection=E['detection'], widen=E['widen'], mass_radius=E['mass_radius'], rows=E['rows'], mr_rows=E['mr_rows']),
                ladder=ladder, method=texts['method'], limits=texts['limits'], use=texts['use'], credits=texts['credits'])
    page = open(W + 'page.html').read().replace('/*SQNET*/', open(W + 'sqnet.js').read()).replace('/*DATA*/null', json.dumps(DATA, separators=(',', ':')))
    open(W + 'starquake.html', 'w').write(page); json.dump({k: stars[k]['py'] for k in stars}, open(W + 'py_readings.json', 'w')); print('page MB', len(page) / 1e6)

if __name__ == '__main__':
    main(json.load(open(W + 'texts.json')))
