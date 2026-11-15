"""Round 2 (15 Nov 2026): can the dev files show the exam's two CWRU losses (near sensor 0.39 vs envelope 0.49, base
plate 0.50 vs 0.67)? Reads every dev reading with the released numpy module (bearingnet.py, its dev-fitted calibration)
and with the classical envelope method (classical.py, threshold tau fitted on the same dev readings, as exam.py does),
and reports balanced accuracy by sensor position. Dev files only; the exam is never read here.

    python bn_devprobe.py            writes work/devprobe_r2.json
"""
import _paths as _P
import os, sys, json, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.MODEL)
import realdata as RD, classical as CL, features as FE, bearingnet as BNM
CLS = ['N', 'OR', 'IR', 'B']


def cwru_dev_ids():
    """the CWRU half of realdata.split(): same seed, same draw order (the Paderborn draws come after)"""
    rng = np.random.default_rng(RD.SEED); cat = json.load(open(RD.D + 'cwru_catalogue.json'))
    norm = sorted(c['id'] for c in cat if c['set'] == 'normal'); p = list(rng.permutation(norm)); dev = [int(p[1])]
    for s in ('de12', 'de48', 'fe12'):
        for c in ('OR', 'IR', 'B'):
            ids = sorted(x['id'] for x in cat if x['set'] == s and x['cls'] == c); dev.append(int(rng.choice(ids)))
    return dev, int(p[0])


def bal_acc(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def classical_scores(x, geom, f_nom):
    x = FE.pad_short(np.asarray(x, float)); st = FE.windows(x); f_cl = CL.speed(x[st[0]:st[0] + FE.NWIN], f_nom)
    return np.mean([CL.scores(x[s:s + FE.NWIN], geom, f_cl) for s in st], 0)


def read_all(ids, net=None):
    net = net or BNM.BearingNet(); out = []
    for r in RD.cwru_readings(ids):
        g = r['geom']; res = net.read(r['x'], fs=FE.FS, bearing=dict(n=g['n'], d=g['d'], D=g['D']), rpm_nominal=r['f_nom'] * 60)
        out.append(dict({k: r[k] for k in ('rec', 'label_txt', 'set', 'cls', 'chan', 'bearing', 'loc', 'kind', 'size_mm')}, size_pred_mm=float(res['size_mm']), p=[float(v) for v in res['p']] if not isinstance(res['p'], dict) else [float(res['p'][c]) for c in CLS],
                        pred=res['condition'], classical=classical_scores(r['x'], g, r['f_nom']).tolist()))
        print(out[-1]['rec'], out[-1]['loc'], out[-1]['kind'], out[-1]['cls'], '->', out[-1]['pred'], flush=True)
    return out


def summarise(R):
    y = np.array([CLS.index(r['cls']) for r in R]); sc = np.array([r['classical'] for r in R]); best = None
    for t in np.arange(0.2, 1.5, 0.01):
        b = bal_acc(y, CL.decide(sc, t))
        if best is None or b > best[0] + 1e-9: best = (b, float(t))
    tau = best[1]; pc = CL.decide(sc, tau); pn = np.array([CLS.index(r['pred']) for r in R]); out = dict(tau=tau)
    S = [r for r in R if r['cls'] != 'N' and r['loc'] == 'near']; t = np.log10([r['size_mm'] for r in S]); pz = np.log10([r['size_pred_mm'] for r in S])
    out['size_near'] = dict(n=len(S), net_mae_dex=float(np.mean(np.abs(pz - t))), median_guess_mae_dex_in_sample=float(np.mean(np.abs(t - np.median(t)))))
    for loc in ('near', 'far', 'base', 'all'):
        m = np.array([(loc == 'all' or r['loc'] == loc) for r in R])
        out[loc] = dict(n=int(m.sum()), net=bal_acc(y[m], pn[m]), classical=bal_acc(y[m], pc[m]), net_correct=int((pn[m] == y[m]).sum()), cl_correct=int((pc[m] == y[m]).sum()))
    return out


if __name__ == '__main__':
    dev, noise = cwru_dev_ids(); print('CWRU dev files', dev, 'background', noise)
    R = [r for r in read_all(dev) if r['kind'] == 'main']; S = summarise(R)
    print(json.dumps(S, indent=1)); json.dump(dict(dev=dev, summary=S, readings=R), open(os.path.join(_P.WORK, 'devprobe_r2.json'), 'w'), indent=1)
