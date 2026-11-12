"""Final exam for one checkpoint: reading-level results on real exam data never used for any choice, against the
classical envelope baseline (threshold tau fitted on dev) and the majority-class guess. Error-bar factors (speed, size)
and the class-probability temperature are fitted on dev only (widen-only). Writes exam_<ckpt>.json."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import bnet as BN, classical as CL
CLS = BN.CLS

def fit_calib(dev):
    """widen-only factors on dev: speed and size sigma so that 68% of dev |error|/sigma <= 1; temperature T>=1 minimising dev NLL"""
    zs = [abs(np.log2(r['f_true'] / r['f_pred'])) / (r['f_sig_rel'] / np.log(2)) for r in dev]
    zz = [abs(r['lsize_pred'] - np.log10(r['size_mm'])) / r['lsize_sig'] for r in dev if r['cls'] in ('OR', 'IR', 'B') and r['size_mm'] > 0]
    return dict(speed=float(max(1, np.quantile(zs, 0.68))), size=float(max(1, np.quantile(zz, 0.68))) if zz else 1.0)

def fit_temp(net):
    """class-probability temperature T >= 1 (widen-only) minimising the dev negative log-likelihood"""
    best = None
    for T in np.arange(1.0, 6.01, 0.25):
        R = [r for r in BN.readings(net, 'dev', dict(temp=T)) if r['kind'] == 'main' and r['cls'] in CLS]
        nll = -np.mean([np.log(max(r['p'][CLS.index(r['cls'])], 1e-6)) for r in R])
        if best is None or nll < best[0] - 1e-9: best = (nll, float(T))
    return best[1]

def fit_nbias(net, T):
    """one additive log-prior on 'healthy', fitted on dev (the network's counterpart of the classical threshold tau):
    maximise dev balanced accuracy, ties -> smallest |bias|"""
    best = None
    for b in sorted(np.arange(-3.0, 3.01, 0.25), key=abs):
        R = [r for r in BN.readings(net, 'dev', dict(temp=T, nbias=b)) if r['kind'] == 'main' and r['cls'] in CLS]
        ba = BN.bal_acc(np.array([CLS.index(r['cls']) for r in R]), np.array([CLS.index(r['pred']) for r in R]))
        if best is None or ba > best[0] + 1e-9: best = (ba, float(b))
    return best[1], best[0]

def fit_tau(dev):
    sc = np.array([r['classical'] for r in dev]); y = np.array([CLS.index(r['cls']) for r in dev]); best = None
    for t in np.arange(0.2, 1.5, 0.01):
        b = BN.bal_acc(y, CL.decide(sc, t))
        if best is None or b > best[0] + 1e-9: best = (b, float(t))
    return best[1], best[0]

def table(R, key, pred='pred'):
    out = {}
    for r in R:
        k = key(r)
        if k is None: continue
        o = out.setdefault(str(k), dict(n=0, correct=0)); o['n'] += 1; o['correct'] += int(r[pred] == r['cls'])
    for o in out.values(): o['acc'] = round(o['correct'] / o['n'], 3)
    return dict(sorted(out.items()))

def block(R, name):
    if not R: return {}
    y = np.array([CLS.index(r['cls']) for r in R]); p = np.array([CLS.index(r['pred']) for r in R]); pc = np.array([CLS.index(r['pred_cl']) for r in R])
    maj = np.bincount(y, minlength=4).argmax()
    conf = lambda a: [[int(np.sum((y == i) & (a == j))) for j in range(4)] for i in range(4)]
    perc = lambda a: {CLS[c]: round(float(np.mean(a[y == c] == c)), 3) for c in range(4) if np.any(y == c)}
    return dict(name=name, n=len(R), net=dict(acc=round(float(np.mean(p == y)), 3), bal_acc=round(BN.bal_acc(y, p), 3), per_class=perc(p), conf=conf(p)),
                classical=dict(acc=round(float(np.mean(pc == y)), 3), bal_acc=round(BN.bal_acc(y, pc), 3), per_class=perc(pc), conf=conf(pc)),
                majority=dict(cls=CLS[maj], acc=round(float(np.mean(y == maj)), 3), bal_acc=round(1 / len(set(y.tolist())), 3)))

def run(ckpt):
    net = BN.load_net(_P.WORK + '/ckpt/' + ckpt)
    dev = [r for r in BN.readings(net, 'dev') if r['kind'] == 'main']; cal = fit_calib(dev); tau, tau_ba = fit_tau(dev); cal['temp'] = fit_temp(net); cal['nbias'], nb_ba = fit_nbias(net, cal['temp'])
    ex = BN.readings(net, 'exam', cal)
    for r in ex: r['pred_cl'] = CLS[int(CL.decide(np.array([r['classical']]), tau)[0])]
    main = [r for r in ex if r['kind'] == 'main' and r['cls'] in CLS]
    cw = [r for r in main if r['src'] == 'CWRU']; near = [r for r in cw if r['loc'] == 'near']; pb = [r for r in main if r['src'] == 'PB']
    res = dict(ckpt=ckpt, calib=cal, tau=tau, tau_dev_bal_acc=tau_ba, nbias_dev_bal_acc=nb_ba,
               cwru_near=block(near, 'CWRU, sensor at the faulty bearing'), cwru_far=block([r for r in cw if r['loc'] == 'far'], 'CWRU, sensor at the other end'),
               cwru_base=block([r for r in cw if r['loc'] == 'base'], 'CWRU, base-plate sensor'), pb=block(pb, 'Paderborn (new machine)'),
               by_set=table(near, lambda r: r['set']), by_load=table(near, lambda r: r['load']), by_size=table(near, lambda r: '%.3f in' % (r['size_mm'] / 25.4) if r['cls'] != 'N' else None),
               by_loc_net=table(cw, lambda r: r['loc']), by_loc_cl=table(cw, lambda r: r['loc'], 'pred_cl'), by_pos=table(near, lambda r: r['pos'] if r['cls'] == 'OR' else None),
               by_set_cls=table(near, lambda r: r['set'] + ' ' + r['cls']), cl_by_set_cls=table(near, lambda r: r['set'] + ' ' + r['cls'], 'pred_cl'),
               pb_by_bearing=table(pb, lambda r: r['bearing_id']), pb_by_bearing_cl=table(pb, lambda r: r['bearing_id'], 'pred_cl'), pb_by_cond=table(pb, lambda r: r['cond']),
               pb_by_kind=table(pb, lambda r: ('artificial' if r['mode'] == 'artificial' else 'real damage') if r['cls'] != 'N' else 'healthy'),
               pb_by_kind_cl=table(pb, lambda r: ('artificial' if r['mode'] == 'artificial' else 'real damage') if r['cls'] != 'N' else 'healthy', 'pred_cl'))
    other = [r for r in ex if r['kind'] == 'other']
    res['other_bearing'] = dict(n=len(other), net_says_healthy=round(float(np.mean([r['pred'] == 'N' for r in other])), 3), classical_says_healthy=round(float(np.mean([r['pred_cl'] == 'N' for r in other])), 3))
    comb = [r for r in ex if r['cls'] == 'OR+IR']
    res['pb_combined'] = dict(n=len(comb), net=dict((c, round(float(np.mean([r['pred'] == c for r in comb])), 3)) for c in CLS), classical=dict((c, round(float(np.mean([r['pred_cl'] == c for r in comb])), 3)) for c in CLS))
    sp = {}
    for nm, S in (('cwru', cw), ('pb', pb)):
        e = np.array([r['f_pred'] / r['f_true'] - 1 for r in S]); z = np.abs(np.log2(np.array([r['f_true'] / r['f_pred'] for r in S]))) / (np.array([r['f_sig_rel'] for r in S]) / np.log(2))
        sp[nm] = dict(net_mae_pct=round(100 * float(np.mean(np.abs(e))), 3), nominal_mae_pct=round(100 * float(np.mean([abs(r['f_nom'] / r['f_true'] - 1) for r in S])), 3),
                      classical_mae_pct=round(100 * float(np.mean([abs(r['f_classical'] / r['f_true'] - 1) for r in S])), 3),
                      cover_1sig=round(float(np.mean(z <= 1)), 3), cover_2sig=round(float(np.mean(z <= 2)), 3), median_sig_pct=round(100 * float(np.median([r['f_sig_rel'] for r in S])), 3))
    res['speed'] = sp; sz = {}
    for nm, S in (('cwru_near', [r for r in near if r['cls'] != 'N']), ('pb', [r for r in pb if r['cls'] != 'N'])):
        t = np.log10([r['size_mm'] for r in S]); pz = np.array([r['lsize_pred'] for r in S]); sg = np.array([r['lsize_sig'] for r in S])
        sz[nm] = dict(n=len(S), mae_dex=round(float(np.mean(np.abs(pz - t))), 3), median_guess_mae_dex=round(float(np.mean(np.abs(t - np.median(t)))), 3),
                      cover_1sig=round(float(np.mean(np.abs(pz - t) <= sg)), 3), by_true={'%.2f mm' % 10 ** k: round(float(10 ** np.median(pz[t.round(4) == k])), 2) for k in sorted(set(t.round(4)))})
    res['size'] = sz
    keep = ('src', 'rec', 'label_txt', 'set', 'load', 'cls', 'chan', 'bearing', 'loc', 'kind', 'pred', 'pred_cl', 'p', 'f_true', 'f_nom', 'f_pred', 'f_sig_rel', 'size_mm', 'lsize_pred', 'lsize_sig', 'bearing_id', 'cond', 'mode', 'classical')
    json.dump(dict(res, readings=[{k: r.get(k) for k in keep} for r in ex]), open(_P.WORK + '/exam_%s.json' % ckpt.replace('.pt', ''), 'w'))
    return res

if __name__ == '__main__':
    r = run(sys.argv[1])
    for k in ('calib', 'tau', 'tau_dev_bal_acc', 'nbias_dev_bal_acc'): print(k, r[k])
    for k in ('cwru_near', 'cwru_far', 'cwru_base', 'pb'):
        b = r[k]; print(b['name'], 'n', b['n'], '| net', b['net']['acc'], b['net']['bal_acc'], b['net']['per_class'], '| classical', b['classical']['acc'], b['classical']['bal_acc'], b['classical']['per_class'], '| majority', b['majority'])
        print('   net conf', b['net']['conf'], ' classical conf', b['classical']['conf'])
    for k in ('by_set_cls', 'cl_by_set_cls', 'by_load', 'by_size', 'by_pos', 'by_loc_net', 'by_loc_cl', 'pb_by_kind', 'pb_by_kind_cl', 'pb_by_bearing', 'pb_by_bearing_cl', 'pb_by_cond', 'other_bearing', 'pb_combined', 'speed', 'size'):
        print(k, json.dumps({a: (b['acc'] if isinstance(b, dict) and 'acc' in b else b) for a, b in r[k].items()}))
