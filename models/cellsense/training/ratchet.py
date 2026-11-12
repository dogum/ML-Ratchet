"""CellSense ratchet. Objective on a frozen set of 4,000 simulated new/aged pairs (seed 999), measured the way the cycler
measures (fixed noise draw), half read with the cell's own early curve and half from the aged curve alone: mean over the two
halves of the mean skill (1 - MAE / MAE of always guessing the median, floored at 0) over total capacity, pulse resistance, lost lithium,
lost graphite, lost LFP, resistance growth and fade (the health half), plus lifetime skill on 3,000 simulated cells (the lifetime
half; zero for a system with no lifetime network). A candidate is kept only if it beats the incumbent by more than 0.01,
AND (the dev gate, added after rung 3 and applied to every rung) its total-capacity error on the six calibration cells,
which are excluded from every exam, is no more than 1 mAh worse than the incumbent's, AND (added with rung 6) its lifetime
forecast on those six cells, read every 50 cycles through life, is no more than 2 points of percentage error worse. From
rung 6 the health half has a third part: the same frozen pairs read as partial discharges (partial_set). A rung is a system:
a health network, optionally a lifetime network and a partial-discharge network that takes any curve stopping short of
2.0 V. The exam cells are reported, never used to choose."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, time, numpy as np
sys.path.insert(0, _P.SRC); import model as M, realexam as RX, train as TR
LOG = _P.WORK + '/ratchet.json'
# frozen lifetime set, versioned with the aging physics: v1 = rungs 4-5's model, v2 = early-life physics (slow graphite tied
# to graphite loss), v3 = v2 with the late-life knobs set by a whole-life curve check (v2.1). A new version re-scores every rung on it and re-takes every decision (rescore()).
LIFEVAL = _P.WORK + '/data/lifeval_v3.npz'; SK = ['q_total', 'ir', 'lli', 'lam_ne', 'lam_pe', 'r_growth', 'fade']

def val_set():
    z = np.load(_P.WORK + '/data/val.npz'); rng = np.random.default_rng(12345)
    Q1, T1, Q0, T0 = TR.augment(rng, z['Q1'], z['T1c'], z['Q0'], z['T0c'], p_ref=1.0); half = np.arange(len(Q1)) % 2 == 1
    Q0 = np.where(half[:, None], np.nan, Q0); T0 = np.where(half[:, None], np.nan, T0)
    return Q1, T1, Q0, T0, z['y'], half

def life_val(net):
    """lifetime skill on 3,000 simulated cells read once at a random cycle (seed 998): 1 - MAE / MAE of guessing the median, on log10 life"""
    if 'log_life' not in getattr(net, 'labels', []): return {'skill': 0.0, 'note': 'no lifetime head'}
    z = np.load(LIFEVAL); rng = np.random.default_rng(54321); Q1, T1, Q0, T0 = TR.augment(rng, z['Q1'], z['T1c'], z['Q0'], z['T0c'], p_ref=1.0)
    P = RX.predict(net, Q1, T1, Q0, T0, z['s']); j = [str(x) for x in z['labels']].index('log_life'); y = z['y'][:, j]
    e = np.abs(P['log_life'] - y); base = np.abs(y - np.median(y)).mean(); jl = [str(x) for x in z['labels']].index('left')
    return {'skill': float(1 - e.mean() / base), 'mae_log': float(e.mean()), 'mpe': float(100 * np.mean(np.abs(10 ** P['log_life'] / 10 ** y - 1))),
            'left_mae': float(np.mean(np.abs(P['left'] - z['y'][:, jl]))), 'in_1sigma': float(np.mean(e < P['log_life_sig']))}

PARTIAL = True    # objective v3 (added with rung 6): a third of the health score is read from partial discharges

def partial_set():
    """the same frozen pairs as partial discharges from full, cut at 30-95% of what each curve delivers (seed 777); the
    reference is the full early curve for even pairs and cut at the same depth for odd ones"""
    z = np.load(_P.WORK + '/data/val.npz'); rng = np.random.default_rng(777)
    Q1, T1, Q0, T0 = TR.augment(rng, z['Q1'], z['T1c'], z['Q0'], z['T0c'], p_ref=1.0); D = np.nanmax(Q1, 1) * rng.uniform(0.3, 0.95, len(Q1))
    Q1, T1 = TR.truncate(Q1, T1, D); odd = np.arange(len(Q1)) % 2 == 1; Q0, T0 = TR.truncate(Q0, T0, np.where(odd, D, np.inf))
    return Q1, T1, Q0, T0, z['y']

def evaluate(net, pnet=None):
    """pnet: a partial-discharge network, if the system routes curves that stop short of 2.0 V to one"""
    Q1, T1, Q0, T0, Y, half = val_set(); P = RX.predict(net, Q1, T1, Q0, T0); out = {}
    sets = [('with_ref', P, Y, ~half), ('no_ref', P, Y, half)]
    if PARTIAL:
        q1, t1, q0, t0, Yp = partial_set(); sets.append(('partial', RX.predict(pnet or net, q1, t1, q0, t0), Yp, np.ones(len(Yp), bool)))
    for name, PP, YY, m in sets:
        r = {}
        for j, k in enumerate(M.LABELS):
            y = YY[m, j]; e = np.abs(PP[k][m] - y); base = np.abs(y - np.median(y)).mean()
            r[k] = {'mae': float(e.mean()), 'skill': float(1 - e.mean() / base), 'in_1sigma': float(np.mean(e < PP[k + '_sig'][m]))}
        r['objective'] = float(np.mean([max(0.0, r[k]['skill']) for k in SK])); out[name] = r     # a label read worse than guessing the median scores 0, not below
    out['health'] = float(np.mean([out[n]['objective'] for n, *_ in sets])); out['life'] = life_val(net)
    out['objective'] = float(0.5 * out['health'] + 0.5 * max(0.0, out['life']['skill'])); return out

def dev(net):
    """total-capacity error on the six calibration cells through life (real data, never in any exam)"""
    import realdata as RD, calibrate as CA
    cells, V = RD.load(); rows = RX.readings(cells, only=CA.CAL); st = lambda k: np.stack([r[k] for r in rows])
    P = RX.predict(net, st('Q1'), st('T1'), st('Q0'), st('T0')); return {'n': len(rows), 'q_total_mae_mAh': float(1000 * np.mean(np.abs(P['q_total'] - st('q_total'))))}

def dev_life(lnet):
    """lifetime error on the six calibration cells (real data, never in any exam): forecast at cycle 100 and every 50 cycles"""
    if 'log_life' not in getattr(lnet, 'labels', []): return None
    import realdata as RD, calibrate as CA, lifeexam as LE
    cells, V = RD.load(); rows = LE.readings(cells, cycles=(100,), every=50, only=CA.CAL); st = lambda k: np.stack([r[k] for r in rows])
    P = RX.predict(lnet, st('Q1'), st('T1'), st('Q0'), st('T0'), np.array([[r['cycle'], r['charge_time']] for r in rows]))
    life = np.array([r['life'] for r in rows]); err = np.abs(10 ** P['log_life'] / life - 1); c100 = np.array([r['cycle'] == 100 or abs(r['cycle'] - 100) <= 3 for r in rows])
    return {'n': len(rows), 'mpe_all': float(100 * err.mean()), 'mpe_c100': float(100 * err[c100].mean())}

def decide(rec, inc):
    if inc is None: rec['accepted'] = True; rec['why'] = 'first rung'; return
    sim_ok = rec['objective'] > inc['objective'] + 0.01; dev_ok = rec['dev']['q_total_mae_mAh'] <= inc['dev']['q_total_mae_mAh'] + 1.0
    dl, dli = rec.get('dev_life'), inc.get('dev_life'); life_ok = dl is None or dli is None or dl['mpe_all'] <= dli['mpe_all'] + 2.0
    rec['accepted'] = bool(sim_ok and dev_ok and life_ok); rec['vs'] = inc['ckpt'] + (f" + {inc['life_ckpt']}" if inc.get('life_ckpt') else '') + (f" + {inc['partial_ckpt']}" if inc.get('partial_ckpt') else '')
    rec['why'] = (f"score {rec['objective']:.3f} vs {inc['objective']:.3f}" + ('' if sim_ok else ' (needs +0.01)') +
                  ('' if dev_ok else f"; calibration cells {rec['dev']['q_total_mae_mAh']:.1f} vs {inc['dev']['q_total_mae_mAh']:.1f} mAh") +
                  ('' if life_ok else f"; calibration cells' lifetime {dl['mpe_all']:.0f}% vs {dli['mpe_all']:.0f}%"))

def main(ckpt, note, life_ckpt=None, partial_ckpt=None):
    t = time.time(); net = RX.load_net(ckpt); lnet = RX.load_net(life_ckpt) if life_ckpt else net; pnet = RX.load_net(partial_ckpt) if partial_ckpt else None
    v = evaluate(net, pnet); v['life'] = life_val(lnet); v['objective'] = float(0.5 * v['health'] + 0.5 * max(0.0, v['life']['skill']))
    real, rows, Pr, Pn = RX.exam(net); life = None
    if 'log_life' in lnet.labels:
        import lifeexam as LE; life = LE.exam(lnet)[0]
    log = json.load(open(LOG)) if os.path.exists(LOG) else []; inc = next((r for r in reversed(log) if r['accepted']), None)
    rec = {'ckpt': os.path.basename(ckpt), 'life_ckpt': os.path.basename(life_ckpt) if life_ckpt else None, 'partial_ckpt': os.path.basename(partial_ckpt) if partial_ckpt else None, 'note': note, 'val': v, 'objective': v['objective'], 'dev': dev(net), 'dev_life': dev_life(lnet),
           'real': real, 'partial': RX.exam_partial(pnet or net, rows), 'life': life, 'eval_s': round(time.time() - t)}
    decide(rec, inc)
    log.append(rec); json.dump(log, open(LOG, 'w'), indent=1)
    sk = lambda h: {k: round(v[h][k]['skill'], 3) for k in M.LABELS}
    print(f"{rec['ckpt']}: objective {v['objective']:.3f} = 0.5 x health {v['health']:.3f} (with ref {v['with_ref']['objective']:.3f}, curve only {v['no_ref']['objective']:.3f}) + 0.5 x lifetime {v['life']['skill']:.3f}")
    print('   skills with ref ', sk('with_ref')); print('   skills curve only', sk('no_ref'))
    print('  ', 'ACCEPTED' if rec['accepted'] else 'rejected', '-', rec['why'])

def rescore(full=True):
    """re-score every logged rung on the current frozen sets, dev cells and exams, and re-take every decision in order"""
    log = json.load(open(LOG))
    for rec in log:
        net = RX.load_net(_P.WORK + '/ckpt/' + rec['ckpt'])
        if full:
            lnet = RX.load_net(_P.WORK + '/ckpt/' + rec['life_ckpt']) if rec.get('life_ckpt') else net
            pnet = RX.load_net(_P.WORK + '/ckpt/' + rec['partial_ckpt']) if rec.get('partial_ckpt') else None
            v = evaluate(net, pnet); v['life'] = life_val(lnet); v['objective'] = float(0.5 * v['health'] + 0.5 * max(0.0, v['life']['skill'])); rec['val'] = v; rec['objective'] = v['objective']; real, rows, _, _ = RX.exam(net); rec['real'] = real; rec['partial'] = RX.exam_partial(pnet or net, rows)
        rec['dev'] = dev(net); rec['dev_life'] = dev_life(RX.load_net(_P.WORK + '/ckpt/' + rec['life_ckpt']) if rec.get('life_ckpt') else net)
        print('rescored', rec['ckpt'], rec.get('life_ckpt'), round(rec['objective'], 3), rec['dev'], rec['dev_life'], flush=True)
    inc = None
    for rec in log:
        decide(rec, inc)
        if rec['accepted']: inc = rec
    json.dump(log, open(LOG, 'w'), indent=1)

if __name__ == '__main__':
    if sys.argv[1] == '--rescore': rescore(full='--dev-only' not in sys.argv)
    else: main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != '-' else None, sys.argv[4] if len(sys.argv) > 4 else None)
