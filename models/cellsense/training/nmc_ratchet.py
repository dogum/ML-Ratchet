"""CellSense NMC811 ratchet (ladder "v2 — NMC811 (LG M50T)").

Objective, on a frozen simulated set (val_<physics>.npz: 3,000 new/aged pairs, seed 999; one fixed measurement-noise draw,
seed 12345): the mean skill (1 - MAE / MAE of guessing the set's median, floored at 0) over capacity, lithium lost,
negative lost and positive lost (and fade when the reference is given), averaged over four readings of every pair:
full curve with its reference, full curve alone, partial discharge (stopped after 1.0 Ah to the end, fixed draw) with
the reference, and partial alone.

Dev gate, on the three dev cells (W3, W4, W8; real; never in the exam): capacity error on partial discharges stopped after
1.5, 2.5 and 3.5 Ah (with and without the reference) may be no more than 5% worse than the incumbent's, and the error on
full curves no more than 2 mAh worse. Kept only if the objective beats the incumbent by more than 0.01 AND the gate
passes (mlratchet.Ratchet). The exam cells are scored and logged for the record, never used to choose.

    python nmc_ratchet.py add <id> "<change>" <ckpt>      evaluate a rung on the current frozen set and judge it
    python nmc_ratchet.py rescore <physics>               new frozen set: re-score every rung, re-take every decision
    python nmc_ratchet.py refresh                         real readings changed: recompute dev and exam, re-take decisions
    python nmc_ratchet.py show
"""
import _paths as _P
import os, sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC); sys.path.insert(0, os.path.dirname(os.path.dirname(_P.MODEL)))
import nmc_model as NM, nmc_train as TR, nmc_data as D
from mlratchet import Ratchet, gates
LOG = os.path.join(_P.WORK, 'nmc', 'ratchet_nmc.json'); DEPTHS = (1.5, 2.5, 3.5)
SK = ['q_total', 'lli', 'lam_ne', 'lam_pe']


END_SIG = 0.002      # Ah: a full curve's capacity is read off its end (the cycler's own count); bar for grid interpolation


def load_net(path):
    """path may end in '+endrule': full curves (reaching 2.5 V) then take their capacity from the curve's end and their
    fade from the two ends, and only partial curves use the network's capacity"""
    rule = path.endswith('+endrule'); path = path[:-len('+endrule')] if rule else path
    ck = torch.load(path, weights_only=False); net = NM.Net(tuple(int(w) for w in ck['widths'].split(',')), n_out=len(ck['labels']))
    net.load_state_dict(ck['state']); net.eval(); net.mu_y, net.sd_y, net.labels = np.array(ck['mu_y']), np.array(ck['sd_y']), list(ck['labels']); net.smooth = ck.get('smooth', 0); net.target = ck.get('target', 'total'); net.endrule = rule; return net


def predict(net, Q1, Q0=None, bs=1024):
    out = []
    with torch.no_grad():
        for i in range(0, len(Q1), bs): out.append(net(torch.from_numpy(NM.features(Q1[i:i + bs], None if Q0 is None else Q0[i:i + bs], smooth=getattr(net, 'smooth', 0)))).numpy())
    o = np.concatenate(out); n = len(net.labels); mu = o[:, :n] * net.sd_y + net.mu_y; sg = np.exp(np.clip(o[:, n:], -7, 3)) * net.sd_y
    if getattr(net, 'target', 'total') == 'rest': mu[:, 0] += NM.q_last(Q1)
    if getattr(net, 'endrule', False):
        full = np.isfinite(Q1[:, -1]); mu[full, 0] = Q1[full, -1]; sg[full, 0] = END_SIG
        if Q0 is not None:
            ok = full & np.isfinite(Q0[:, -1]); j = net.labels.index('fade'); mu[ok, j] = 1 - Q1[ok, -1] / Q0[ok, -1]; sg[ok, j] = END_SIG / 4.8
    return {k: mu[:, j] for j, k in enumerate(net.labels)} | {k + '_sig': sg[:, j] for j, k in enumerate(net.labels)}


def val_sets(physics):
    z = np.load(os.path.join(_P.WORK, 'nmc', f'val_{physics}.npz')); rng = np.random.default_rng(12345)
    Q1, Q0 = TR.augment(rng, z['Q1'].astype(np.float64), z['Q0'].astype(np.float64), p_ref=1.0); lab = [str(x) for x in z['labels']]
    Y = {k: z['y'][:, lab.index(k)] for k in NM.LABELS}; D_ = rng.uniform(1.0, np.nanmax(Q1, 1)); Qp = TR.truncate(Q1, D_)
    return [('full_ref', Q1, Q0), ('full', Q1, None), ('partial_ref', Qp, Q0), ('partial', Qp, None)], Y


def objective(net, physics):
    sets, Y = val_sets(physics); out = {}
    for name, Q1, Q0 in sets:
        P = predict(net, Q1, Q0); s = {}
        for k in SK + (['fade'] if Q0 is not None else []):
            e = np.abs(P[k] - Y[k]); s[k] = float(1 - e.mean() / np.abs(Y[k] - np.median(Y[k])).mean())
        out[name] = dict(skills=s, objective=float(np.mean([max(0.0, v) for v in s.values()])))
    out['objective'] = float(np.mean([out[n]['objective'] for n, *_ in sets])); return out


def real_readings(cells, depths=DEPTHS):
    """every capacity test of these cells, read in full and cut after each depth, with the cell's first test as the
    reference (tests 2 onward) and without it (all tests)"""
    R = D.readings(cells); out = []
    for r in R:
        for d in (None,) + tuple(depths):
            q = r['q'].copy()
            if d is not None: q[q > d] = np.nan
            for ref in (True, False):
                if ref and r['diag'] == 1: continue
                out.append(dict(cell=r['cell'], diag=r['diag'], depth=d, ref=ref, Q1=q, Q0=r['q_ref'] if ref else None, q_total=r['q_total'], q_ref=r['q_total_ref']))
    return out


def run_real(net, rows):
    P = {}
    for ref in (True, False):
        idx = [i for i, r in enumerate(rows) if r['ref'] == ref]
        if not idx: continue
        Q1 = np.stack([rows[i]['Q1'] for i in idx]); Q0 = np.stack([rows[i]['Q0'] for i in idx]) if ref else None; p = predict(net, Q1, Q0)
        for k, v in p.items(): P.setdefault(k, np.full(len(rows), np.nan))[idx] = v
    return P


def real_scores(net, cells):
    rows = real_readings(cells); P = run_real(net, rows); qt = np.array([r['q_total'] for r in rows]); e = 1000 * np.abs(P['q_total'] - qt)
    part = np.array([r['depth'] is not None for r in rows]); out = {'q_partial_mAh': float(e[part].mean()), 'q_full_mAh': float(e[~part].mean())}
    for d in DEPTHS:
        for ref in (True, False):
            m = np.array([r['depth'] == d and r['ref'] == ref for r in rows]); out[f'q_{d}_{"ref" if ref else "noref"}_mAh'] = float(e[m].mean())
    return out


def gate(cand, inc):
    ok1 = cand['q_partial_mAh'] <= inc['q_partial_mAh'] * 1.05; ok2 = cand['q_full_mAh'] <= inc['q_full_mAh'] + 2.0
    why = f"dev partial {cand['q_partial_mAh']:.1f} vs {inc['q_partial_mAh']:.1f} mAh" + ('' if ok1 else ' (more than 5% worse)') + \
          f", dev full {cand['q_full_mAh']:.1f} vs {inc['q_full_mAh']:.1f} mAh" + ('' if ok2 else ' (more than 2 mAh worse)')
    return ok1 and ok2, why


def ladder(log):
    r = Ratchet('cellsense', margin=0.01, dev_gate=gate, frozen_set=log['scored_on'], name='v2 — NMC811 (LG M50T)')
    for g in log['rungs']: r.judge(g['id'], g['change'], g['sim'][log['scored_on']], g['dev'])
    return r


def save(log):
    r = ladder(log)
    for g, rr in zip(log['rungs'], r.rungs): g['decision'], g['why'] = rr.decision, rr.why
    json.dump(log, open(LOG, 'w'), indent=1); return r


if __name__ == '__main__':
    log = json.load(open(LOG)) if os.path.exists(LOG) else {'scored_on': None, 'frozen_sets': [], 'rungs': []}
    if sys.argv[1] == 'add':
        rid, change, ck = sys.argv[2], sys.argv[3], sys.argv[4]; physics = log['scored_on'] or (sys.argv[5] if len(sys.argv) > 5 else 'textbook')
        if log['scored_on'] is None: log['scored_on'] = physics; log['frozen_sets'] = [physics]
        net = load_net(os.path.join(_P.WORK, 'nmc', 'ckpt', ck)); v = objective(net, physics)
        rec = dict(id=rid, change=change, ckpt=ck, sim={physics: v['objective']}, val={physics: v}, dev=real_scores(net, D.DEV), exam=real_scores(net, D.EXAM))
        log['rungs'] = [g for g in log['rungs'] if g['id'] != rid] + [rec]; r = save(log); g = r.rungs[-1]
        print(rid, 'objective %.3f' % v['objective'], {n: round(v[n]['objective'], 3) for n in ('full_ref', 'full', 'partial_ref', 'partial')})
        print('   dev', {k: round(x, 1) for k, x in rec['dev'].items()}); print('   exam (never used)', {k: round(x, 1) for k, x in rec['exam'].items()})
        print('  ', g.decision.upper(), '-', g.why)
    elif sys.argv[1] == 'rescore':
        physics = sys.argv[2]; old = {g['id']: g.get('decision') for g in log['rungs']}
        for g in log['rungs']:
            net = load_net(os.path.join(_P.WORK, 'nmc', 'ckpt', g['ckpt'])); v = objective(net, physics); g['sim'][physics] = v['objective']; g['val'][physics] = v
            print('rescored', g['id'], physics, round(v['objective'], 3), flush=True)
        log['scored_on'] = physics; log['frozen_sets'] = log['frozen_sets'] + ([physics] if physics not in log['frozen_sets'] else []); r = save(log)
        for g in log['rungs']: print(g['id'], old[g['id']], '->', g['decision'], '|', g['why'])
    elif sys.argv[1] == 'refresh':          # the real readings changed (a data-handling fix): recompute dev and exam for every rung, re-take every decision
        old = {g['id']: g.get('decision') for g in log['rungs']}
        for g in log['rungs']:
            net = load_net(os.path.join(_P.WORK, 'nmc', 'ckpt', g['ckpt'])); g['dev'] = real_scores(net, D.DEV); g['exam'] = real_scores(net, D.EXAM)
        r = save(log)
        for g in log['rungs']: print(g['id'], old[g['id']], '->', g['decision'], '| dev %.1f / %.1f' % (g['dev']['q_partial_mAh'], g['dev']['q_full_mAh']), '|', g['why'])
    elif sys.argv[1] == 'show':
        r = save(log)
        for g in log['rungs']: print(g['id'], g['decision'], {k: round(v, 3) for k, v in g['sim'].items()}, 'dev %.1f/%.1f' % (g['dev']['q_partial_mAh'], g['dev']['q_full_mAh']), '|', g['why'])
