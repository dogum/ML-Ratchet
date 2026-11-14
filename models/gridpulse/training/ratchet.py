"""GridPulse ratchet. Objective (frozen simulated validation set): mean skill over MW lost, inertia and MW/Hz at 60 s,
and nadir and 60 s settling forecast at 10 s (skill = 1 - MAE / MAE of guessing the median, floored at 0).
A rung is kept only if the objective rises by more than 0.01 AND on the 4 GB dev events the mean of its four errors
(loss MdAPE, inertia MdAPE, nadir@10 s MAE, settling@10 s MAE), each relative to the incumbent's, is no more than 10 %
worse. Physics/measurement changes get a new frozen set; then every rung is re-scored on it in order.
Usage: python3 ratchet.py add <rung> <valname> "<note>"   |   python3 ratchet.py rescore <valname>"""
from common import *
import evaluate as EV
LOG = f'{ROOT}/ratchet.json'
KEYS = ('loss_mdape', 'E_mdape', 'nadir10_mae', 'settle10_mae')


def decide(rec, inc):
    if inc is None:
        rec['accepted'] = True; rec['why'] = 'first rung'; return
    s_ok = rec['objective'] > inc['objective'] + 0.01
    rel = float(np.mean([rec['dev'][k] / max(inc['dev'][k], 1e-9) for k in KEYS])); d_ok = rel <= 1.10
    rec['dev_rel'] = rel; rec['accepted'] = bool(s_ok and d_ok); rec['vs'] = inc['rung']
    rec['why'] = (f"score {rec['objective']:.3f} vs {inc['objective']:.3f}" + ('' if s_ok else ' (needs +0.01)') +
                  f"; dev errors {100 * (rel - 1):+.0f}% vs the incumbent" + ('' if d_ok else ' (gate: <= +10%)'))


def score(rung, valname):
    v = EV.val_scores(rung, valname); d, tab = EV.dev_gb(rung)
    return v, d, tab


def add(rung, valname, note):
    log = json.load(open(LOG)) if os.path.exists(LOG) else []
    inc = next((r for r in reversed(log) if r['accepted']), None)
    v, d, tab = score(rung, valname)
    rec = dict(rung=rung, note=note, val=valname, objective=v['objective'], skills=skills(v), dev=d,
               dev_table=tab.round(4).to_dict('records'))
    decide(rec, inc); log.append(rec); json.dump(log, open(LOG, 'w'), indent=1)
    show(rec)


def skills(v):
    b = v['by_N']
    return {'loss@60': b[60]['log_loss']['skill'], 'E@60': b[60]['log_E']['skill'], 'beta@60': b[60]['log_beta']['skill'],
            'head@60': b[60]['log_head']['skill'], 'ffr@60': b[60]['log_ffr']['skill'],
            'nadir@5': b[5]['nadir']['skill'], 'nadir@10': b[10]['nadir']['skill'], 'settle@10': b[10]['settle']['skill'],
            'loss@10': b[10]['log_loss']['skill'], 'E@10': b[10]['log_E']['skill'],
            'GB_E@60': b[60]['GB']['log_E'], 'GB_loss@60': b[60]['GB']['log_loss'], 'cov_loss@60': b[60]['log_loss']['cov1'],
            'cov_E@60': b[60]['log_E']['cov1']}


def show(rec):
    print(f"{rec['rung']}: objective {rec['objective']:.3f} | " + ' '.join(f'{k} {x:.2f}' for k, x in rec['skills'].items()))
    print('  dev', {k: round(x, 3) for k, x in rec['dev'].items()}, '->', 'ACCEPTED' if rec['accepted'] else 'rejected', '-', rec['why'])


def rescore(valname):
    log = json.load(open(LOG)); inc = None
    for rec in log:
        v, d, tab = score(rec['rung'], valname)
        rec.update(val=valname, objective=v['objective'], skills=skills(v), dev=d, dev_table=tab.round(4).to_dict('records'))
        decide(rec, inc); inc = rec if rec['accepted'] else inc; show(rec)
    json.dump(log, open(LOG, 'w'), indent=1)


if __name__ == '__main__':
    a = sys.argv
    if a[1] == 'add':
        add(a[2], a[3], a[4])
    else:
        rescore(a[2])
