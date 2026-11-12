"""SunDoctor ratchet: a candidate replaces the incumbent only if it beats it on the frozen simulated validation set.
Objective = mean skill over pf, soiling, fraction online, shading, clipping and outage (1 - MAE / MAE of predicting the
median) + 0.5 x step-detection AUC + 0.5 x dead-string counting accuracy. The real PVDAQ exam is reported, never used."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, time
sys.path.insert(0, _P.SRC)
import evaluate as E, realexam as RX
LOG = _P.WORK + '/ratchet.json'

def main(ckpt, note):
    net = E.load(ckpt); t = time.time(); v = E.evaluate(net)
    days, fr, rows = RX.exam(net); real = RX.score(days, fr, rows)
    log = json.load(open(LOG)) if os.path.exists(LOG) else []
    inc = next((r for r in reversed(log) if r['accepted']), None)
    rec = {'ckpt': os.path.basename(ckpt), 'note': note, 'val': v, 'objective': v['objective'], 'real': real, 'eval_s': round(time.time() - t)}
    if inc is None: rec['accepted'] = True; rec['why'] = 'first rung'
    else:
        rec['accepted'] = bool(v['objective'] > inc['objective'] + 0.01); rec['why'] = f"objective {v['objective']:.3f} vs {inc['objective']:.3f}"; rec['vs'] = inc['ckpt']
    log.append(rec); json.dump(log, open(LOG, 'w'), indent=1)
    s = lambda k: f"{v[k]['skill']:.2f}"
    print(f"{rec['ckpt']}: objective {v['objective']:.3f} | skill pf {s('pf')} frac {s('frac')} soil {s('soil')} shade {s('shade')} clip {s('clip')} outage {s('outage')} | step AUC {v['step_auc']:.3f} count {v['count_acc']:.3f} (dead {v['count_acc_dead']:.3f})")
    print(f"   real 2107: {real['n_inverter_days']} inverter-days, count acc {real['count_acc']:.3f}, when dead {real['count_acc_when_dead']}, healthy->healthy {real['healthy_called_healthy']:.3f}, frac MAE {real['frac_mae']:.3f}, onsets {real['onsets_found_pm1day']}/{real['onsets_true']} (alarms {real['step_alarms']})")
    print('   confusion (rows true 0/1/2/3+ dead, cols predicted):', real['confusion_true_rows_0_1_2_3plus'])
    print('  ', 'ACCEPTED' if rec['accepted'] else 'rejected', '-', rec['why'])

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
