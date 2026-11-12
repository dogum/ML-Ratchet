"""Ratchet for the neutron-star model. Objective = Gaussian sensitivity (SNR 6-12, FAP 1e-3) - 2 x median chirp-mass error;
glitch false alarms must stay at or below 5%. GW170817 is the final exam and is never used to decide."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, os
LOG = _P.WORK + '/bns_rungs.json'

def objective(v): return v['sens'] - 2 * v['mc_relerr_med']

def main(name, note):
    v = json.load(open(f'{_P.WORK}/ckpt/{name}_eval.json'))
    real = json.load(open(f'{_P.WORK}/ckpt/{name}_real.json'))
    log = json.load(open(LOG)) if os.path.exists(LOG) else []
    inc = next((r for r in reversed(log) if r['accepted']), None)
    rec = {'name': name, 'ckpt': f'{_P.WORK}/ckpt/{name}.pt', 'note': note, 'val': v, 'objective': objective(v),
           'real': {'score': real['peak']['score'], 'mc': real['mc_combined'], 'bg_max': real['bg_max'], 'n_bg_above': real['n_bg_above_event'],
                    'bg_live_s': real['bg_live_s'], 'raw_nogate': real['raw_nogate']['score'], 'single': real['single']}}
    if inc is None: rec['accepted'] = True; rec['why'] = 'first rung'
    else:
        rec['accepted'] = bool(rec['objective'] > inc['objective'] + 0.01 and v['glitch_fa'] <= 0.05)
        rec['why'] = f"objective {rec['objective']:.3f} vs {inc['objective']:.3f}; glitch false alarms {v['glitch_fa']:.3f}"
    log.append(rec); json.dump(log, open(LOG, 'w'), indent=1)
    print(name, 'ACCEPTED' if rec['accepted'] else 'rejected', rec['why'])
    print('  eff', v['eff'], 'mc_err %.3f' % v['mc_relerr_med'], 't_err %.0f ms' % v['t_abserr_ms'])
    print('  GW170817: score %.2f  Mc %.4f  (LIGO 1.1975)  background max %.2f over %.1f h, %d windows louder; raw ungated %.2f; alone %s' % (
        rec['real']['score'], rec['real']['mc'], rec['real']['bg_max'], rec['real']['bg_live_s'] / 3600, rec['real']['n_bg_above'], rec['real']['raw_nogate'], rec['real']['single']))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
