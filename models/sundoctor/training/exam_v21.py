"""The real exams for v2.1, run once after the release choice (15 Nov 2026), with v2.0 re-scored on the same rebuilt data.
numpy weights (the torch checkpoints of v2.0 were not kept): work/rungs/s8_release.npz (v2.0) and ../sundoctor_weights.npz
(v2.1). Same scoring as realexam2.py (2107 counts, 9068 snow / stuck / outage) and episodes2.py (string failures and
inverter-offline episodes), with and without the irradiance sensor. Nothing here feeds any choice. Writes
work/exam_v21.json."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.MODEL)
import evaluate2 as E2, realexam as RX, realexam2 as RX2, episodes2 as EP, sundoctor

E2.predict = lambda net, X, bs=100: net.predict(np.asarray(X, np.float32))     # numpy network in place of torch

def episodes(net, home):
    days, P, G, T, fr, geo = RX.load(); eps = EP.find(fr, days); cache = {}
    for e in eps:
        j = e['inv']
        if j not in cache: cache[j] = RX2.diagnose(net, P[:, :, j].astype(float), G, T, geo, home)
        o = cache[j]; d = e['day']
        if e['kind'] == 'string':
            e['onset_found'] = bool(np.nanmax(o['step_p'][d - 1:d + 2]) > 0.5)
            if e['repair'] is not None: e['repair_found'] = bool(np.nanmax(o['step_p'][e['repair'] - 1:e['repair'] + 2]) > 0.5)
        else:
            e['outage_found'] = bool(np.nanmean(o['outage'][d:d + 3]) > 0.5)
        e['date'] = str(days[d].date())
    s = [e for e in eps if e['kind'] == 'string']; off = [e for e in eps if e['kind'] == 'offline']
    return {'string_failures': len(s), 'onsets_found_pm1day': sum(e['onset_found'] for e in s), 'repairs': sum(e['repair'] is not None for e in s),
            'repairs_found': sum(e.get('repair_found', False) for e in s), 'offline_events': len(off), 'offline_found': sum(e['outage_found'] for e in off),
            'offline_missed_dates': [(e['inv'], e['date']) for e in off if not e['outage_found']]}

if __name__ == '__main__':
    out = {}
    for name, path in (('v2.0', _P.WORK + '/rungs/s8_release.npz'), ('v2.1', os.path.join(_P.MODEL, 'sundoctor_weights.npz'))):
        net = sundoctor.SunDoctorNet(path); r = {}
        for home in (False, True):
            m = 'home' if home else 'poa'
            r['2107_' + m] = RX2.exam_2107(net, home)[0]; r['episodes_' + m] = episodes(net, home); r['9068_' + m] = RX2.exam_9068(net, home)[0]
            print(name, m, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in {**r['episodes_' + m], **{a: b for a, b in r['9068_' + m].items() if a in ('stuck_auc_2pct', 'stuck_found_big', 'snow_auc', 'outage_auc', 'outage_found')}, 'count_acc': r['2107_' + m]['count_acc']}.items()}), flush=True)
        out[name] = r
    json.dump(out, open(_P.WORK + '/exam_v21.json', 'w'), indent=1, default=str)
