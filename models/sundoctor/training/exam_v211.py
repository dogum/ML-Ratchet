"""Dated supplementary exam (27 Nov 2026; audit queue): the released v2.1 weights re-scored on the same rebuilt PVDAQ data
as exam_v21.py, once with the exam's half-day rule (as published) and once with min_daylight=0.9 (days missing more than
10% of their 08:00-16:00 solar-time readings are dropped, as sundoctor.diagnose(min_daylight=0.9) does). The network is
unchanged; only which days count. Nothing here feeds any choice. Writes work/exam_v211_min_daylight.json."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.MODEL)
import exam_v21 as X, realexam as RX, realexam2 as RX2, real9068 as R9, sundoctor

def gappy_days():
    days, P, G, T, fr, geo = RX.load(); d9, d, g9 = R9.load(); out = {}
    for name, p in (('2107', P), ('9068', d['P'])):
        half = np.isfinite(p).mean(1) > 0.5; dl = np.isfinite(p[:, 32:64]).mean(1)
        out[name] = dict(inverter_days_counted=int(half.sum()), dropped_by_min_daylight_0_9=int((half & (dl < 0.9)).sum()), warned_lt_0_75=int((half & (dl < 0.75)).sum()))
    return out

if __name__ == '__main__':
    net = sundoctor.SunDoctorNet(os.path.join(_P.MODEL, 'sundoctor_weights.npz')); out = dict(run_utc=__import__('time').strftime('%Y-%m-%d %H:%M UTC', __import__('time').gmtime()), days=gappy_days())
    print(json.dumps(out['days']), flush=True)
    for tag, md in (('half_day_rule', None), ('min_daylight_0_9', 0.9)):
        RX2.MIN_DAYLIGHT = md; r = {}
        for home in (False, True):
            m = 'home' if home else 'poa'
            r['2107_' + m] = RX2.exam_2107(net, home)[0]; r['episodes_' + m] = X.episodes(net, home); r['9068_' + m] = RX2.exam_9068(net, home)[0]
            print(tag, m, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in {**r['episodes_' + m], **{a: b for a, b in r['9068_' + m].items() if a in ('stuck_auc_2pct', 'stuck_found_big', 'snow_auc', 'outage_auc', 'outage_found')}, 'count_acc': r['2107_' + m]['count_acc']}.items()}), flush=True)
        out[tag] = r
    json.dump(out, open(_P.WORK + '/exam_v211_min_daylight.json', 'w'), indent=1, default=str)
