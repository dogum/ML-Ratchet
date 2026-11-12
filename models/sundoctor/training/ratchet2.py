"""SunDoctor v2 ratchet. Objective on the frozen v2 set (2,000 simulated fortnights: half with an irradiance sensor, half
without; fixed and tracking systems; snow and stuck trackers): the mean over the two halves of [mean skill over pf, frac,
soiling, shading, clipping, outage, snow, stuck + 0.5 x step AUC + 0.5 x dead-string counting accuracy]. Kept only if it
rises by more than 0.01. Real exams on systems 2107 and 9068 are reported, never used."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, _P.SRC)
import evaluate2 as E2, evaluate as E1, realexam2 as RX2
LOG = _P.WORK + '/ratchet2.json'
OLD = [0, 1, 3, 4, 6, 7, 8, 9, 10, 11]

class V1Adapter(nn.Module):
    """the released v1 network seen through v2's inputs and outputs (no snow / stuck heads)"""
    def __init__(self, v1):
        super().__init__(); self.v1 = v1; self.days = v1.days
    def forward(self, x):
        d, g = self.v1(x[:, OLD]); B, _, D = d.shape; o = torch.zeros(B, 19, D)
        o[:, 0:7] = d[:, 0:7]; o[:, 9:16] = d[:, 7:14]; o[:, 18] = d[:, 14]; o[:, 16:18] = 0.0
        return o, g

def summarize(net):
    out = {}
    for home in (False, True):
        m = 'home' if home else 'poa'
        r, _ = RX2.exam_2107(net, home); out['2107_' + m] = r
        s, _ = RX2.exam_9068(net, home); out['9068_' + m] = s
    return out

def main(ckpt, note):
    t = time.time()
    net = V1Adapter(E1.load(ckpt)).eval() if ckpt.endswith('s3_polish.pt') else E2.load(ckpt)
    v = E2.evaluate(net); real = summarize(net)
    log = json.load(open(LOG)) if os.path.exists(LOG) else []
    inc = next((r for r in reversed(log) if r['accepted']), None)
    rec = {'ckpt': os.path.basename(ckpt), 'note': note, 'val': v, 'objective': v['objective'], 'real': real, 'eval_s': round(time.time() - t)}
    if inc is None: rec['accepted'] = True; rec['why'] = 'starting point (v1 release)'
    else: rec['accepted'] = bool(v['objective'] > inc['objective'] + 0.01); rec['why'] = f"objective {v['objective']:.3f} vs {inc['objective']:.3f}"; rec['vs'] = inc['ckpt']
    log.append(rec); json.dump(log, open(LOG, 'w'), indent=1)
    sk = lambda h: {k: round(v[h][k]['skill'], 2) for k in E2.SKILLS}
    print(f"{rec['ckpt']}: objective {v['objective']:.3f}  (with sensor {v['poa']['objective']:.3f}, home {v['home']['objective']:.3f}, trackers {v['trackers']['objective']:.3f})")
    print('   skills with sensor', sk('poa')); print('   skills home       ', sk('home'))
    for k in ('2107_poa', '2107_home'):
        r = real[k]; print(f"   {k}: count {r['count_acc']:.3f} when dead {r['count_acc_when_dead']:.3f} healthy {r['healthy_called_healthy']:.3f} onsets {r['onsets_found_pm1day']}/{r['onsets_true']} alarms {r['step_alarms']}")
    for k in ('9068_poa', '9068_home'):
        r = real[k]; print(f"   {k}:", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()})
    print('  ', 'ACCEPTED' if rec['accepted'] else 'rejected', '-', rec['why'])

def rescore():
    """recompute the real-data exams for every logged rung (after an answer-key fix); selection is untouched"""
    log = json.load(open(LOG))
    for rec in log:
        ck = _P.WORK + '/ckpt/' + rec['ckpt']; net = V1Adapter(E1.load(ck)).eval() if ck.endswith('s3_polish.pt') else E2.load(ck)
        rec['real'] = summarize(net); print('rescored', rec['ckpt'], flush=True)
    json.dump(log, open(LOG, 'w'), indent=1)

if __name__ == '__main__':
    rescore() if sys.argv[1] == '--rescore' else main(sys.argv[1], sys.argv[2])
