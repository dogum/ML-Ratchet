"""Everything the SunDoctor page shows, computed with the released checkpoint."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, base64, numpy as np, pandas as pd
sys.path.insert(0, _P.SRC)
import realexam as RX, evaluate as E, simbatch as SB, inject as IJ

C0, C1 = 20, 76                  # carpet columns: 05:00-19:00 solar time
def u8(a): return base64.b64encode(np.asarray(a, np.uint8).tobytes()).decode()
def f16(a): return base64.b64encode(np.asarray(a, '<f2').tobytes()).decode()
def r(x, d=4): return None if x is None or not np.isfinite(x) else round(float(x), d)

def inverter_run(net, P, G, T, geo, j):
    p = P[:, :, j].astype(float); valid = (np.isfinite(p) & np.isfinite(G)).mean(1) > 0.5
    o1 = RX.run_inverter(net, p, G, T, geo); ref = RX.reference(o1['pf'], valid & (o1['outage'] < 0.2))
    o = RX.run_inverter(net, p, G, T, geo, ref); o['ref'] = ref; o['valid'] = valid; return o

def case(name, desc, days, P, G, T, geo, fr, o, j, a, b, dc_kw=37.2):
    sl = slice(a, b); p = P[sl, :, j].astype(float); g = G[sl]; ta = T[sl]
    pe = SB.expected_power(np.nan_to_num(np.maximum(g, 0)), np.nan_to_num(ta, nan=15), geo['ca'][sl])
    car = np.nan_to_num(p[:, C0:C1] / 0.81, nan=0); ratio = np.where(pe[:, C0:C1] > 0.04, p[:, C0:C1] / np.maximum(pe[:, C0:C1], 1e-3), np.nan)
    with np.errstate(all='ignore'): usual = np.nanpercentile(ratio, 70, axis=0, keepdims=True)
    ratio = ratio / usual          # relative to this inverter's usual output-per-sunlight at that time of day (sensor quirks cancel)
    e = np.nansum(np.nan_to_num(p), 1) * dc_kw * 0.25
    out, clip, sh, so, fr_ = [np.clip(np.nan_to_num(o[k][sl]), 0, 0.99) for k in ('outage', 'clip', 'shade', 'soil', 'frac')]
    with np.errstate(all='ignore'):
        ec = e / (1 - out); en = ec / (1 - clip); eu = en / (1 - sh)
    loss = {'outage': ec - e, 'clipping': en - ec, 'shading': eu - en, 'soiling': eu * so / (1 - so), 'dead': eu / np.maximum(fr_, 0.05) * (1 - fr_)}
    k_true = np.round(6 * (1 - fr[sl, j])); clean = np.isfinite(fr[sl, j]) & (np.abs(6 * (1 - fr[sl, j]) - k_true) < 0.25)
    return {'name': name, 'desc': desc, 'inverter': j + 1, 'dates': [str(d.date()) for d in days[sl]], 'n': b - a,
            'carpet': u8(np.clip(car * 255, 0, 255)), 'ratio': u8(np.nan_to_num(np.clip(ratio / 1.25, 0, 1) * 255, nan=0)), 'ratio_ok': u8(np.isfinite(ratio) * 255),
            'k_pred': [r(6 * (1 - x), 3) for x in o['frac'][sl]], 'k_sig': [r(6 * x, 3) for x in o['frac_sig'][sl]], 'step_p': [r(x, 3) for x in o['step_p'][sl]],
            'k_true': [float(k) if c else None for k, c in zip(k_true, clean)], 'valid': [bool(v) for v in o['valid'][sl]],
            'kwh': [r(x, 1) for x in e], 'loss': {k: [r(x, 1) for x in v] for k, v in loss.items()},
            'pf': [r(x, 3) for x in o['pf'][sl]], 'ref': [r(x, 3) for x in o['ref'][sl]], 'shade': [r(x, 3) for x in o['shade'][sl]], 'clip': [r(x, 3) for x in o['clip'][sl]],
            'soil': [r(x, 3) for x in o['soil'][sl]], 'outage': [r(x, 3) for x in o['outage'][sl]]}

def lab_window(name, days, P, G, T, geo, o, j, a, D=14):
    sl = slice(a, a + D); p = np.nan_to_num(P[sl, :, j].astype(float)); g = G[sl]; ta = T[sl]
    pe = SB.expected_power(np.nan_to_num(np.maximum(g, 0)), np.nan_to_num(ta, nan=15), geo['ca'][sl]); pu = IJ.unclipped(p, pe)
    return {'name': name, 'dates': [str(d.date()) for d in days[sl]], 'p': f16(p), 'pu': f16(pu), 'g': f16(np.nan_to_num(g, nan=-1)), 'ta': f16(np.nan_to_num(ta, nan=15)),
            'cz': f16(geo['cz'][sl]), 'ca': f16(geo['ca'][sl]), 'gc': f16(geo['poa_clear'][sl] / 1000.0), 'el': f16(geo['el'][sl]), 'az': f16(geo['az'][sl]),
            'ref': r(np.nanmax(o['pf'][sl]), 4), 'base': {k: [r(x, 4) for x in o[k][sl]] for k in ('frac', 'shade', 'clip', 'soil', 'outage')}}

def main(ckpt, out=_P.LAB + '/data.json'):
    net = E.load(ckpt); days, P, G, T, fr, geo = RX.load(); di = {str(d.date()): i for i, d in enumerate(days)}
    runs = {}
    def run(j):
        if j not in runs: runs[j] = inverter_run(net, P, G, T, geo, j)
        return runs[j]
    ep = json.load(open(ckpt.replace('.pt', '_episodes.json')))
    s_eps = [e for e in ep['episodes'] if e['kind'] == 'string' and e['repair'] is not None and e['onset_found'] and e['inv'] != 10]
    off = next(e for e in ep['episodes'] if e['kind'] == 'offline' and e['inv'] == 0 and e['date'] == '2019-06-18')
    e2 = s_eps[len(s_eps) // 2]
    C = []
    hero = case('hero', '', days, P, G, T, geo, fr, run(10), 10, di['2019-09-17'], di['2019-11-01'])
    C.append(case('A string dies, then is fixed', 'Inverter 11 loses one of its six strings on 27 Sep 2019; a crew restores it on 16 Oct.', days, P, G, T, geo, fr, run(10), 10, di['2019-09-06'], di['2019-11-01']))
    C.append(case('Another string failure', f"Inverter {e2['inv'] + 1}: one string out from {e2['date']} until {e2['repair_date']}.", days, P, G, T, geo, fr, run(e2['inv']), e2['inv'], e2['day'] - 14, min(e2['repair'] + 16, len(days))))
    C.append(case('Inverter offline', 'Inverter 1 goes dark on 18 Jun 2019 and stays off for nine days.', days, P, G, T, geo, fr, run(0), 0, off['day'] - 12, off['day'] + 22))
    C.append(case('Winter row shading', 'A healthy inverter through the low winter sun. Rows of panels start shading the row behind.', days, P, G, T, geo, fr, run(10), 10, di['2019-11-15'], di['2020-01-15']))
    C.append(case('Spring clipping', 'Cool, bright spring days push the array past what the 30 kW inverter can deliver.', days, P, G, T, geo, fr, run(10), 10, di['2019-03-20'], di['2019-05-10']))
    lab = [lab_window('Spring', days, P, G, T, geo, run(10), 10, di['2019-04-08']), lab_window('Summer', days, P, G, T, geo, run(10), 10, di['2019-07-08']),
           lab_window('Winter', days, P, G, T, geo, run(10), 10, di['2019-12-04'])]
    real = json.load(open(ckpt.replace('.pt', '_real.json'))); inj = json.load(open(ckpt.replace('.pt', '_inject.json')))['summary']
    rd = json.load(open(ckpt.replace('.pt', '_rdtools.json'))); ser = np.load(ckpt.replace('.pt', '_inv11_series.npz'))
    sd = pd.to_datetime(ser['days'], unit='D'); df = pd.DataFrame({k: ser[k] for k in ('pi', 'pf', 'shade', 'clip', 'soil', 'frac')}, index=sd)[ser['healthy']]
    mon = df.groupby(df.index.month).mean()
    season = {'month': list(range(1, 13)), **{k: [r(x) for x in mon[k]] for k in mon.columns}, 'rdtools_soil': [r(rd['soiling']['monthly'][str(m)][0]) if 'monthly' in rd['soiling'] else None for m in range(1, 13)]}
    R = json.load(open(_P.WORK + '/ratchet.json'))
    rungs = [{'ckpt': x['ckpt'], 'note': x['note'], 'accepted': x['accepted'], 'why': x['why'], 'objective': x['objective'],
              'skill': {k: x['val'][k]['skill'] for k in ('pf', 'frac', 'soil', 'shade', 'clip', 'outage')}, 'step_auc': x['val']['step_auc'], 'count_acc': x['val']['count_acc'],
              'count_acc_dead': x['val']['count_acc_dead'], 'real': x['real']} for x in R]
    import torch
    samples = 0
    for x in R:
        ck = torch.load(_P.WORK + '/ckpt/' + x['ckpt'], weights_only=False); samples += ck['args']['steps'] * ck['args']['bs']
    data = {'cases': C, 'hero': hero, 'lab': lab, 'carpet_cols': [C0, C1], 'exam': {'real': real, 'inject': inj, 'episodes': ep['summary'], 'rdtools': {k: v for k, v in rd.items() if k != 'soiling'},
            'soiling': {k: v for k, v in rd['soiling'].items() if k != 'monthly'}, 'season': season}, 'rungs': rungs,
            'physics': json.load(open(_P.WORK + '/physics_check.json')), 'release': {'ckpt': ckpt.split('/')[-1], 'samples': samples, 'val': E.evaluate(net)},
            'clip_pu': 0.806, 'dc_kw': 37.2}
    json.dump(data, open(out, 'w')); print('web data MB %.2f' % (len(json.dumps(data)) / 1e6))

if __name__ == '__main__':
    main(sys.argv[1])
