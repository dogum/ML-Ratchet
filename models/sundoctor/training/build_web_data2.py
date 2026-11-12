"""Everything the SunDoctor v2 page shows, computed with the released v2 checkpoint: California (2107) cases in both
modes, Colorado (9068) tracker and snow cases, the lab windows, every exam, and both ratchet ladders."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, numpy as np, pandas as pd, torch
sys.path.insert(0, _P.SRC)
import realexam as RX, realexam2 as RX2, evaluate2 as E2, simbatch2 as SB2, inject as IJ, real9068 as R9
from build_web_data import u8, f16, r, C0, C1
CK = _P.WORK + '/ckpt/'

def L(a, d=3): return [r(x, d) for x in a]

def chain(e, o, sl, etop=None):
    """split each day's shortfall into causes, undoing them in the simulator's order. On near-total-loss days (snow, an
    outage) undoing from zero gives zero, so those days are sized from the top instead: etop, the healthy array's energy."""
    out, clip, snow, sh, st, so, fr_ = [np.clip(np.nan_to_num(o[k][sl]), 0, 0.99) for k in ('outage', 'clip', 'snow', 'shade', 'stuck', 'soil', 'frac')]
    with np.errstate(all='ignore'):
        ec = e / (1 - out); en = ec / (1 - clip); es = en / (1 - snow); eu = es / (1 - sh); et = eu / (1 - st); e0 = et / (1 - so)
        L_ = {'outage': ec - e, 'clipping': en - ec, 'snow': es - en, 'shading': eu - es, 'stuck': et - eu, 'soiling': e0 - et, 'dead': e0 / np.maximum(fr_, 0.05) * (1 - fr_)}
        if etop is not None:
            F = (1 - out) * (1 - clip) * (1 - snow) * (1 - sh) * (1 - st); top = (F < 0.3) & np.isfinite(etop) & (etop > e)
            x0 = etop; x1 = x0 * fr_; x2 = x1 * (1 - so); x3 = x2 * (1 - st); x4 = x3 * (1 - sh); x5 = x4 * (1 - snow); x6 = x5 * (1 - clip)
            T_ = {'dead': x0 - x1, 'soiling': x1 - x2, 'stuck': x2 - x3, 'shading': x3 - x4, 'snow': x4 - x5, 'clipping': x5 - x6, 'outage': np.maximum(x6 - e, 0)}
            L_ = {k: np.where(top, T_[k], v) for k, v in L_.items()}
    return L_

def top_energy(o, sl, p, pe, pc, dc_kw, home):
    """the healthy array's energy each day: its own healthy level times the physics expectation (measured sunlight with a
    sensor; without one, clear-sky sunlight times the window's typical clearness, judged from power alone)"""
    ref = np.where(np.isfinite(o['ref'][sl]), o['ref'][sl], np.nanmedian(o['pf'][sl])); e = np.nansum(np.nan_to_num(p), 1) * dc_kw * 0.25
    if not home:
        ok = np.isfinite(pe).mean(1) > 0.5; return np.where(ok, ref * np.nansum(pe, 1) * dc_kw * 0.25, np.nan)
    ecs = ref * np.sum(pc, 1) * dc_kw * 0.25
    F = np.prod([1 - np.clip(np.nan_to_num(o[k][sl]), 0, 0.99) for k in ('outage', 'snow', 'stuck')], 0)
    good = (F > 0.7) & (ecs > 0); kt = np.nanmedian(e[good] / ecs[good]) if good.sum() >= 3 else 0.6
    return ecs * kt

def mode_out(o, sl, e, etop=None, n_str=6):
    return {'k_pred': L(n_str * (1 - o['frac'][sl])), 'k_sig': L(n_str * o['frac_sig'][sl]), 'step_p': L(o['step_p'][sl]), 'valid': [bool(v) for v in o['valid'][sl]],
            'loss': {k: L(v, 1) for k, v in chain(e, o, sl, etop).items()}, 'pf': L(o['pf'][sl]), 'ref': L(o['ref'][sl]),
            **{k: L(o[k][sl]) for k in ('shade', 'clip', 'soil', 'outage', 'snow', 'stuck', 'snow_sig', 'stuck_sig')}}

def norm_ratio(ratio, q=70):
    with np.errstate(all='ignore'): usual = np.nanpercentile(ratio, q, axis=0, keepdims=True)
    return ratio / usual

# ------------------------------------------------------------------ California, 2107
def ca_case(name, desc, days, P, G, T, geo, fr, oP, oH, j, a, b, dc_kw=37.2):
    sl = slice(a, b); p = P[sl, :, j].astype(float); g = G[sl]; ta = np.nan_to_num(T[sl], nan=15)
    pe = SB2.expected_power(np.nan_to_num(np.maximum(g, 0)), ta, geo['ca'][sl]); pc = SB2.expected_power(geo['poa_clear'][sl] / 1000, ta, geo['ca'][sl])
    car = np.nan_to_num(p[:, C0:C1] / 0.81, nan=0)
    ratio = norm_ratio(np.where(pe[:, C0:C1] > 0.04, p[:, C0:C1] / np.maximum(pe[:, C0:C1], 1e-3), np.nan))
    rcs = norm_ratio(np.where(pc[:, C0:C1] > 0.04, p[:, C0:C1] / np.maximum(pc[:, C0:C1], 1e-3), np.nan), 85)
    e = np.nansum(np.nan_to_num(p), 1) * dc_kw * 0.25
    k_true = np.round(6 * (1 - fr[sl, j])); clean = np.isfinite(fr[sl, j]) & (np.abs(6 * (1 - fr[sl, j]) - k_true) < 0.25)
    enc = lambda x: u8(np.nan_to_num(np.clip(x / 1.25, 0, 1) * 255, nan=0))
    return {'name': name, 'desc': desc, 'inverter': j + 1, 'dates': [str(d.date()) for d in days[sl]], 'n': b - a,
            'carpet': u8(np.clip(car * 255, 0, 255)), 'ratio': enc(ratio), 'ratio_ok': u8(np.isfinite(ratio) * 255), 'ratio_cs': enc(rcs), 'ratio_cs_ok': u8(np.isfinite(rcs) * 255),
            'k_true': [float(k) if c else None for k, c in zip(k_true, clean)], 'kwh': L(e, 1),
            'modes': {'poa': mode_out(oP, sl, e, top_energy(oP, sl, p, np.where(np.isfinite(g), pe, np.nan), pc, dc_kw, False)),
                      'home': mode_out(oH, sl, e, top_energy(oH, sl, p, None, pc, dc_kw, True))}}

def lab_window(name, days, P, G, T, geo, o, oh, j, a, D=14):
    sl = slice(a, a + D); p = np.nan_to_num(P[sl, :, j].astype(float)); g = G[sl]; ta = T[sl]
    pe = SB2.expected_power(np.nan_to_num(np.maximum(g, 0)), np.nan_to_num(ta, nan=15), geo['ca'][sl]); pu = IJ.unclipped(p, pe)
    return {'name': name, 'dates': [str(d.date()) for d in days[sl]], 'p': f16(p), 'pu': f16(pu), 'g': f16(np.nan_to_num(g, nan=-1)), 'ta': f16(np.nan_to_num(ta, nan=15)),
            'cz': f16(geo['cz'][sl]), 'ca': f16(geo['ca'][sl]), 'gc': f16(geo['poa_clear'][sl] / 1000.0), 'el': f16(geo['el'][sl]), 'az': f16(geo['az'][sl]),
            'ref': {'poa': r(o['ref'][a + D // 2], 4), 'home': r(oh['ref'][a + D // 2], 4)}}

# ------------------------------------------------------------------ Colorado, 9068
def co_case(name, desc, focus, days, d, geo, K, oP, oH, j, a, b):
    sl = slice(a, b); p = d['P'][sl, :, j].astype(float); ta = np.nan_to_num(d['T'][sl], nan=15); inv = 1 if j < 4 else 2
    pc = SB2.expected_power(geo['poa_clear'][sl] / 1000, ta, geo['ca'][sl])
    g = np.where(K['sensor_ok'][sl, None], d['G'][sl], np.nan); pe = np.where(np.isfinite(g), SB2.expected_power(np.nan_to_num(np.maximum(g, 0)), ta, geo['ca'][sl]), np.nan)
    car = np.nan_to_num(p[:, C0:C1] / 0.9, nan=0)
    rcs = norm_ratio(np.where(pc[:, C0:C1] > 0.04, p[:, C0:C1] / np.maximum(pc[:, C0:C1], 1e-3), np.nan), 85)
    A = d['A'][sl][..., d['inv'] == inv].astype(np.float32); th = geo['theta'][sl]
    with np.errstate(all='ignore'): dev = np.nanmean(np.abs(A - th[..., None]), -1)
    up = geo['cz'][sl] > 0.1; dev = np.where(up & np.isfinite(dev), dev, np.nan)[:, C0:C1]
    e = np.nansum(np.nan_to_num(p), 1) * R9.SYS['pdc_kw'] * 0.25
    snow_key = np.where(K['snow_known'][sl, j], K['snow'][sl, j], np.nan); stuck_key = K['stuck%d' % inv][sl]; out_key = K['outage'][sl, j]
    enc = lambda x: u8(np.nan_to_num(np.clip(x / 1.25, 0, 1) * 255, nan=0))
    return {'name': name, 'desc': desc, 'focus': focus, 'module': j + 1, 'inverter': inv, 'dates': [str(x.date()) for x in days[sl]], 'n': b - a,
            'carpet': u8(np.clip(car * 255, 0, 255)), 'ratio_cs': enc(rcs), 'ratio_cs_ok': u8(np.isfinite(rcs) * 255),
            'track': u8(np.nan_to_num(np.clip(dev / 40, 0, 1) * 255, nan=0)), 'track_ok': u8(np.isfinite(dev) * 255),
            'kwh': L(e, 0), 'key': {'snow': L(snow_key), 'stuck': L(stuck_key), 'outage': L(out_key), 'tmean': L(K['tmean'][sl], 1)},
            'modes': {'poa': mode_out(oP, sl, e, top_energy(oP, sl, p, pe, pc, R9.SYS['pdc_kw'], False)), 'home': mode_out(oH, sl, e, top_energy(oH, sl, p, None, pc, R9.SYS['pdc_kw'], True))}}

def main(ckpt, out=_P.LAB + '/data.json'):
    net = E2.load(ckpt); name = os.path.basename(ckpt).replace('.pt', '')
    days, P, G, T, fr, geo = RX.load(); di = {str(x.date()): i for i, x in enumerate(days)}
    runs = {}
    def run(j, home):
        if (j, home) not in runs: runs[(j, home)] = RX2.diagnose(net, P[:, :, j].astype(float), G, T, geo, home)
        return runs[(j, home)]
    J = lambda f, alt=None: json.load(open(f)) if os.path.exists(f) else alt          # missing exams (a quick dev build) fall back to v1's
    ep = J(CK + name + '_episodes.json', J(CK + 's3_polish_episodes.json')); eph = J(CK + name + '_episodes_home.json', ep)
    s_eps = [e for e in J(CK + 's3_polish_episodes.json')['episodes'] if e['kind'] == 'string' and e['repair'] is not None and e['onset_found'] and e['inv'] != 10]
    off = next(e for e in ep['episodes'] if e['kind'] == 'offline' and e['inv'] == 0 and e['date'] == '2019-06-18')
    e2 = s_eps[len(s_eps) // 2]
    def cc(nm, desc, j, a, b): return ca_case(nm, desc, days, P, G, T, geo, fr, run(j, False), run(j, True), j, a, b)
    hero = cc('hero', '', 10, di['2019-09-17'], di['2019-11-01'])
    C = [cc('A string dies, then is fixed', 'Inverter 11 loses one of its six strings on 27 Sep 2019; a crew restores it on 16 Oct.', 10, di['2019-09-06'], di['2019-11-01']),
         cc('Another string failure', f"Inverter {e2['inv'] + 1}: one string out from {e2['date']} until {e2['repair_date']}.", e2['inv'], e2['day'] - 14, min(e2['repair'] + 16, len(days))),
         cc('Inverter offline', 'Inverter 1 goes dark on 18 Jun 2019 and stays off for nine days.', 0, off['day'] - 12, off['day'] + 22),
         cc('Winter row shading', 'A healthy inverter through the low winter sun. Rows of panels start shading the row behind.', 10, di['2019-11-15'], di['2020-01-15']),
         cc('Spring clipping', 'Cool, bright spring days push the array past what the 30 kW inverter can deliver.', 10, di['2019-03-20'], di['2019-05-10'])]
    lab = [lab_window(nm, days, P, G, T, geo, run(10, False), run(10, True), 10, di[dt]) for nm, dt in (('Spring', '2019-04-08'), ('Summer', '2019-07-08'), ('Winter', '2019-12-04'))]
    # Colorado
    cdays, d9, geo9 = R9.load(); K = dict(np.load(_P.WORK + '/data/sys9068_keys.npz')); ci = {str(x.date()): i for i, x in enumerate(cdays)}
    cruns = {}
    def crun(j, home):
        if (j, home) not in cruns:
            g = d9['G'] if home else np.where(K['sensor_ok'][:, None], d9['G'], np.nan)
            cruns[(j, home)] = RX2.diagnose(net, d9['P'][:, :, j].astype(np.float64), g, d9['T'], geo9, home)
        return cruns[(j, home)]
    def co(nm, desc, focus, j, a, b): return co_case(nm, desc, focus, cdays, d9, geo9, K, crun(j, False), crun(j, True), j, ci[a], ci[b] + 1)
    CO = [co('Snowstorm, then frozen trackers', 'Snow buries the field on 21 Jan 2018 and slides off over four days. After a second storm in February, the trackers on inverter 1 freeze in place for four days.', 'snow', 1, '2018-01-14', '2018-02-24'),
          co('Snow on the panels', 'A February 2019 storm covers the array for three days; it is gone by the first sunny afternoon.', 'snow', 2, '2019-01-25', '2019-02-25'),
          co('Commissioning: trackers not tracking yet', 'October 2017, the plant\'s first weeks. Most of inverter 1\'s trackers sit off their target angle on many days while the control system is tuned.', 'stuck', 0, '2017-09-28', '2017-11-10'),
          co('Inverter 2 trips', 'Module 5 (inverter 2) goes dark from 25 to 29 Oct 2017 while inverter 1 keeps producing.', 'outage', 4, '2017-10-10', '2017-11-14'),
          co('Snow, winter 2023', 'Three storms in a month, January to February 2023.', 'snow', 6, '2023-01-12', '2023-02-20')]
    # exams
    R2 = json.load(open(_P.WORK + '/ratchet2.json')); rec = next(x for x in R2 if x['ckpt'] == os.path.basename(ckpt)); base = R2[0]
    inj1 = J(CK + 's3_polish_inject.json')['summary']; inj = J(CK + name + '_inject.json', {'summary': inj1})['summary']; injh = J(CK + name + '_inject_home.json', {'summary': inj1})['summary']
    rdn = name if os.path.exists(CK + name + '_rdtools.json') else 's3_polish'
    rd = J(CK + rdn + '_rdtools.json'); ser = np.load(CK + rdn + '_inv11_series.npz')
    sd = pd.to_datetime(ser['days'], unit='D'); df = pd.DataFrame({k: ser[k] for k in ('pi', 'pf', 'shade', 'clip', 'soil', 'frac')}, index=sd)[ser['healthy']]
    mon = df.groupby(df.index.month).mean()
    season = {'month': list(range(1, 13)), **{k: [r(x) for x in mon[k]] for k in mon.columns}, 'rdtools_soil': [r(rd['soiling']['monthly'][str(m)][0]) if 'monthly' in rd['soiling'] else None for m in range(1, 13)]}
    R1 = json.load(open(_P.WORK + '/ratchet.json'))
    rungs1 = [{'ckpt': x['ckpt'], 'note': x['note'], 'accepted': x['accepted'], 'why': x['why'], 'objective': x['objective'],
               'skill': {k: x['val'][k]['skill'] for k in ('pf', 'frac', 'soil', 'shade', 'clip', 'outage')}, 'count_acc': x['val']['count_acc'], 'real': x['real']} for x in R1]
    rungs2 = [{'ckpt': x['ckpt'], 'note': x['note'], 'accepted': x['accepted'], 'why': x['why'], 'objective': x['objective'], 'poa': x['val']['poa']['objective'], 'home': x['val']['home']['objective'],
               'trackers': x['val']['trackers']['objective'], 'skill_poa': {k: x['val']['poa'][k]['skill'] for k in E2.SKILLS}, 'skill_home': {k: x['val']['home'][k]['skill'] for k in E2.SKILLS},
               'real': {'home_when_dead': x['real']['2107_home']['count_acc_when_dead'], 'poa_when_dead': x['real']['2107_poa']['count_acc_when_dead'],
                        'snow_auc': x['real']['9068_poa']['snow_auc'], 'stuck_auc': x['real']['9068_poa']['stuck_auc_2pct'], 'outage_auc_home': x['real']['9068_home']['outage_auc']}} for x in R2]
    samples = 0
    for x in R1 + [y for y in R2 if y['ckpt'] != 's3_polish.pt']:
        c = torch.load(CK + x['ckpt'], weights_only=False); samples += c['args']['steps'] * c['args']['bs']
    data = {'cases': C, 'hero': hero, 'lab': lab, 'co': CO, 'carpet_cols': [C0, C1],
            'exam': {'real': rec['real'], 'real_v1': base['real'], 'inject': inj, 'inject_home': injh, 'inject_v1': inj1, 'episodes': ep['summary'], 'episodes_home': eph['summary'],
                     'episodes_v1': json.load(open(CK + 's3_polish_episodes.json'))['summary'],
                     'episodes_v1_home': json.load(open(CK + 's3_polish_episodes_home.json'))['summary'] if os.path.exists(CK + 's3_polish_episodes_home.json') else None, 'rdtools': {k: v for k, v in rd.items() if k != 'soiling'},
                     'soiling': {k: v for k, v in rd['soiling'].items() if k != 'monthly'}, 'season': season},
            'rungs': rungs1, 'rungs2': rungs2, 'physics': json.load(open(_P.WORK + '/physics_check.json')),
            'release': {'ckpt': os.path.basename(ckpt), 'samples': samples, 'val': rec['val']}, 'clip_pu': 0.806, 'dc_kw': 37.2,
            'limits': json.load(open(_P.LAB + '/limits.json')) if os.path.exists(_P.LAB + '/limits.json') else [],
            'co_sys': {'dc_kw': R9.SYS['pdc_kw'], 'sensor_ok_days': int(K['sensor_ok'].sum()), 'days': len(cdays)}}
    json.dump(data, open(out, 'w')); print('web data MB %.2f' % (len(json.dumps(data)) / 1e6))

if __name__ == '__main__':
    main(sys.argv[1])
