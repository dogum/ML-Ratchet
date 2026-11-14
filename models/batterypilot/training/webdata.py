"""Data for the showpiece page: lab node-years (real exam prices + precomputed strategy traces), exam tables, ratchet,
twin test, calibration.  python3 webdata.py <rung>  -> web/data.js (window.BPDATA = {...})"""
import base64, json, sys
import numpy as np, pandas as pd
import _paths  # noqa: E402  (this folder, GridSim and GridData on sys.path; outputs to WORK)
from battery import Battery, settle
from evaluate import run_strategy
from model import PolicyNet
from realdata import node_frame, fill
from exam import dev_thresholds

ROOT = _paths.WORK   # was the sandbox workspace grid/batterypilot
LAB = [('ERCOT', n, y) for n in ('HB_HOUSTON', 'HB_NORTH', 'HB_WEST', 'HB_SOUTH') for y in (2023, 2024, 2025)] + \
      [('CAISO', n, y) for n in ('TH_SP15_GEN-APND', 'TH_NP15_GEN-APND') for y in (2024, 2025)]


def b64(a, dt):
    return base64.b64encode(np.ascontiguousarray(np.asarray(a, dt)).tobytes()).decode()


def main(rung):
    net = PolicyNet(f'{ROOT}/ckpt/{rung}.npz')
    w = dict(np.load(f'{ROOT}/ckpt/{rung}.npz'))
    layers = []
    i = 0
    while f'W{i}' in w:
        W = w[f'W{i}']; layers.append(dict(W=b64(W.ravel(), '<f4'), b=b64(w[f'b{i}'], '<f4'), nin=W.shape[1], nout=W.shape[0]))
        i += 1
    D = dict(net=dict(x_mu=b64(w['x_mu'], '<f4'), x_sd=b64(w['x_sd'], '<f4'), layers=layers), lab=[])
    check = []
    for m, n, y in LAB:
        f, _ = fill(node_frame(m, n, f'{y}-01-01', f'{y + 1}-01-01'))
        f = f.dropna(subset=['rt', 'da'])
        p = {c: f[c].values for c in ['rt', 'da', 'df', 'T', 'hour', 'day', 'dow', 'doy']}
        thr = dev_thresholds(m, n)
        tz = 'America/Chicago' if m == 'ERCOT' else 'America/Los_Angeles'
        loc = f.index.tz_convert(tz)
        assert len(f) == len(pd.date_range(f.index[0], f.index[-1], freq='h')), 'lab path must be gap-free'
        # compact encodings: prices x2 and temperature x100 as int16, demand forecast /4 as uint16 (the page's JS
        # derives the local calendar from t0_utc with the IANA time zone, exactly like pandas)
        rec = dict(market=m, node=n.replace('_GEN-APND', ''), year=y, n=len(f), t0_utc=f.index[0].isoformat(),
                   tz=tz, rt=b64(np.clip(np.round(p['rt'] * 2), -32768, 32767), '<i2'),
                   da=b64(np.clip(np.round(p['da'] * 2), -32768, 32767), '<i2'),
                   df=b64(np.clip(np.round(p['df'] / 4), 0, 65535), '<u2'), T=b64(np.round(p['T'] * 100), '<i2'),
                   thr=thr, strat={})
        for d in (1, 2, 4):
            bat = Battery(1.0, d)
            for st in ('pf', 'rolling', 'spread', 'threshold', 'bp'):
                s = run_strategy(st, p, bat, net, thr)
                nt = s['gross_t'] - s['deg_t']
                daily = pd.Series(nt).groupby(p['day'] - p['day'][0]).sum()
                rec['strat'][f'{st}{d}'] = dict(daily=b64(daily.values, '<f4'), net=round(s['net'], 1),
                                               gross=round(s['gross'], 1), deg=round(s['deg'], 1),
                                               cycles=round(s['cycles'], 1))
                if st in ('pf', 'rolling', 'bp'):
                    rec['strat'][f'{st}{d}']['soc'] = b64(np.round(np.clip(s['soc'], 0, 1) * 255), '<u1')
                if st == 'bp' and n in ('HB_HOUSTON', 'TH_SP15_GEN-APND') and d == 2:
                    check.append(dict(node=n, year=y, d=d, soc_first48=[round(float(x), 6) for x in s['soc'][:49]],
                                      net=s['net'], gross=s['gross'], deg=s['deg']))
        D['lab'].append(rec)
        print(m, n, y, flush=True)
    for k in ('report', 'rows'):
        pass
    D['exam'] = json.load(open(f'{ROOT}/data/exam/{rung}_report.json'))
    D['ratchet'] = json.load(open(f'{ROOT}/ratchet.json'))
    D['ratchet'].pop('rule', None)
    try:
        D['twin'] = json.load(open(f'{ROOT}/data/twin_{rung}.json'))
        D['twin_week'] = json.load(open(f'{ROOT}/data/twin_week.json'))
    except FileNotFoundError:
        pass
    cal = json.load(open(f'{ROOT}/data/calibration.json'))
    D['calibration'] = dict(params=cal['calibrated'], real=cal['stageA_best']['real'], sim=cal['stageA_best']['sim'],
                            uri=cal['stageB_best'])
    D['heroNote'] = ('It beats the simple rules almost everywhere and roughly ties the classical day-ahead optimiser: '
                     'ahead with 4-hour batteries, behind with 1-hour ones, and behind in CAISO.')
    D['limits'] = [
        'It does not beat the classical day-ahead optimiser overall. Mean share of perfect hindsight: ERCOT 0.667 vs 0.685, '
        'CAISO 0.642 vs 0.681. It wins with 4-hour batteries (ERCOT 0.77 vs 0.74) and loses with 1-hour ones (0.56 vs 0.63), '
        'where the optimiser cycles harder (about 300 vs 245 cycles a year).',
        'The simulator is easier than reality. On simulated twins of the dev years it captures 91-94 % (2019) and 64-68 % '
        '(2021); on the real prices of the same years, 80 % and 38 %. Real spikes are less predictable from day-ahead prices '
        'than the simulated ones.',
        'Energy arbitrage only, settled hourly at the mean of the four 15-minute real-time prices. No ancillary services '
        '(where much ERCOT battery revenue came from in 2023), no 5-minute spikes, a 1 MW price taker, and no congestion: '
        'the simulator is a copper plate, so node-to-node differences in the exam are untrained.',
        'The CAISO side of the simulator is a family of settings (solar x1.6, fixed imports x0.6, carbon price, $1,000-2,000 '
        'caps) checked against five dev months, not a fitted model. Its prices are more volatile than real CAISO prices.',
        'Scarcity calibration ended at the edge of its search grid (ORDC mu = -4,000 MW, sigma = 5,000 MW; cold outages 55 %), '
        'and simulated day-ahead prices run 10-20 % above real-time on average (real: 1-5 %).',
        'Wear costs are assumptions: LiFePO4 cells at $130/kWh, 6,000 full cycles to 80 %, Woehler exponent 1.3, HVAC '
        'container. Calendar ageing is not charged to any strategy.',
        'A comparison network trained on real 2018/2020/2022 ERCOT prices (breaking the physics-only rule; not released) '
        'scored 0.690 in ERCOT and 0.564 in CAISO: real data helps a little in Texas, the simulator helps in California.']
    s = 'window.BPDATA = ' + json.dumps(D, separators=(',', ':')) + ';\n'
    open(f'{ROOT}/web/data.js', 'w').write(s)
    json.dump(check, open(f'{ROOT}/web/pycheck.json', 'w'))
    print('data.js MB', len(s) / 1e6)


if __name__ == '__main__':
    main(sys.argv[1])
