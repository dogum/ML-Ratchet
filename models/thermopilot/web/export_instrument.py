"""Export ThermoPilot's live instrument for the website.

The data come from the original lab page (models/thermopilot/lab/index.html, built by training/build_page.py from
training/scripts/lab_export.py): three RECS 2020 exam respondents' twins (a heat-pump house, a gas furnace + central
AC house, an electric-baseboard home) in five real 2017 climates (BDG2 site weather: Minneapolis, Ogden, Berkeley,
Austin, Tempe), a winter and a summer week, and five real tariffs. For each home x climate x week the page holds the
controller's own linear model of the house (BuildingSim's exact hourly 5R1C maps for the nominal twin, with the
heat-pump / furnace / AC tables); for each tariff it holds the prices and the full BuildingSim run of every controller
(the "truth" the browser model is compared with). BuildingSim itself is Python and is not in the repository yet, so
weather changes snap between these five real climates.

The browser runs ThermoPilot closed-loop on the linear house, hour by hour, in either mode: v1.1 (default) the card
MPC, a 24 h dynamic programme on the card's 12-number linear house map (fitted here to the example's hourly maps with
thermopilot_core.fit_map) -> setpoints -> comfort shield -> house step; or v1.0, features -> MLP -> setpoints -> comfort
shield -> house step. A fixed thermostat and the occupant's own setback schedule run next to it. The full BuildingSim
runs stored in the lab page are the v1.0 network's (and the other controllers'); there are none for v1.1. Controls change the input:
climate, week, tariff, the comfort weight lambda the network is told, the random forecast error and a forecast bias.

This file runs the same closed loop in Python with the shipped inference module (thermopilot_core.py) on float16
weights and writes reference outputs for site/tests/parity.test.ts.

    python models/thermopilot/web/export_instrument.py      # writes site/public/instruments/thermopilot/

The weather in the examples is derived from Building Data Genome 2 (CC BY-SA 4.0): the exporter writes
LICENSE-DATA.md next to the data.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(MODEL))
sys.path.insert(0, MODEL)
sys.path.insert(0, REPO)

import thermopilot_core as tp  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir('thermopilot')
CTRL = ['fixed', 'setback', 'precool', 'mpc_c', 'oracle', 'thermopilot']
TABLES = ['Qmax', 'Qh', 'Ph', 'eff', 'Qs', 'Pc']
LIVE = ['A', 'At', 'C', 'Ct', 'To', 'G', 'base', 'state', 'hour', 'wd']
RUN = ['price', 'lo', 'hi', 'dperiod', 'drate', 'frate', 'drem', 'frem']
NZ = 232                       # forecast-noise draws (N + 40, as the page)
LICENSE = """# Data licence: CC BY-SA 4.0 (Building Data Genome 2)

The data files in this folder (`examples.json`, `examples.bin`, `reference.json`) are derived from the Building Data
Genome 2 dataset (BDG2): Miller, C., et al. (2020), "The Building Data Genome Project 2, energy meter data from the
ASHRAE Great Energy Predictor III competition", Scientific Data 7, 368, https://doi.org/10.1038/s41597-020-00712-x;
data at https://github.com/buds-lab/building-data-genome-project-2. Licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). Changes: the 2017 hourly site weather of five BDG2 sites (Hog,
Bobcat, Bear, Bull, Fox) was cleaned by BuildingData (UTC, gaps interpolated), cut to one winter and one summer week,
run through BuildingSim with twins of three RECS 2020 respondents (public domain) and URDB / ERCOT tariffs, and the
outdoor temperature, solar, the house maps and the simulated runs rounded and stored as float32. You may share and
adapt them under the same licence, with this credit. This licence covers the data files only; the code in this
repository is MIT. The network weights (`weights.*`) are not BDG2 data.
"""


def lab_data():
    s = open(os.path.join(MODEL, 'lab', 'index.html'), encoding='utf-8').read()
    tag = '<script id="data" type="application/json">'
    a = s.index(tag) + len(tag)
    b = s.index('</script>', a)
    return json.loads(s[a:b].replace('<\\/', '</'))


def interp(x, xs, ys):
    return float(np.interp(x, xs, ys))


def ctx_of(ex, run, N, price_mul=1.0):
    cl = lambda a: np.array([a[min(i, len(a) - 1)] for i in range(N)], float)
    lo, hi = cl(run['lo']), cl(run['hi'])
    month = np.full(N, 1)
    return dict(price=cl(run['price']) * price_mul, lo=lo, hi=hi, state=ex['state'].astype(int),
                pref_lo=float(ex['pref_lo']), pref_hi=float(ex['pref_hi']), dperiod=cl(run['dperiod']).astype(int),
                drate=cl(run['drate']), frate=cl(run['frate']), hour=ex['hour'].astype(int), wd=ex['wd'] > 0.5,
                ym=month, doy=month, drem=cl(run['drem']), frem=cl(run['frem']))


def forecast(ex, z, k, fe, bias):
    """24 h forecast at hour k: the real weather + bias + an error that grows with lead time (the page's model)."""
    To, G = ex['To'], ex['G']
    N = len(To)
    T, Gs = np.zeros(24), np.zeros(24)
    for j in range(24):
        i = min(k + j, N - 1)
        e = 0.0
        if j > 0 and fe > 0:
            e = fe * (0.6 + 1.6 * (1 - np.exp(-j / 24))) * (0.7 * z[(k + j) % NZ] + 0.3 * z[(k + 2 * j + 5) % NZ])
        T[j] = To[i] + bias + e
        Gs[j] = G[i]
    return T, Gs


def simulate(ex, ctx, n, ctrl, gas_price):
    """The page's browser house: BuildingSim's hourly 5R1C maps of the nominal twin + equipment tables."""
    card = ex['card']
    Tg = card['Tg']
    has_cool = bool(np.any(card['Qs'] > 0))
    Tm, Tz, q, e = float(ex['Tm0']), float(ctx['pref_lo']) + 0.5, 0.0, 0.0
    bill = gas = DH = kWh = 0.0
    peaks, rate = {}, {}
    T_out, h_out, c_out = [], [], []
    for k in range(n):
        h, c = ctrl(k, Tz, q, e)
        Tb = ex['A'][k] + ex['At'][k] * Tm
        be, To = float(ex['Aq']), ex['To'][k]
        Qmax, Qs = interp(To, Tg, card['Qmax']), interp(To, Tg, card['Qs'])
        Qhd = min((h - Tb) / be, Qmax) if Tb < h else 0.0
        Qcd = min((Tb - c) / be, Qs) if (Tb > c and Qs > 0) else 0.0
        T = Tb + be * (Qhd - Qcd)
        Tm = ex['C'][k] + ex['Ct'][k] * Tm + float(ex['Cq']) * (Qhd - Qcd)
        P = g = run_h = 0.0
        if card['kind'] == 'hp':
            Qh, Ph = interp(To, Tg, card['Qh']), interp(To, Tg, card['Ph'])
            f = min(Qhd, Qh) / Qh if Qh > 0 else 0.0
            P = Ph * f / (0.85 + 0.15 * f) if f > 0 else 0.0
            P += max(Qhd - Qh, 0.0)
            run_h = 1.0 if Qhd > Qh else f
        elif card['gas_heat']:
            g = Qhd / max(interp(To, Tg, card['eff']), 0.5)
            run_h = Qhd / Qmax if Qmax > 0 else 0.0
        else:
            P = Qhd
            run_h = 0.0 if card['kind'] == 'baseboard' else (Qhd / Qmax if Qmax > 0 else 0.0)
        Pc = interp(To, Tg, card['Pc'])
        fcl = Qcd / Qs if Qs > 0 else 0.0
        if fcl > 0:
            P += Pc * fcl / (0.85 + 0.15 * fcl)
        P += card['P_fan'] * min(run_h + fcl, 1.0)
        el = (ex['base'][k] + P) / 1000.0
        bill += ctx['price'][k] * el + gas_price * g / 1000.0
        gas += g / 1000.0
        kWh += P / 1000.0
        if ctx['drate'][k] > 0:
            key = int(ctx['dperiod'][k])
            peaks[key] = max(peaks.get(key, 0.0), el)
            rate.setdefault(key, float(ctx['drate'][k]))
        DH += max(ctx['lo'][k] - T, 0.0) + (max(T - ctx['hi'][k], 0.0) if has_cool else 0.0)
        Tz, q, e = T, Qhd - Qcd, el
        T_out.append(T); h_out.append(h); c_out.append(c)
    demand = sum(peaks[p] * rate[p] for p in peaks)
    return dict(T=np.array(T_out), h=np.array(h_out), c=np.array(c_out), bill=bill + demand, demand=demand, DH=DH,
                kWh=kWh, gas=gas)


def card_map(ex):
    """The v1.1 card's house map, fitted to the example's hourly maps (the nominal twin's BuildingSim maps)."""
    N = len(ex['To'])
    return tp.fit_map(ex['To'], ex['G'], ex['base'], ex['A'], ex['At'], ex['C'], ex['Ct'], np.full(N, ex['Aq']),
                      np.full(N, ex['Cq']))


def run_pilot(W, ex, run, lam_mul=1.0, fe=1.0, bias=0.0, z=None, n=168, price_mul=1.0, log_hours=(), mode='mpc'):
    """ThermoPilot closed loop on the browser house. Logged at the hours in log_hours: the card MPC's 36 action values
    ($) for mode='mpc', the network's 36 logits for mode='network'."""
    N = len(ex['To'])
    ctx = ctx_of(ex, run, N, price_mul)
    pilot = tp.pilot(ex['card'], ctx, float(ex['lam']) * lam_mul, ex['base'], float(ex['gas_price']), mode=mode, W=W)
    rec, logits, acts = {}, [], []
    if mode == 'network':
        fstep = pilot.feat.step

        def feat_step(*a):
            rec['x'] = fstep(*a)
            return rec['x']
        pilot.feat.step = feat_step

    def ctrl(k, Tz, q, e):
        T, G = forecast(ex, z, k, fe, bias)
        sp = pilot.step(k, Tz, q, e, T, G)
        acts.append(pilot.last)
        if k in log_hours:
            logits.append(np.array(pilot.Q) if mode == 'mpc' else tp.mlp_forward(W, rec['x'].astype(np.float64)))
        return sp['heat_sp'], sp['cool_sp']
    r = simulate(ex, ctx, n, ctrl, float(ex['gas_price']))
    r['acts'], r['logits'] = np.array(acts), logits
    return r, ctx


def main():
    D = lab_data()
    L = D['lab']
    homes, climates, tariffs, weeks = L['homes'], L['climates'], L['tariffs'], L['weeks']
    nH, nC, nW, nT = len(homes), len(climates), len(weeks), len(tariffs)

    # ---- weights: float16 (normalisation in float32), and the same values for the Python reference
    W = dict(np.load(os.path.join(MODEL, 'thermopilot_v1.npz')))
    layers = [k for k in W if k[0] in 'Wb']
    W16 = roundtrip({k: W[k] for k in layers}, 'f16')
    Wr = {k: v.astype(np.float64) for k, v in W16.items()}
    Wr.update(mu=W['mu'].astype(np.float32).astype(np.float64), sd=W['sd'].astype(np.float32).astype(np.float64),
              shield=np.float64(W['shield']))
    n_params = int(sum(W[k].size for k in layers))
    write_pack(OUT / 'weights', {**{k: W[k] for k in layers}, 'mu': W['mu'], 'sd': W['sd']}, dtype='f16',
               keep={'mu': 'f32', 'sd': 'f32'},
               meta=dict(n_params=n_params, shield=int(W['shield']), layers=len(layers) // 2,
                         arch='298-384-384-36'))

    # ---- examples: 30 house maps (home x climate x week), 150 tariff runs (x tariff), forecast noise
    rng = np.random.default_rng(20261117)
    z = rng.standard_normal(NZ).astype(np.float32)
    T = {f'L_{k}': np.zeros((nH * nC * nW, 192), np.float32) for k in LIVE}
    T.update({f'K_{k}': np.zeros((nH * nC * nW, 91), np.float32) for k in TABLES + ['Tg']})
    T.update({f'R_{k}': np.zeros((nH * nC * nW * nT, 168), np.float32) for k in RUN + ['Ttp', 'Tout']})
    T['z'] = z
    live_meta, run_meta, exs = [], [], {}
    for hi in range(nH):
        for ci in range(nC):
            for wi in range(nW):
                li = (hi * nC + ci) * nW + wi
                m = L['live'][f'{hi}_{ci}_{wi}']
                for k in LIVE:
                    T[f'L_{k}'][li] = m[k]
                for k in TABLES + ['Tg']:
                    T[f'K_{k}'][li] = m['card'][k]
                cs = {k: m['card'][k] for k in ('UA', 'C', 'area', 'kind', 'gas_heat', 'fan_cont', 'res', 'P_fan', 'backup')}
                # the Python view of this example, on exactly the float32 values the browser reads
                ex = {k: T[f'L_{k}'][li].astype(np.float64) for k in LIVE}
                ex.update(Aq=np.float32(m['Aq']).item(), Cq=np.float32(m['Cq']).item(), Tm0=np.float32(m['Tm0']).item(),
                          lam=m['lam'], gas_price=m['gas_price'], pref_lo=homes[hi]['pref']['lo'], pref_hi=homes[hi]['pref']['hi'])
                tabs = {k: T[f'K_{k}'][li].astype(np.float64) for k in TABLES + ['Tg']}
                # v1.1 card fields: the house map, heater size and efficiency (the browser's units are single speed
                # and have no constant fan, like the browser house)
                assert not cs['fan_cont']
                cs.update(map=card_map(ex), Qheat=float(np.max(tabs['Qmax'])), eff0=float(np.median(tabs['eff'])))
                live_meta.append(dict(Aq=m['Aq'], Cq=m['Cq'], Tm0=m['Tm0'], lam=m['lam'], gas_price=m['gas_price'], card=cs))
                ex['card'] = dict(cs, **tabs)
                exs[li] = ex
                for ti in range(nT):
                    ri = li * nT + ti
                    r = L['runs'][f'{hi}_{ci}_{wi}_{ti}']
                    for k in RUN:
                        T[f'R_{k}'][ri] = r[k]
                    T['R_Ttp'][ri] = r['ctrl']['thermopilot']['T']
                    T['R_Tout'][ri] = r['T_out']
                    run_meta.append({c: dict(bill=r['ctrl'][c]['bill'], DH=r['ctrl'][c]['DH'], J=r['ctrl'][c]['J'],
                                             kWh=r['ctrl'][c]['kWh'], demand=r['ctrl'][c]['demand'])
                                     for c in CTRL})
    meta = dict(homes=[dict(label=h['label'], id=h['id'], system=h['system'], area=h['area'], year=h['year'],
                            state=h['state'], pref_lo=h['pref']['lo'], pref_hi=h['pref']['hi']) for h in homes],
                climates=climates, tariffs=tariffs, weeks=weeks, live=live_meta, runs=run_meta, nz=NZ,
                week_days=[['Mon 9 Jan', 'Sun 15 Jan'], ['Mon 10 Jul', 'Sun 16 Jul']], year=2017)
    write_pack(OUT / 'examples', T, dtype='f32', meta=meta)

    # browser-house vs full BuildingSim at the as-built settings (for the notes and the report)
    agree, Jw = [], []
    for li, ex in exs.items():
        for ti in range(nT):
            r = L['runs'][f'{li // (nC * nW)}_{(li // nW) % nC}_{li % nW}_{ti}']
            run = {k: T[f'R_{k}'][li * nT + ti].astype(np.float64) for k in RUN}
            res, ctx = run_pilot(Wr, ex, run, z=z.astype(np.float64), mode='network')
            r11, _ = run_pilot(Wr, ex, run, z=z.astype(np.float64), mode='mpc')
            fx = simulate(ex, ctx, 168, lambda k, *a: (ctx['pref_lo'], ctx['pref_hi']), float(ex['gas_price']))
            agree.append((res['bill'], r['ctrl']['thermopilot']['bill']))
            lam = float(ex['lam'])
            Jw.append([x['bill'] + lam * x['DH'] for x in (r11, res, fx)])
    a = np.array(agree)
    print(f'browser house vs full BuildingSim, v1.0 network bill over {len(a)} runs: median ratio '
          f'{np.median(a[:, 0] / a[:, 1]):.3f}, IQR {np.percentile(a[:, 0] / a[:, 1], 25):.3f}-'
          f'{np.percentile(a[:, 0] / a[:, 1], 75):.3f}')
    j = np.array(Jw)
    print(f'browser house, {len(j)} weeks, J (bill + lambda K h) vs fixed: v1.1 card MPC saves '
          f'{100 * (1 - j[:, 0].sum() / j[:, 2].sum()):.2f} %, v1.0 network {100 * (1 - j[:, 1].sum() / j[:, 2].sum()):.2f} %; '
          f'v1.1 lower J in {int(np.sum(j[:, 0] < j[:, 1] - 1e-9))} weeks, higher in {int(np.sum(j[:, 0] > j[:, 1] + 1e-9))}')

    # ---- parity cases: closed loop on the browser house (setpoints, temperatures, bill, comfort, logits)
    zz = z.astype(np.float64)
    cases = []
    picks = [(0, 4, 1, 0), (0, 0, 0, 2), (1, 2, 0, 1), (1, 3, 1, 4), (2, 1, 0, 3)]
    variants = [('as built', dict()), ('lambda x4, no forecast error', dict(lam_mul=4.0, fe=0.0)),
                ('forecast +3 K bias, error x2', dict(bias=3.0, fe=2.0)), ('lambda x0.25, bias -3 K', dict(lam_mul=0.25, bias=-3.0))]
    for (hi, ci, wi, ti), mode in [(p, md) for md in ('mpc', 'network') for p in picks]:
        li = (hi * nC + ci) * nW + wi
        ri = li * nT + ti
        run = {k: T[f'R_{k}'][ri].astype(np.float64) for k in RUN}
        for vname, kw in variants:
            prm = dict(lam_mul=1.0, fe=1.0, bias=0.0)
            prm.update(kw)
            res, _ = run_pilot(Wr, exs[li], run, z=zz, log_hours=(5, 77, 150), mode=mode, **prm)
            out = np.concatenate([res['h'], res['c'], res['T'], [res['bill'], res['DH']], *res['logits']])
            cases.append(dict(example=li, run=ri, variant=f'{mode}: {vname}', mode=mode, lam_mul=prm['lam_mul'], fe=prm['fe'], bias=prm['bias'],
                              log_hours=[5, 77, 150], outputs=[float(v) for v in out]))
            print(f"{mode:8s} {homes[hi]['label']:32s} {climates[ci][1]:22s} {weeks[wi]:6s} {tariffs[ti][0]:16s} {vname:32s} "
                  f"bill ${res['bill']:7.2f}  {res['DH']:5.1f} K h  (v1.0 full sim ${L['runs'][f'{hi}_{ci}_{wi}_{ti}']['ctrl']['thermopilot']['bill']:.2f})")
    write_json(OUT / 'reference.json', dict(cases=cases, tolerance=2e-3))
    (OUT / 'LICENSE-DATA.md').write_text(LICENSE, encoding='utf-8')
    tot = sum(f.stat().st_size for f in OUT.iterdir())
    print(f'wrote {OUT} ({tot / 1e6:.2f} MB), {n_params:,} weights')


if __name__ == '__main__':
    main()
