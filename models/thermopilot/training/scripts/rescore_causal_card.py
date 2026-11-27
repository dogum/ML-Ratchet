"""Dated supplementary re-score, 27 Nov 2026 (audit follow-up, owner-approved): v1.1 (the card MPC behind the 6 h
shield, exam tag `card4`) with a CAUSAL evaluation card. The released exam fitted each house's card (`full_card`: UA,
C, P_fan, Qheat and the 12-number house map) on the nominal twin over the evaluation month itself, so the card carried
realized weather (wind via infiltration) beyond the 24 h forecast. Here the card is fitted the same way on the nominal
twin over the calendar month BEFORE the episode (same weather source, same comfort preferences), and the evaluation
episode is otherwise unchanged (same true house, tariff, noisy forecast, seed). Nothing is trained or chosen; the
exam is seen. Plan in NOTEBOOK.md (27 Nov 2026).

    rescore_causal_card.py SET repro [N]      # first N cases through the released path: must reproduce pol_card4.json
    rescore_causal_card.py SET run [I0 I1]    # -> results/SET/pol_card4causal.json[.partI0]
    then: evaluate.py SET merge card4causal; intervals.py card4causal; intervals.py card4causal vs:card4 causal_vs_card4
"""
import os, sys, json, time, traceback
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORK as ROOT
import numpy as np
from tp import policy as P
from tp.episode import Episode
from tp.features import full_card
import evaluate as EV

SHIELD = 4          # card4: v1.0's 6 h shield (thermopilot_core.shield_sp mode >= 4)


def prior_card(c):
    """Card fitted on the nominal twin over the month before the episode (Dec of the previous year for January).
    The window ends exactly at the episode start: an Episode would pad its model window by H + 2 hours (into the
    evaluation month), and Model's descriptors (UA) and card_map average over the whole window (Codex, PR #28)."""
    import pandas as pd
    import buildingsim as bs
    from tp import env
    from tp.episode import prep_spec
    from tp.mpc import Model
    y, m = (c['year'] - 1, 12) if c['month'] == 1 else (c['year'], c['month'] - 1)
    df, meta = env.weather(c['wsrc'], y)
    tz = meta['tz']
    t0 = pd.Timestamp(year=y, month=m, day=1, tz=tz).tz_convert('UTC')
    t1 = pd.Timestamp(year=c['year'], month=c['month'], day=1, tz=tz).tz_convert('UTC')
    spec = prep_spec(c['spec_nom'], c['pref'])
    sim = bs.Simulator(spec, (df, meta), t0, t1, dt='1h', warmup_days=0, record_trend=False)
    M = Model(sim)
    assert pd.Timestamp(sim.times[-1]) < t1 and M.n == len(M.To) == int(round((t1 - t0) / pd.Timedelta(hours=1)))
    lo, hi, state = env.band_schedule(sim, c['pref'])
    return full_card(M, dict(lo=lo, hi=hi), c['pref']['hi']), (y, m)


def controller(ep, card, pref_hi):
    m = P.core.CardMPC(card, ep.ctx, ep.lam, ep.model.base_W, ep.gas_price)
    inner = m.step
    def step(k, *a):
        sp = inner(k, *a)
        h, cc = P.core.shield_sp(ep.ctx, k, sp['heat_sp'], sp['cool_sp'], SHIELD)
        return dict(heat_sp=h, cool_sp=cc)
    m.step = step
    return P.CoreAdapter(m, ep.fc_noisy)


def card_diff(a, b):
    out = {k: (b[k] - a[k]) / a[k] for k in ('UA', 'P_fan', 'Qheat') if a[k]}
    for k in ('At', 'Aq', 'Ct', 'Cq'):
        out['map_' + k] = b['map'][k] - a['map'][k]
    return {k: float(v) for k, v in out.items()}


def main():
    name, cmd = sys.argv[1], sys.argv[2]
    cs = EV.load_cases(name)
    od = os.path.join(ROOT, 'results', name)
    if cmd == 'repro':
        ref = json.load(open(os.path.join(od, 'pol_card4.json')))
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        for c in cs[:n]:
            ep = EV.episode(c)
            mm, _ = ep.run(controller(ep, full_card(ep.model, ep.ctx, c['pref']['hi']), c['pref']['hi']).act)
            print(c['id'], 'J now %.4f logged %.4f' % (mm['J'], ref[c['id']]['J']), flush=True)
        return
    i0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    i1 = int(sys.argv[4]) if len(sys.argv) > 4 else len(cs)
    out = {}
    t0 = time.time()
    todo = cs[i0:i1]
    if cmd == 'fix':     # re-run the cases that errored in the merged file (e.g. a missing price file), in place
        fn = os.path.join(od, 'pol_card4causal.json')
        out = json.load(open(fn))
        todo = [c for c in cs if 'error' in out.get(c['id'], {'error': 1})]
        i0, i1 = 0, len(cs)
        print(len(todo), 'cases to re-run', flush=True)
    for c in todo:
        try:
            ep = EV.episode(c)
            card, ym = prior_card(c)
            mm, _ = ep.run(controller(ep, card, c['pref']['hi']).act)
            r = EV.clean(mm)
            r['card_month'] = list(ym)
            r['card_vs_episode_card'] = card_diff(full_card(ep.model, ep.ctx, c['pref']['hi']), card)
            out[c['id']] = r
        except Exception as e:
            out[c['id']] = dict(error=traceback.format_exc()[-600:])
            print(c['id'], 'ERROR', repr(e)[:200], flush=True)
    part = '' if (i0, i1) == (0, len(cs)) else f'.part{i0}'
    json.dump(out, open(os.path.join(od, f'pol_card4causal.json{part}'), 'w'))
    print(name, 'card4causal', len(out), 'cases %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
