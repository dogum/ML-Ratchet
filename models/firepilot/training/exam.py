"""FINAL EXAM (run once, after the ratchet is closed). The 48-fire exam replay subset (B's frozen list), REAL exam
landscapes and HRRR weather, graded in the simulator.

  python3 exam.py A <final_model> [shard nshard]   stage A, all 48 fires (cheap)
  python3 exam.py B <final_model> [shard nshard]   stage B, seeded fire order, as compute allows (MPC + PF bound)
  python3 exam.py summary <final_model>             -> results/exam_summary.json
  Physics: FIREPILOT_PHYSICS (default p1, as v1.0 was graded). p1 writes results/exam/ and exam_summary.json; any
  other physics writes results/exam_<physics>/ and exam_summary_<physics>.json and reads sigma_widen_<physics>.json.
  (resumable per fire: a crash resumes the same run; nothing is re-chosen. Stage A/B split decided at ~10:58 EST on
  2026-11-19 BEFORE any exam output existed, because the shared machine was ~2.5x oversubscribed.)

Per fire (seeded hidden draw shared by both budgets):
  * fidelity: the no-error truth physics free-run from first detection vs the observed FEDS PM perimeters (24/48/72 h)
  * budgets 300 and 1000 m/h: no action, the five classical baselines, r0 (proxy planner), FirePilot (value net),
    forecast-world CEM search (MPC with the forecast: 2 iterations x 6 per mode + 5 seeds, ~29 sims), perfect-foresight
    CEM bound (truth: 3 x 6 per mode + every other plan as a seed, ~46 sims).
Writes results/exam/<fid>.json and results/exam_summary.json (by period, bootstrap CIs over fires, coverage of the
value net's error bars with the widen-only factor fitted on the dev gate).
"""
import os
import sys
import json
import time
import traceback

import _paths  # noqa: E402  (repository paths: shared/fire on sys.path; outputs under WORK, see _paths.py)
import numpy as np

from fp.scenarios import real_scenario, BUDGETS_EVAL, hidden_draw
from fp.planner import Context, ValueNet, plan_value, plan_proxy, cem_search, BASELINES
from fp.env import PHYSICS as VERSION
from fidelity import fidelity_case
import firedata as fd

ROOT = _paths.WORK
SUF = '' if VERSION == 'p1' else '_' + VERSION        # p1 keeps the shipped names
# 27 Nov 2026: FIREPILOT_EXAM_TAG (e.g. v1.1) gives a new version its own exam folder, summary and widen file, so its
# per-fire files can never be mixed with v1.0's (stage A and B resume per fire). Every start is logged in exam.started.
_TAG = os.environ.get('FIREPILOT_EXAM_TAG', '')
SUF = SUF + ('_' + _TAG if _TAG else '')
OUT = os.path.join(ROOT, 'results', 'exam' + SUF)
SEED = 20261121


def exam_fires():
    d = json.load(open(_paths.FIRE_DATA + '/proc/baselines/replay_subset.json'))
    return list(d['fires'])


def stage_b_order():
    """Seeded order of the 48 fires for stage B (MPC + perfect-foresight bound), fixed before the exam ran."""
    return [int(i) for i in np.random.default_rng(SEED).permutation(len(exam_fires()))]


def run_fire(k, fid, net, out=None, exam_flag=True):
    """Stage A (every fire): fidelity, no action, the five fixed baselines, r0, FirePilot."""
    OUT_ = out or OUT
    fn = os.path.join(OUT_, fid + '.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    t0 = time.time()
    rs = np.random.default_rng(SEED + k)
    hid = hidden_draw(rs)
    rec = dict(fire_id=fid, k=k, hidden=hid, budgets={})
    try:
        S0 = real_scenario(fid, BUDGETS_EVAL[0], dict(f_ros=1.0, f_ws=1.0, d_wd=0.0), for_exam_scoring=exam_flag)
    except Exception as e:
        rec['skip'] = repr(e)
        json.dump(rec, open(fn, 'w'))
        return rec
    rec['fidelity'] = fidelity_case(S0)
    rec['a0'] = S0.a0()
    rec['shape'] = list(S0.burned0.shape)
    rec['edge_window'] = S0.meta.get('edge_window')
    del S0
    for b in BUDGETS_EVAL:
        S = real_scenario(fid, b, hid, for_exam_scoring=exam_flag)
        ctx = Context.from_scenario(S)
        rr = dict(area_none=S.none()['area'], a0=S.a0(), rows={})
        plans = {k2: dict(p) for k2, p in BASELINES.items()}
        t1 = time.time()
        plans['r0'], _ = plan_proxy(ctx, np.random.default_rng(k))
        rr['r0_plan_s'] = time.time() - t1
        t1 = time.time()
        plans['firepilot'], info = plan_value(ctx, net, np.random.default_rng(k))
        rr['firepilot_plan_s'] = time.time() - t1
        rr['firepilot_pred'], rr['firepilot_sigma'] = info['pred'], info['sigma']
        masks = {}
        for name, p in plans.items():
            ev = S.evaluate(ctx.fr, p, return_arr=True)
            a = ev.pop('arr')
            rr['rows'][name] = dict(plan=p, **ev)
            masks[name] = np.packbits(np.isfinite(a) & (a <= S.horizon))
        if b == BUDGETS_EVAL[0]:
            np.savez_compressed(os.path.join(OUT_, fid + '_ctx.npz'), burned0=ctx.burned0, arr_f=ctx.arr_f,
                                fl_f=ctx.fl_f, cost=ctx.cost, clock_t=S.clock[0], head=ctx.head,
                                fuel=np.asarray(S.ls['fuel'], np.int16), fl_t=S.none()['fl'].astype(np.float32),
                                none72=np.packbits(np.isfinite(S.none()['arr']) & (S.none()['arr'] <= S.horizon)),
                                arr_t=np.nan_to_num(S.none()['arr'], nan=-1).astype(np.float32),
                                obs=np.array([np.packbits(o['mask']) for o in S.meta['obs']]) if S.meta['obs'] else np.zeros(0),
                                obs_h=np.array([o['h'] for o in S.meta['obs']]))
        np.savez_compressed(os.path.join(OUT_, f'{fid}_{int(b)}_masks.npz'), clock_w=S.clock[1], **masks)
        rec['budgets'][str(int(b))] = rr
        print('A', fid, int(b), ' '.join('%s=%.2f' % (n[:6], r['saved']) for n, r in rr['rows'].items()), flush=True)
    rec['wall_s'] = time.time() - t0
    json.dump(rec, open(fn + '.tmp', 'w'), default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    os.replace(fn + '.tmp', fn)
    return rec


def run_fire_b(k, fid, out=None, exam_flag=True):
    """Stage B (seeded order, as compute allows): forecast-world CEM (MPC, 2 x 6 per mode + seeds) and the
    perfect-foresight CEM bound (truth world, 3 x 6 per mode, seeded with every other plan). Needs stage A."""
    OUT_ = out or OUT
    fn = os.path.join(OUT_, fid + '.B.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    A = json.load(open(os.path.join(OUT_, fid + '.json')))
    if 'skip' in A:
        return None
    t0 = time.time()
    rec = dict(fire_id=fid, k=k, budgets={})
    for b in BUDGETS_EVAL:
        bk = str(int(b))
        S = real_scenario(fid, b, A['hidden'], for_exam_scoring=exam_flag)
        ctx = Context.from_scenario(S)
        assert abs(S.none()['area'] - A['budgets'][bk]['area_none']) < 1e-6
        rr = dict(rows={})
        t1 = time.time()
        pm, mi = cem_search(S, ctx.fr, 'forecast', np.random.default_rng(k), iters=2, pop=6)
        rr['mpc_plan_s'], rr['mpc_sims'] = time.time() - t1, mi['n_sims']
        mz = dict(np.load(os.path.join(OUT_, f'{fid}_{bk}_masks.npz')))
        ev = S.evaluate(ctx.fr, pm, return_arr=True)
        a = ev.pop('arr')
        rr['rows']['mpc_forecast'] = dict(plan=pm, **ev)
        mz['mpc_forecast'] = np.packbits(np.isfinite(a) & (a <= S.horizon))
        t1 = time.time()
        init = [r['plan'] for r in A['budgets'][bk]['rows'].values()] + [pm]
        pf, pi = cem_search(S, ctx.fr, 'truth', np.random.default_rng(k + 1), iters=3, pop=6, init=init)
        ev = S.evaluate(ctx.fr, pf, return_arr=True)
        a = ev.pop('arr')
        rr['rows']['pf_bound'] = dict(plan=pf, n_sims=pi['n_sims'], **ev)
        mz['pf_bound'] = np.packbits(np.isfinite(a) & (a <= S.horizon))
        rr['pf_s'] = time.time() - t1
        np.savez_compressed(os.path.join(OUT_, f'{fid}_{bk}_masks.npz'), **mz)
        rec['budgets'][bk] = rr
        print('B', fid, bk, 'mpc=%.2f pf=%.2f fp=%.2f' % (rr['rows']['mpc_forecast']['saved'], rr['rows']['pf_bound']['saved'],
              A['budgets'][bk]['rows']['firepilot']['saved']), '%.0fs' % (time.time() - t0), flush=True)
    rec['wall_s'] = time.time() - t0
    json.dump(rec, open(fn + '.tmp', 'w'), default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    os.replace(fn + '.tmp', fn)
    return rec


def merged(fid, out=None):
    """Stage A record with stage B rows merged in (if stage B ran for this fire)."""
    OUT_ = out or OUT
    A = json.load(open(os.path.join(OUT_, fid + '.json')))
    fb = os.path.join(OUT_, fid + '.B.json')
    A['stage_b'] = os.path.exists(fb)
    if A['stage_b']:
        B = json.load(open(fb))
        for bk, rr in B['budgets'].items():
            A['budgets'][bk]['rows'].update(rr['rows'])
            A['budgets'][bk].update({k: v for k, v in rr.items() if k != 'rows'})
    return A


def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, float)
    rs = np.random.default_rng(seed)
    m = [v[rs.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(recs, widen):
    """Per budget and period. Stage-A methods on every scored fire; stage-B methods (mpc_forecast, pf_bound) on the
    stage-B subset, with FirePilot and the baselines re-summarised on that same subset ('B|' keys)."""
    per = fd.fires('exam').set_index('Event_ID')['exam_period']
    out = {}
    for b in BUDGETS_EVAL:
        bk = str(int(b))
        for period in ('all', 'early', 'main', 'recent'):
            for sub in ('A', 'B'):
                rs_ = [r for r in recs if 'skip' not in r and (period == 'all' or per.get(r['fire_id']) == period)
                       and (sub == 'A' or r.get('stage_b'))]
                if not rs_:
                    continue
                names = list(rs_[0]['budgets'][bk]['rows'].keys())
                if sub == 'A':
                    names = [n for n in names if n not in ('mpc_forecast', 'pf_bound')]
                row = dict(n=len(rs_))
                for nme in ['none'] + names:
                    if nme == 'none':
                        v = [0.0] * len(rs_)
                        ha = [0.0] * len(rs_)
                    else:
                        v = [r['budgets'][bk]['rows'][nme]['saved'] for r in rs_]
                        ha = [r['budgets'][bk]['area_none'] - r['budgets'][bk]['rows'][nme]['area'] for r in rs_]
                    row[nme] = dict(saved_mean=float(np.mean(v)), saved_median=float(np.median(v)), ci95=boot_ci(v),
                                    ha_saved_total=float(np.sum(ha)))
                fp_ = np.array([r['budgets'][bk]['rows']['firepilot']['saved'] for r in rs_])
                for other in [n for n in names if n != 'firepilot']:
                    ov = np.array([r['budgets'][bk]['rows'][other]['saved'] for r in rs_])
                    row[f'firepilot_beats_{other}'] = float(np.mean(fp_ > ov + 1e-9))
                    row[f'firepilot_ties_{other}'] = float(np.mean(np.abs(fp_ - ov) <= 1e-9))
                    row[f'firepilot_minus_{other}_ci95'] = boot_ci(fp_ - ov)
                if sub == 'A':
                    pred = np.array([r['budgets'][bk]['firepilot_pred'] for r in rs_])
                    sig = np.array([r['budgets'][bk]['firepilot_sigma'] for r in rs_]) * widen
                    row['pred_mean'] = float(pred.mean())
                    row['bias_truth_minus_pred'] = float(np.mean(fp_ - pred))
                    row['coverage90'] = float(np.mean(np.abs(fp_ - pred) <= 1.645 * sig))
                    row['coverage50'] = float(np.mean(np.abs(fp_ - pred) <= 0.674 * sig))
                    best_fixed = np.max([[r['budgets'][bk]['rows'][n]['saved'] for n in BASELINES] for r in rs_], 1)
                    row['oracle_best_fixed_per_fire'] = float(best_fixed.mean())
                else:
                    pfv = np.array([r['budgets'][bk]['rows']['pf_bound']['saved'] for r in rs_])
                    row['firepilot_share_of_pf'] = float(fp_.mean() / max(pfv.mean(), 1e-9))
                out[f'{sub}|{bk}|{period}'] = row
    return out


if __name__ == '__main__':
    if sys.argv[1] == '--dryrun-dev':
        # code test on ONE DEV fire (never an exam fire): same per-fire pipeline, output to tmp/dryrun_dev
        fid = sys.argv[2]
        assert not fd.guard.is_exam(fid), 'dry run is for dev fires only'
        out = os.path.join(ROOT, 'tmp', 'dryrun_dev')
        os.makedirs(out, exist_ok=True)
        rec = run_fire(0, fid, ValueNet(os.path.join(ROOT, 'models', sys.argv[3] + '.npz')), out=out, exam_flag=False)
        run_fire_b(0, fid, out=out, exam_flag=False)
        m = merged(fid, out)
        print(json.dumps({b: {k: round(v['saved'], 3) for k, v in r['rows'].items()} for b, r in m['budgets'].items()}))
        print(list(summarize([m], 1.0).keys()))
        sys.exit(0)
    stage, model = sys.argv[1], sys.argv[2]           # stage: A | B | summary
    shard, nshard = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    os.makedirs(OUT, exist_ok=True)
    if _TAG and stage in ('A', 'B'):
        import hashlib, datetime
        mp = os.path.join(ROOT, 'models', model + '.npz')
        with open(os.path.join(OUT, 'exam.started'), 'a') as fh:
            fh.write(f"{datetime.datetime.utcnow().isoformat()}Z stage={stage} shard={shard}/{nshard} model={model} "
                     f"sha256={hashlib.sha256(open(mp, 'rb').read()).hexdigest()} physics={VERSION}
")
    fires = exam_fires()
    if stage == 'A':
        net = ValueNet(os.path.join(ROOT, 'models', model + '.npz'))
        for k, fid in enumerate(fires):
            if k % nshard != shard:
                continue
            try:
                run_fire(k, fid, net)
            except Exception:
                print('FAIL', fid, traceback.format_exc()[-1500:], flush=True)
        print('stage A shard done', flush=True)
    elif stage == 'B':
        order = stage_b_order()
        for j, k in enumerate(order):
            if j % nshard != shard:
                continue
            if os.path.exists(os.path.join(OUT, fires[k] + '.B.json.stop')):
                break
            try:
                run_fire_b(k, fires[k])
            except Exception:
                print('FAIL', fires[k], traceback.format_exc()[-1500:], flush=True)
        print('stage B shard done', flush=True)
    elif stage == 'summary':
        # Stage B is a prefix of the seeded order per shard; the summary uses whatever stage B fires finished.
        assert all(os.path.exists(os.path.join(OUT, f + '.json')) for f in fires), 'stage A incomplete'
        recs = [merged(f) for f in fires]
        cal = json.load(open(_paths.latest(f'results/sigma_widen{SUF}.json')))
        from fidelity import summarize as fsum
        fid_sum = fsum([r['fidelity'] for r in recs if 'fidelity' in r])
        summ = dict(model=model, physics=VERSION, widen=cal['widen'], n_fires=len(fires), n_scored=sum('skip' not in r for r in recs),
                    n_stage_b=sum(bool(r.get('stage_b')) for r in recs),
                    stage_b_fires=[r['fire_id'] for r in recs if r.get('stage_b')],
                    skipped={r['fire_id']: r['skip'] for r in recs if 'skip' in r},
                    planner=summarize(recs, cal['widen']), fidelity=fid_sum)
        json.dump(summ, open(os.path.join(ROOT, 'results', f'exam_summary{SUF}.json'), 'w'), indent=1)
        print(json.dumps({k: {m: (v[m]['saved_mean'] if isinstance(v.get(m), dict) else v[m]) for m in v
                              if isinstance(v.get(m), dict) or m in ('n', 'coverage90')} for k, v in summ['planner'].items()}, indent=0)[:6000])
