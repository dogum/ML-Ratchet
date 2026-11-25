"""Write ratchet.json + card.md for one arm from train/decisions.json and the harness evaluations.
usage: python3 finalize.py <arm>
decisions.json[<arm>] = {arm_id, label, family, family_label, released, core (card text), assumption (card text),
                         rungs: [{id, change, decision, why}], noise_checks: [ids], notes: [..]}
Numbers (sim val / dev medians, CPU) are read from <arm>/work/eval/<rung>_summary.json and work/train_<rung>.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.path.dirname(HERE)
RULE = ('keep if sim val median NSE (all 800 frozen runs) > incumbent + 0.01 AND dev median NSE (52 dev basins, '
        'WY2001-2014) >= 0.95 x incumbent\'s')
FE = ('None beyond the shared arms_seq pipeline (common/seqlib.py), identical for all four sequence arms: FlowCast\'s '
      '11 scaled daily features (P/10, log1p P, PET/5, T_mean/T_min/T_max /10, SRAD/150, vp/1000, wind/3, sin/cos day '
      'of year) + FlowCast\'s 18-value static vector (batch["static"], standardised with the train bank mean/sd) '
      'repeated every day; target in units of the basin mean precipitation exp(static[11]); 730-day windows (365 '
      'warm-up + 365 target), inference in 365-day blocks each with its own 365-day warm-up (FlowCast). Causal: day t '
      'uses forcing up to day t only.')


def load(arm, rung):
    s = json.load(open(os.path.join(SEQ, arm, 'work', 'eval', f'{rung}_summary.json')))
    p = os.path.join(SEQ, arm, 'work', f'train_{rung}.json')
    t = json.load(open(p)) if os.path.exists(p) else {}
    return s, t


def nums(arm, rung):
    s, t = load(arm, rung)
    sv, dv = s['simval']['ungauged'], s['dev']['ungauged']
    fc = dv['vs'].get('flowcast_v1', {}).get('paired', {})
    return {'simval_nse': round(sv['nse'], 4), 'simval_kge': round(sv['kge'], 4),
            'simval_cov90': round(sv['cov90'], 3) if sv.get('cov90') is not None else None,
            'dev_nse': round(dv['nse'], 4), 'dev_kge': round(dv['kge'], 4), 'dev_fhv': round(dv['fhv'], 1),
            'dev_cov50': round(dv['cov50'], 3), 'dev_cov90': round(dv['cov90'], 3),
            'dev_crps_rel': round(dv['crps_rel'], 3),
            'dev_by_class_nse': {k: {'n': v['n'], 'nse': round(v['nse'], 3)} for k, v in (dv.get('by_class_nse') or {}).items()},
            'train_cpu_h': t.get('train_cpu_h'), 'n_params': t.get('n_params'), 'steps': (t.get('cfg') or {}).get('steps')}


def fmt(x, n=3):
    return '' if x is None else (f'{x:.{n}f}' if isinstance(x, float) else str(x))


def main(arm):
    D = json.load(open(os.path.join(HERE, 'decisions.json')))[arm]
    rungs, cpu = [], 0.0
    for r in D['rungs']:
        n = nums(arm, r['id'])
        cpu += n['train_cpu_h'] or 0.0
        rungs.append({**r, **{k: n[k] for k in ('simval_nse', 'dev_nse', 'simval_kge', 'dev_kge', 'dev_fhv', 'dev_cov50',
                                                'dev_cov90', 'dev_crps_rel', 'train_cpu_h', 'n_params', 'steps')}})
    noise = []
    for nid in D.get('noise_checks', []):
        n = nums(arm, nid)
        cpu += n['train_cpu_h'] or 0.0
        noise.append({'id': nid, **{k: n[k] for k in ('simval_nse', 'dev_nse', 'train_cpu_h')}})
    rel = nums(arm, D['released'])
    exp = json.load(open(os.path.join(SEQ, arm, 'work', 'export_check.json')))
    meta = {
        'arm_id': D['arm_id'], 'label': D['label'], 'family': D['family'], 'family_label': D['family_label'],
        'group': 'seq', 'protocols': ['ungauged'], 'numpy_only': True, 'numpy_only_reason': '',
        'compute': {'train_cpu_hours': round(cpu, 3),
                    'eval_note': 'harness evaluate.py per rung (800 sim val runs + 52 dev basins) not included: '
                                 'about 0.5-4 CPU-min per rung',
                    'notes': f'{len(D["rungs"])} rungs' + (f' + {len(noise)} seed re-runs (noise check)' if noise else '')
                             + '; one thread (torch.set_num_threads(1)), time.process_time() of the training process, '
                               'shared 2-core machine'},
        'feature_engineering': FE, 'extra_files': ['../common'], 'margin': 0.01, 'rule': RULE,
        'rungs': rungs, 'released': D['released'], 'noise_checks': noise,
        'released_scores': rel, 'n_params': rel['n_params'],
        'export_check': exp,
        'bands': 'raw pinball quantiles (no dev widening; arm C conformalises every arm)',
    }
    json.dump(meta, open(os.path.join(SEQ, arm, 'ratchet.json'), 'w'), indent=1)
    # ------------------------------------------------------------- card
    L = [f'# {D["label"]} (`{D["arm_id"]}`)', '',
         f'River Bake-off, group A (sequence learners). Family: **{D["family_label"]}** (`{D["family"]}` on the Model '
         'families page). Protocol: **ungauged** (forcing + static attributes only; never reads streamflow). numpy-only '
         'inference (`predict.py` -> `../common/seqlib.py` + `weights.npz`).', '',
         '## The built-in assumption', D['assumption'], '',
         '## Network', D['core'], f'Parameters: {rel["n_params"]:,} (the four sequence arms hold 24.7k-26.9k).', '',
         '## Shared with every sequence arm (only the core differs)',
         '- Inputs: ' + FE.split(': ', 1)[1], '- Head: Linear(hidden -> 6): mean = softplus; 5/25/50/75/95 % quantiles = '
         'cumulative softplus (monotone); units of p_mean, times p_mean -> mm/day. Bands are raw (no dev widening).',
         '- Loss: FlowCast\'s: NSE* on the mean with an input-only scale (Fu runoff ratio of the forcing aridity, '
         'w = 2.6) + the same-scaled pinball loss on the quantiles.',
         '- Training data: the harness bank (12,000 simulated runs x 4 years, physics p1, FlowCast\'s recipe), random '
         '730-day windows, the same rng seed so every arm sees the same windows in the same order; Adam, lr 2e-3 '
         'cosine to 1e-4, gradient clip 1.0, batch 32, last step used (no early stopping). Zero real labels.', '',
         '## Ratchet', f'Rule: {RULE}. Sim val = harness `evaluate.py`, all 800 frozen runs; dev = 52 dev basins.', '',
         '| Rung | Change | Steps | Sim val NSE | Dev NSE | Dev KGE | Dev FHV % | Dev cov50/90 | CPU h | Decision | Why |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rungs:
        L.append(f'| {r["id"]} | {r["change"]} | {r["steps"]} | {fmt(r["simval_nse"])} | {fmt(r["dev_nse"])} | '
                 f'{fmt(r["dev_kge"])} | {fmt(r["dev_fhv"], 1)} | {fmt(r["dev_cov50"], 2)}/{fmt(r["dev_cov90"], 2)} | '
                 f'{fmt(r["train_cpu_h"], 2)} | **{r["decision"]}** | {r["why"]} |')
    if noise:
        L += ['', 'Seed re-runs of r0 (noise check, not rungs): ' + '; '.join(
            f'{x["id"]}: sim {x["simval_nse"]:.3f}, dev {x["dev_nse"]:.3f}' for x in noise) + '.']
    bc = rel['dev_by_class_nse']
    L += ['', f'## Released rung {D["released"]}: dev gate (52 basins, WY2001-2014)',
          f'NSE {rel["dev_nse"]:.3f}, KGE {rel["dev_kge"]:.3f}, FHV {rel["dev_fhv"]:.1f} %, 50/90 % coverage '
          f'{rel["dev_cov50"]:.2f}/{rel["dev_cov90"]:.2f}, CRPS/mean flow {rel["dev_crps_rel"]:.3f}. By class (median '
          'NSE): ' + ', '.join(f'{k} {v["nse"]:.2f} (n {v["n"]})' for k, v in bc.items()) + '. FlowCast v1.0 as released: 0.425 (snow 0.45, '
          'arid 0.18, humid 0.56, regulated 0.28).', '',
          '## Compute', f'Training {meta["compute"]["train_cpu_hours"]:.2f} single-thread CPU-hours in all '
          f'({meta["compute"]["notes"]}). Budget: about 4.', '',
          '## numpy release check',
          f'numpy float64 (`predict.py`) vs the torch float32 checkpoint on {exp["cases"]} cases (16 sim val windows, 3 '
          f'dev basins x 14 years): max |diff| {exp["max_abs_diff_mean_rel_to_mean_flow"]:.1e} (mean) and '
          f'{exp["max_abs_diff_quantiles_rel_to_mean_flow"]:.1e} (quantiles) of the mean flow; against torch in float64 '
          f'{exp["torch_float64"]["max_abs_diff_mean_rel"]:.1e} / {exp["torch_float64"]["max_abs_diff_quantiles_rel"]:.1e}.',
          '', '## Limits'] + [f'- {x}' for x in D['limits']] + ['', '## Notes'] + [f'- {x}' for x in D.get('notes', [])]
    open(os.path.join(SEQ, arm, 'card.md'), 'w').write('\n'.join(L) + '\n')
    print(json.dumps({'arm': arm, 'cpu_h': meta['compute']['train_cpu_hours'], 'released': D['released'],
                      'sim': rel['simval_nse'], 'dev': rel['dev_nse']}))


if __name__ == '__main__':
    main(sys.argv[1])
