"""Round 2: the round-1 Mondrian conformal wrapper (arms_filt/conformal/fit_wrapper.py) around a round-2 arm.

usage: python fit_conformal2.py <base_arm_dir> <dev_preds.npz>
  <dev_preds.npz> = evaluate.py --save-preds output of the REGISTERED base arm (dev, ungauged).
The method is unchanged: arms_filt/conformal/lib/cf_mondrian.py (fit_job, apply, strata, offset) and the strata/offsets
of arms_filt/conformal/dev_meta.json, both read-only. Differences from fit_wrapper.py: the base must be in round 2's
registry (register2.py), the wrapper is written to round2/arms/<base_id>_conformal/, and its predict.py finds the
library and the harness by their round-1 paths. Ungauged only. No exam data.
"""
import json
import os
import sys
import time

import r2
import numpy as np

CONF = os.path.join(r2.BAKEOFF, 'arms_filt', 'conformal')
sys.path.insert(0, os.path.join(CONF, 'lib'))
import cf_mondrian as CF  # noqa: E402
import score as S  # noqa: E402
import armlib  # noqa: E402
import register2 as REG2  # noqa: E402

TEMPLATE = '''"""Round-2 conformal wrapper: {base_id} with Mondrian split-conformal bands (round 1's layer, unchanged).

The wrapped arm's predict() runs unchanged; its 5-95 % and 25-75 % band endpoints are shifted in log space per basin
stratum (snow / arid / humid / other), with shifts fitted on {base_id}'s dev residuals only (calib.json; method in
arms_filt/conformal/lib/cf_mondrian.py). mean and q50 are untouched. numpy only (plus the wrapped arm).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = json.load(open(os.path.join(HERE, 'calib.json')))
_LIB = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'arms_filt', 'conformal', 'lib'))
_HARNESS = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'harness'))
for _p in (_LIB, _HARNESS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import cf_mondrian as CF  # noqa: E402
import armlib  # noqa: E402

BASE = os.path.normpath(os.path.join(HERE, CAL['base_rel']))
_base_predict, _base_mod = armlib.load_predict(BASE)


def _wrap(b, o):
    sh = CAL['jobs'][CF.job_key(b)][CF.stratum(b['classes'])]
    return CF.apply(o, sh, CF.offset(b['static']))


def predict(batch):
    return _wrap(batch, _base_predict(batch))


def predict_many(batches):
    if hasattr(_base_mod, 'predict_many'):
        outs = _base_mod.predict_many(batches)
    else:
        outs = [_base_predict(b) for b in batches]
    return [_wrap(b, o) for b, o in zip(batches, outs)]
'''


def basin_scores(P, o=None):
    y = P[:, 0].astype(float)
    m = P[:, 1].astype(float)
    Q = P[:, 2:7].astype(float) if o is None else np.stack([o[k] for k in CF.QK], 1)
    return S.interval_scores(y, m, np.sort(Q, 1))


def to_out(P):
    return {'mean': P[:, 1].astype(float), **{k: P[:, 2 + j].astype(float) for j, k in enumerate(CF.QK)}}


def main(base, preds_file):
    t0 = time.process_time()
    base = os.path.abspath(base)
    meta = armlib.read_meta(base)
    bid = meta['arm_id']
    regs = REG2.registered()
    if bid not in regs:
        sys.exit(f'{bid} is not in the round-2 registry: wrap only frozen arms')
    ok, h, changed = REG2.verify(regs[bid])
    if not ok:
        sys.exit(f'{bid} changed since registration: {changed}')
    DM = json.load(open(os.path.join(CONF, 'dev_meta.json')))
    mt = {b: (DM[b]['stratum'], DM[b]['c']) for b in DM}
    Z = np.load(preds_file)
    preds = {b: Z[b] for b in Z.files}
    sh = CF.fit_job(preds, mt)
    jobs = {'ungauged': {s: sh.get(s, sh['pooled']) for s in CF.STRATA + ['pooled']}}
    rows = []
    for b, P in preds.items():
        raw = basin_scores(P)
        loo = CF.fit_job({k: v for k, v in preds.items() if k != b}, mt)
        out = basin_scores(P, CF.apply(to_out(P), loo.get(mt[b][0], loo['pooled']), mt[b][1]))
        rows.append({'basin': b, 'stratum': mt[b][0], **{f'raw_{k}': raw[k] for k in ('cov50', 'cov90', 'crps_rel')},
                     **{f'loo_{k}': out[k] for k in ('cov50', 'cov90', 'crps_rel')}})
    R = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k not in ('basin', 'stratum')}
    wid = f'{bid}_conformal'
    wdir = os.path.join(r2.ROUND2, 'arms', wid)
    os.makedirs(os.path.join(wdir, 'work'), exist_ok=True)
    base_rel = os.path.relpath(base, wdir).replace('\\', '/')
    extra = [base_rel, '../../../arms_filt/conformal/lib']
    for x in meta.get('extra_files', []):
        extra.append(os.path.relpath(os.path.normpath(os.path.join(base, x)), wdir).replace('\\', '/'))
    calib = {'base_id': bid, 'base_hash': h, 'base_rel': base_rel,
             'method': 'Mondrian split-conformal (asymmetric CQR in log space), strata snow/arid/humid/other, basins '
                       'weighted equally (round 1 arms_filt/conformal/lib/cf_mondrian.py)',
             'c_frac': CF.C_FRAC, 'min_basins': CF.MIN_BASINS, 'jobs': jobs}
    json.dump(calib, open(os.path.join(wdir, 'calib.json'), 'w'), indent=1)
    open(os.path.join(wdir, 'predict.py'), 'w').write(TEMPLATE.replace('{base_id}', bid))
    fit_h = (time.process_time() - t0) / 3600
    open(os.path.join(wdir, 'card.md'), 'w').write(
        f"# {wid}: {meta.get('label', bid)} + Mondrian conformal bands (River Bake-off round 2)\n\n"
        f"The registered round-2 arm `{bid}` (hash `{h[:16]}`) run unchanged; its 5-95 % and 25-75 % bands are "
        f"recalibrated by round 1's Mondrian split-conformal layer (`arms_filt/conformal/lib/cf_mondrian.py`, "
        f"unchanged; strata snow / arid / humid / other; shifts in log space), fitted on `{bid}`'s daily dev "
        f"predictions (52 dev basins, WY2001-2014). Mean and q50 are the base arm's, so NSE, KGE and FHV are "
        f"identical; only coverage, CRPS and band width change. No exam data.\n\n"
        f"Dev, mean over basins: cov50 {R['raw_cov50']:.2f} -> {R['loo_cov50']:.2f} (leave-one-basin-out), "
        f"cov90 {R['raw_cov90']:.2f} -> {R['loo_cov90']:.2f}; CRPS/Q {R['raw_crps_rel']:.3f} -> "
        f"{R['loo_crps_rel']:.3f}.\n\nCompute: fit {fit_h:.4f} CPU-h. Limits: as round 1's wrappers (52 dev "
        f"basins, 8-22 per stratum; days within a basin are not exchangeable).\n")
    rj = {'arm_id': wid, 'label': f"{meta.get('label', bid)} + Mondrian conformal", 'family': 'conformal',
          'family_label': f'Conformal prediction (wrapper around {bid})', 'group': 'round2',
          'protocols': ['ungauged'], 'wraps': bid, 'wraps_hash': h, 'numpy_only': bool(meta['numpy_only']),
          'numpy_only_reason': '', 'compute': {'train_cpu_hours': round(fit_h, 4),
                                               'notes': f'conformal fit only on dev predictions of {bid}'},
          'feature_engineering': f"none beyond {bid}'s; the offset uses static[11] and the class list (strata)",
          'extra_files': extra, 'margin': 0.0, 'compare_to': [f'{c}_conformal' for c in (meta.get('compare_to') or [])
                                                              if not c.startswith('ens_')] or None,
          'rule': 'No ladder: one conformal fit on dev residuals (method fixed in advance).',
          'rungs': [{'id': 'c0', 'change': 'Mondrian split-conformal bands fitted on dev residuals',
                     'decision': 'released', 'why': f"dev cov90 (mean over basins, LOBO) {R['loo_cov90']:.2f}"}],
          'released': 'c0'}
    json.dump(rj, open(os.path.join(wdir, 'ratchet.json'), 'w'), indent=1)
    json.dump({'mean_over_basins': R, 'basins': rows}, open(os.path.join(wdir, 'work', 'fit_report.json'), 'w'),
              indent=1, default=float)
    print(wid, json.dumps(R))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
