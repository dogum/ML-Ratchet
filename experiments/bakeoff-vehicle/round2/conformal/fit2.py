"""Round 2, C: v1.0's Mondrian conformal wrapper (arms_phys/conformal/fit_wrapper.py, imported, not edited) around a
round-2 arm registered in round2/registry. Dev residuals only.

python fit2.py <round-2 arm dir> <dev_preds.parquet>    -> round2/arms/<arm_id>_conformal/ (then register2.py it)
The dev predictions are the arm's own evaluate.py output for its released folder (same hash). The method, strata,
template and card are v1.0's; the wrapper's lib is a byte copy of v1.0's (round2/arms/lib/vcf_mondrian.py).
"""
import importlib.util
import json
import os
import shutil
import sys
import time

R2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R2)
import paths as P  # noqa: E402
import config as C  # noqa: E402
C.REGISTRY = P.REGISTRY
P.setup()
import pandas as pd  # noqa: E402

V1 = os.path.join(P.BAKEOFF, 'arms_phys', 'conformal')
_s = importlib.util.spec_from_file_location('v1_fit_wrapper', os.path.join(V1, 'fit_wrapper.py'))
FW = importlib.util.module_from_spec(_s); _s.loader.exec_module(FW)
P.patch_registry(FW.REG)
FW.DEV_STRATA = os.path.join(P.WORK, 'dev_strata.json')
LIB = os.path.join(P.ARMS, 'lib')


def main(base, devp):
    os.makedirs(LIB, exist_ok=True)
    if not os.path.exists(os.path.join(LIB, 'vcf_mondrian.py')):
        shutil.copy(os.path.join(V1, 'lib', 'vcf_mondrian.py'), os.path.join(LIB, 'vcf_mondrian.py'))
    t0 = time.process_time()
    base = os.path.abspath(base)
    meta = FW.armlib.read_meta(base)
    bid = meta['arm_id']
    regs = FW.REG.registered()
    if bid not in regs:
        sys.exit(f'{bid} is not registered in round2/registry')
    ok, h, ch = FW.REG.verify(regs[bid])
    if not ok:
        sys.exit(f'{bid} changed since registration: {ch}')
    tracks = list(meta['tracks'])
    wid = f'{bid}_conformal'; wdir = os.path.join(P.ARMS, wid)
    os.makedirs(os.path.join(wdir, 'work'), exist_ok=True)
    base_rel = os.path.relpath(base, wdir)
    calib = {'base_id': bid, 'base_hash': h, 'base_rel': base_rel, 'method': 'Mondrian split-conformal (asymmetric CQR '
             'in log space), strata powertrain x road/dyno/hd (v1.0 arms_phys/conformal)', 'min_units': FW.CF.MIN_UNITS,
             'tracks': {}}
    reports = {}
    for t in tracks:
        Fr = pd.read_parquet(devp)
        shifts, rep = FW.fit_track(Fr)
        calib['tracks'][t] = {'shifts': shifts}; reports[t] = rep
        m = [rep[x]['D16_mean'] for x in ('raw', 'in_sample', 'cross_fit')]
        print(f"{bid} [{t}] D16-mean cov50/cov90 raw {m[0]['cov50']:.3f}/{m[0]['cov90']:.3f} -> cv {m[2]['cov50']:.3f}/"
              f"{m[2]['cov90']:.3f}; crps {m[0]['crps_log']:.4f} -> cv {m[2]['crps_log']:.4f}", flush=True)
    extra = [base_rel, '../lib'] + [os.path.relpath(os.path.normpath(os.path.join(base, x)), wdir)
                                    for x in meta.get('extra_files', [])]
    json.dump(calib, open(os.path.join(wdir, 'calib.json'), 'w'), indent=1)
    tpl = open(os.path.join(V1, 'wrapper_predict_template.py')).read().replace('{base_id}', bid)
    open(os.path.join(wdir, 'predict.py'), 'w').write(tpl)
    fit_h = (time.process_time() - t0) / 3600
    FW.write_card(wdir, wid, bid, meta, h, tracks, reports, calib, fit_h)
    cvs = ', '.join(f"{t} cov50/90 {reports[t]['cross_fit']['D16_mean']['cov50']:.2f}/"
                    f"{reports[t]['cross_fit']['D16_mean']['cov90']:.2f}" for t in tracks)
    rj = {'arm_id': wid, 'label': f"{meta.get('label', bid)} + Mondrian conformal", 'family': 'conformal',
          'family_label': f'Conformal prediction (wrapper around {bid})', 'group': 'round2', 'tracks': tracks,
          'wraps': bid, 'wraps_hash': h, 'numpy_only': bool(meta['numpy_only']), 'numpy_only_reason': '',
          'real_labels_in_training': False,
          'compute': {'train_cpu_hours': round(fit_h, 4), 'base_train_cpu_hours': meta.get('compute', {}).get('train_cpu_hours'),
                      'notes': f'conformal fit only on dev residuals of {bid}'},
          'feature_engineering': f"none beyond {bid}'s; strata from arch, cls and the dynamometer flag",
          'extra_files': extra, 'margin': 0.0,
          'rule': 'No ladder: one conformal fit on dev residuals (v1.0 method, fixed in advance).',
          'rungs': [{'id': 'c0', 'change': 'Mondrian split-conformal bands fitted on dev residuals (v1.0 method)',
                     'decision': 'released', 'why': 'the only rung; dev coverage over the D16 sets (grouped 10-fold cross-fit): ' + cvs}],
          'released': 'c0'}
    json.dump(rj, open(os.path.join(wdir, 'ratchet.json'), 'w'), indent=1)
    json.dump({'base_id': bid, 'tracks': reports}, open(os.path.join(wdir, 'work', 'fit_report.json'), 'w'), indent=1, default=float)
    print('wrote', wdir)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
