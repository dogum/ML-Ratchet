"""Final exam of v1.1 (rung r9_align), run once after the release choice, next to the release re-scored on the same rows.

    python rescore_v11.py actual      # work/data/banks/real_exam.npz -> work/data/exam/{release,v1.1}.json
    python rescore_v11.py forecast    # forecast weather 2024-2025, the shipped fetch list (rescore_forecast 'full')

Demand and wind are v1.0 unchanged. Solar: the reporting offset of the 168 h history is read at issue from the inputs
(demandcast.solar_offset, counted from each grid's dev reference in demandcast_meta.json), the history is put on the
physics clock, the r8 network runs, and the forecast is moved by the offset. exam.main() runs unchanged with rescore_release.py's numpy stand-in for evaluate.py; sigma scales are
re-fitted on the dev bank for each version, as exam.py does.
"""
import os
import sys

import numpy as np

from common import DATA, jdump
import rescore_release as RR      # installs the numpy evaluate shim and imports exam
import demandcast as DC

exam = RR.exam
predict_v10 = RR.predict


def predict_v11(W, B):
    out = predict_v10(W, B)
    X = np.nan_to_num(B['xs'].astype(np.float32)).astype(np.float64)
    ref = DC.DemandCast().offset_ref(B['meta_ba'], len(X))     # each grid's dev reference (demandcast_meta.json)
    r, _ = DC.solar_offset(X, ref)
    Xa = DC.align_solar_x(X, r)
    mu, sig = DC._forward(W['s'], Xa, 's', B['ps'].astype(np.float32))
    mu = np.clip(mu, 0, 1.2)
    night = B['night'] > 0.5
    mu = np.where(night, 0.0, mu); sig = np.where(night, DC.FLOOR['s'], sig)
    mu = DC._shift(mu, r, 0.0); sig = DC._shift(sig, r, DC.FLOOR['s'])
    out['s'] = (mu.astype(np.float32), sig.astype(np.float32))
    return out


def both(data):
    exam.DATA = data
    for name, fn in (('release', predict_v10), ('v1.1', predict_v11)):
        RR.shim.predict = fn
        exam.main(name)
    B = dict(np.load(f'{data}/banks/real_exam.npz'))
    r, _ = DC.solar_offset(np.nan_to_num(B['xs'].astype(np.float64)), DC.DemandCast().offset_ref(B['meta_ba'], len(B['xs'])))
    yr = np.array([d[:4] for d in B['meta_day']])
    off = {ba: {y: [int(((B['meta_ba'] == ba) & (yr == y) & B['hass'] & (B['meta_fc'] == 0) & (r == k)).sum())
                    for k in (-1, 0, 1)] for y in sorted(set(yr))} for ba in sorted(set(B['meta_ba']))}
    jdump(off, f'{data}/exam/v1.1_offsets_read.json')
    print('offsets read (-1/0/+1 days, actual-weather rows):', off)


if __name__ == '__main__':
    if sys.argv[1] == 'actual':
        both(DATA)
    else:
        import rescore_forecast as RF
        os.environ['DC_OPENMETEO'] = RF.scope_dir('full')
        sub = f'{DATA}/forecast_full'
        os.makedirs(f'{sub}/banks', exist_ok=True); os.makedirs(f'{sub}/exam', exist_ok=True)
        if not os.path.exists(f'{sub}/banks/real_exam.npz'):
            import shutil
            shutil.copy2(f'{DATA}/banks/real_dev.npz', f'{sub}/banks/real_dev.npz')
            RF.realbank.DATA = sub
            RF.realbank.build('exam')
        both(sub)
