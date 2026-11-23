"""Dev-only calibration of the released rung (widen-only), written to calibration.json BEFORE the exam.
- widen[k] for Mw (k=0) and the along-strike centroid (k=1): the 68th percentile over dev events of
  |median - truth| / (68% half-width on that side), floored at 1 (widen only, never narrow).
- widen['amp'] for posterior-predictive log10 amplitudes at held-out dev DARTs, the same way.
- delay_frac: the travel-time delay of the physics version in use (from the twin test), applied to forecasts.
- dumb baseline: the dev-median Mw and the dev-median held-out amplitude."""
import os, sys, json, numpy as np
from _paths import ROOT, DATA
import tsim, evallib, dev_eval
rung = sys.argv[1]; physics = sys.argv[2] if len(sys.argv) > 2 else 'p0'
import simbank; PH = simbank.load_phys()[physics]; delay = PH.get('delay_frac', 0.0); gain = PH.get('amp_gain', 1.0)
geo = tsim.Geometry(); gf = tsim.GFBank(); net = evallib.load_net(rung)
summ, rows = dev_eval.evaluate(net, gf, geo, delay_frac=delay, n=1000, gain=gain)
def factor(err, lo_w, hi_w):
    # err = truth - median; truth below median -> compare with the lower half-width (fixed 22 Nov 2026: the release's
    # factors were computed with the two sides swapped; see NOTEBOOK.md)
    z = [abs(e) / (l if e < 0 else h) for e, l, h in zip(err, lo_w, hi_w) if (l if e < 0 else h) > 0]
    return float(max(1.0, np.percentile(z, 68))) if z else 1.0
e0 = [r['mw_true'] - r['mw'] for r in rows]
f0 = factor(e0, [r['mw'] - r['mw_lo'] for r in rows], [r['mw_hi'] - r['mw'] for r in rows])
rs = [r for r in rows if r['sc_true'] is not None]
f1 = factor([r['sc_true'] - r['sc'] for r in rs], [r['sc'] - r['sc_lo'] for r in rs], [r['sc_hi'] - r['sc'] for r in rs])
A = [a for r in rows for a in r.get('amp', [])]
ea = [np.log10(a['obs']) - np.log10(a['med']) for a in A]
fa = factor(ea, [np.log10(a['med'] / a['lo']) for a in A], [np.log10(a['hi'] / a['med']) for a in A])
cal = dict(rung=rung, physics=physics, widen={'0': f0, '1': f1, 'amp': fa}, delay_frac=delay, amp_gain=gain, amp_model_sd=0.0,
           dumb=dict(mw=float(np.median([r['mw_true'] for r in rows])), amp=float(np.median([a['obs'] for a in A]))),
           dev_summary=summ, n_dev_events=len(rows), n_dev_targets=len(A), method=__doc__.strip())
json.dump(cal, open(os.path.join(ROOT, 'calibration.json'), 'w'), indent=1)
print(json.dumps(cal, indent=1))
