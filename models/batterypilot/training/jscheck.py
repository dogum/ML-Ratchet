"""JS == Python check of the network + offer executor (dev data, HB_HOUSTON and CAISO SP15).  python3 jscheck.py <rung>"""
import json, subprocess, sys
import numpy as np
import _paths  # noqa: E402  (this folder, GridSim and GridData on sys.path; outputs to WORK)
from model import PolicyNet, run_policy
from battery import Battery
from realdata import node_frame, fill
ROOT = _paths.WORK   # was the sandbox workspace grid/batterypilot
rung = sys.argv[1]
w = dict(np.load(f'{ROOT}/ckpt/{rung}.npz'))
cases = []
for m, n, s, e in [('ERCOT', 'HB_HOUSTON', '2019-07-01', '2019-09-01'), ('CAISO', 'TH_SP15_GEN-APND', '2023-09-01', '2023-11-01')]:
    f, _ = fill(node_frame(m, n, s, e)); f = f.dropna(subset=['rt', 'da'])
    p = {c: f[c].values for c in ['rt', 'da', 'df', 'T', 'hour', 'day', 'dow', 'doy']}
    for d in (1, 2, 4):
        bat = Battery(1.0, d)
        ch, dis, soc, V = run_policy(PolicyNet(f'{ROOT}/ckpt/{rung}.npz'), p, bat, bat.c_lin(p['T']))
        cases.append(dict(p={k: [float(x) for x in v] for k, v in p.items()}, d=d, soc=soc.tolist(), V=V.tolist()))
W = dict(x_mu=w['x_mu'].tolist(), x_sd=w['x_sd'].tolist(),
         layers=[dict(W=w[f'W{i}'].ravel().tolist(), b=w[f'b{i}'].tolist(), nin=w[f'W{i}'].shape[1], nout=w[f'W{i}'].shape[0]) for i in range(3)])
import os  # noqa: E402
os.makedirs(f'{ROOT}/web', exist_ok=True)
IN = os.path.join(ROOT, 'web', 'jscheck_in.json')            # was a sandbox scratch file
BPJS = os.path.join(_paths.MODEL, 'lab', 'bp.js')            # the release's web/bp.js, kept as lab/bp.js
json.dump(dict(W=W, cases=cases), open(IN, 'w'))
js = r"""
require(%s);
const I = JSON.parse(require('fs').readFileSync(%s));""" % (json.dumps(BPJS), json.dumps(IN)) + r"""
const out = [];
for (const c of I.cases) {
  const bat = BP.battery(c.d);
  const r = BP.runPolicy(I.W, c.p, bat, 0, c.p.rt.length, 0);
  let ms = 0, mv = 0;
  for (let i = 0; i < r.soc.length; i++) ms = Math.max(ms, Math.abs(r.soc[i] - c.soc[i]));
  for (let i = 0; i < r.V.length; i++) for (let k = 0; k < 5; k++) mv = Math.max(mv, Math.abs(r.V[i][k] - c.V[i][k]) / (1 + Math.abs(c.V[i][k])));
  out.push({d: c.d, max_abs_soc_diff: ms, max_rel_value_diff: mv, hours: c.p.rt.length});
}
console.log(JSON.stringify(out));
"""
r = subprocess.run(['node', '-e', js], capture_output=True, text=True)
print(r.stdout, r.stderr[:2000])
res = json.loads(r.stdout)
json.dump(dict(rung=rung, cases=res, note='dev data (ERCOT HB_HOUSTON Jul-Aug 2019, CAISO SP15 Sep-Oct 2023), full precision'),
          open(f'{ROOT}/web/jscheck.json', 'w'), indent=1)
