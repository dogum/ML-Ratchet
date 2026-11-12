"""Check the browser path end to end: app.js builds the 12 channels and sdnet.js runs the network; compare with the Python
release (sundoctor.channels + numpy network) on a real lab fortnight, with and without the sensor."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, re, subprocess, base64, numpy as np
sys.path.insert(0, _P.WORK + '/release'); sys.path.insert(0, _P.SRC)
import sundoctor as SD
W = _P.LAB + '/'
data = json.load(open(W + 'data.json')); lab = data['lab'][0]
f16 = lambda b: np.frombuffer(base64.b64decode(b), '<f2').astype(np.float64)
D = len(lab['dates']); sh = (D, 96)
p, g, ta, cz, ca, gc = [f16(lab[k]).reshape(sh) for k in ('p', 'g', 'ta', 'cz', 'ca', 'gc')]
g = np.where(g < -0.5, np.nan, g)
net = SD.SunDoctorNet(_P.WORK + '/release/sundoctor_weights.npz')
res = {}
app = open(W + 'app.js').read()
fn = re.search(r'function expected\(g, ta, ca\) \{.*?\n\}', app, re.S).group(0) + '\n' + re.search(r'function diagnose\(W, P, G, TA, ref, home\) \{.*?\n\}', app, re.S).group(0)
for home in (False, True):
    ref = lab['ref']['home' if home else 'poa']
    X = SD.channels(p, None if home else g, ta, cz, ca, gc, ref, home)[None]
    o = net.predict(X)
    js = f"""
const SunDoctorJS = require('{W}sdnet.js'); const WEIGHTS = require('{W}weights.json');
const NET = SunDoctorJS.load(WEIGHTS), META = NET.meta;
{fn}
const A = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const Wn = {{ dates: A.dates, _gc: A.gc, _ca: A.ca, _cz: A.cz }};
const o = diagnose(Wn, A.p, A.g, A.ta, A.ref, A.home);
console.log(JSON.stringify(o));
"""
    open(_P.WORK + '/scratch/parity.js', 'w').write(js)
    inp = json.dumps({'dates': lab['dates'], 'gc': gc.ravel().tolist(), 'ca': ca.ravel().tolist(), 'cz': cz.ravel().tolist(), 'p': p.ravel().tolist(),
                      'g': np.nan_to_num(g, nan=-1).ravel().tolist(), 'ta': ta.ravel().tolist(), 'ref': ref, 'home': home})
    out = json.loads(subprocess.run(['node', _P.WORK + '/scratch/parity.js'], input=inp, capture_output=True, text=True, check=True).stdout)
    err = max(float(np.max(np.abs(np.array(out[k]) - o[k][0]))) for k in ('pf', 'frac', 'shade', 'clip', 'soil', 'outage', 'snow', 'stuck', 'step_p'))
    res['home' if home else 'poa'] = err
print('max |JS - numpy| over all daily outputs:', res)
