// JS simulator vs Python (src/webheart.py reference beats)
global.atob = (s) => Buffer.from(s, 'base64').toString('binary');
global.BNet = require('./bnet.js'); const BS = require('./beatsim.js'), fs = require('fs');
const pk = JSON.parse(fs.readFileSync(__dirname + '/heart.json')), ref = JSON.parse(fs.readFileSync(__dirname + '/heart_ref.json'));
let t = Date.now(); BS.init(pk); console.log('init ms', Date.now() - t);
function cmp(nm, a0, b) { const M = pk.M, a = new Float64Array(a0.length); for (let i = 0; i < a0.length / 3; i++) for (let r = 0; r < 3; r++) a[3 * i + r] = M[r][0] * a0[3 * i] + M[r][1] * a0[3 * i + 1] + M[r][2] * a0[3 * i + 2]; const n = Math.min(a.length / 3, b.length); let se = 0, sb = 0, sab = 0, sa2 = 0, sb2 = 0;
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) { const x = a[3 * i + c], y = b[i][c]; se += (x - y) ** 2; sb += y * y; } console.log(nm, 'len js', a.length / 3, 'py', b.length, 'rel rms', Math.sqrt(se / sb).toExponential(3)); }
t = Date.now(); const tn = BS.vmap({ kind: 'normal', bbb: 'none', myo: ref.myo, fast: ref.fast }); console.log('dijkstra+map ms', Date.now() - t);
t = Date.now(); cmp('normal', BS.vbeat(tn, 0.8, 1), ref.normal); console.log('dipole ms', Date.now() - t);
const tv = BS.vmap({ kind: 'pvc', focus: ref.focus, myo: ref.myo, fast: ref.fast, engage: 3.0 }); cmp('pvc', BS.vbeat(tv, 0.6, 1), ref.pvc);
cmp('atrial', BS.abeat(pk.sinus), ref.atr); console.log('qrs width js', BS.qrsWidth(tn), BS.qrsWidth(tv), 'py', ref.qw_n, ref.qw_v);
t = Date.now(); const s = BS.strip({ type: 'V', focus: ref.focus, afocus: pk.sinus, myo: ref.myo, engage: 3, bbb: 'none', frontal: 0, coupling: 0.6, hr: 72, noise: 'all', snr: 25, seed: 3 }); console.log('strip ms', Date.now() - t, 'n', s.x.length, 'ann', s.ann.slice(0, 10), 'labels', s.labels.join(''));
