// JS == Python check: features + network on the page's embedded real examples, against the Python release module's outputs.
// Usage: node web/test_js.js  (reads web/weights.json, web/data_auto.json, web/py_ref.json written by src/py_ref.py)
const fs = require('fs'), BNet = require('./bnet.js');
const W = JSON.parse(fs.readFileSync(__dirname + '/weights.json')), D = JSON.parse(fs.readFileSync(__dirname + '/data_auto.json'));
const ref = JSON.parse(fs.readFileSync(__dirname + '/py_ref.json')), net = BNet.load(W); let worstF = 0, worstO = 0;
D.examples.slice(0, ref.length).forEach((e, i) => {
  const b = Buffer.from(e.x, 'base64'), q = new Int16Array(b.buffer, b.byteOffset, b.length / 2), x = Float64Array.from(q, (v) => v / 32767 * e.q);
  const f = BNet.features(x, e.geom, e.f_nom), o = BNet.forward(net, f.F, Float32Array.from(f.sc));
  let dF = 0; for (let j = 0; j < f.F.length; j++) dF = Math.max(dF, Math.abs(f.F[j] - ref[i].F[j]));
  let dO = 0; for (let j = 0; j < 8; j++) dO = Math.max(dO, Math.abs(o[j] - ref[i].o[j]));
  worstF = Math.max(worstF, dF); worstO = Math.max(worstO, dO); console.log(e.short, 'max|dF|', dF.toExponential(2), 'max|dOut|', dO.toExponential(2));
});
console.log('worst feature diff', worstF.toExponential(2), 'worst output diff', worstO.toExponential(2));
fs.writeFileSync(__dirname + '/../checks/js_parity.json', JSON.stringify({ n: ref.length, worst_feature_abs: worstF, worst_output_abs: worstO }));
