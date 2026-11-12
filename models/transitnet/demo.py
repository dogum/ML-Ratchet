"""TransitNet on three real TESS stars: a mini-Neptune, a super-Earth and a false positive (none in training or choosing)."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import transitnet as tn
net = tn.TransitNet()
for f in ('toi270c', 'pimenc', 'toi617'):
    z = np.load(os.path.join(HERE, 'examples', f + '.npz')); m = json.loads(str(z['meta']))
    secs = [dict(t=z[f't{j}'], f=z[f'f{j}'], c1=z[f'c1{j}'], c2=z[f'c2{j}']) for j in range(m['n_sectors'])]
    r = net.read(secs, m['period'], m['epoch'], m['duration_hours'], star=m['star'])
    print(f"\n{m['name']} ({m['n_sectors']} sectors): planet {100 * r['p_planet']:.0f}%  eclipsing binary {100 * r['p_eb']:.0f}%  background binary {100 * r['p_blend']:.0f}%  nothing {100 * r['p_nothing']:.0f}%")
    print(f"  Rp/R* {r['k']:.4f} ({r['k_lo']:.4f}-{r['k_hi']:.4f})  b {r['b']:.2f} ({r['b_lo']:.2f}-{r['b_hi']:.2f})  a/R* {r['aR']:.1f} ({r['aR_lo']:.1f}-{r['aR_hi']:.1f})  duration {r['T14_h']:.2f} h"
          + (f"  -> {r['Rp_earth']:.2f} Earth radii" if 'Rp_earth' in r else ''))
    pub = m['published']
    print('  published:', ', '.join(f'{k} {v:.4g}' for k, v in pub.items() if isinstance(v, (int, float)) and v is not None) if m['kind'] == 'planet' else 'follow-up: ' + pub['comments'])
