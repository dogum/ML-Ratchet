"""BeatNet on four real exam excerpts (MIT-BIH DS2, 60 s each, drawn by a fixed seed within strata: a record rich in
V beats, one in S beats, one with fusion beats, one normal). None was used in training or choosing.
NOT A MEDICAL DEVICE."""
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from beatnet import BeatNet
net = BeatNet(); C = net.meta['classes']
for f in sorted(glob.glob(os.path.join(HERE, 'examples', '*.npz'))):
    z = np.load(f); m = json.loads(str(z['meta'])); out = net.classify(z['ecg'], m['fs'], z['beats'])
    ref = z['ref']; show = (z['beats'] >= z['show'][0]) & (z['beats'] < z['show'][1]); lab = out['label'][show]; r = ref[show]
    print(f"\nMIT-BIH record {m['record']}, {m['start_s'] / 60:.1f} min in ({m['why']}); heart rate {out['heart_rate_bpm']:.0f} bpm, median QRS {np.median(out['qrs_ms'][show]):.0f} ms")
    print('  reference:', ''.join(r)); print('  BeatNet:  ', ''.join(lab))
    for c in 'NSVF':
        n = int((r == c).sum())
        if n: print(f"  {c}: {n:3d} beats, BeatNet found {int(((r == c) & (lab == c)).sum()):3d}; BeatNet called {int((lab == c).sum()):3d} beats {c}")
