"""BearingNet demo: read a few real exam recordings (fixed random draw within classes; never used for training or for any choice).
Run:  python3 demo.py"""
import os, sys, json, time, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import bearingnet as BNet

if __name__ == '__main__':
    net = BNet.BearingNet(); t0 = time.time(); right = 0; files = sorted(glob.glob(os.path.join(HERE, 'examples', '*.npz')))
    print('%-34s %-6s | %-6s %-28s | %-14s | %s' % ('recording', 'truth', 'net', 'p(N, OR, IR, B)', 'rpm true/net', 'size mm true/net'))
    for f in files:
        z = np.load(f); m = json.loads(str(z['meta']))
        r = net.read(z['x'], fs=float(z['fs']), bearing=m['bearing'], rpm_nominal=m['rpm_nominal'])
        p = ' '.join('%.2f' % r['p'][c] for c in ('N', 'OR', 'IR', 'B')); right += r['condition'] == m['truth']
        print('%-34s %-6s | %-6s %-28s | %5.0f / %5.0f±%-3.0f | %s / %.2f' % (m['name'][:34], m['truth'], r['condition'], p, m['rpm_true'], r['rpm'], r['rpm_sig'],
                                                                    '%.2f' % m['size_mm'] if m['size_mm'] else '-', r['size_mm']))
    print('%d of %d right, %.1f s' % (right, len(files), time.time() - t0))
