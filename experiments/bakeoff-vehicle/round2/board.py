"""Round 2: every evaluated folder's sim val skill / MdAPE and dev D16 (+ the five shared-bias dev sets' bias).
python board.py [filter]"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SH = ['VED_EV', 'VED_PHEV', 'Mersin_diesel', 'Mersin_cng', 'ICCT']
flt = sys.argv[1] if len(sys.argv) > 1 else ''
for f in sorted(glob.glob(os.path.join(HERE, 'work', 'eval', '*', '*_summary.json'))):
    if flt not in f:
        continue
    s = json.load(open(f)); sv = s.get('simval', {}); dv = s.get('dev', {})
    sets = dv.get('sets', {})
    b = ' '.join(f"{k} {sets[k]['bias']:+.3f}" for k in SH if k in sets)
    print(f"{os.path.relpath(f, os.path.join(HERE, 'work', 'eval')):60s} skill {sv.get('skill', float('nan')):.5f} "
          f"MdAPE {sv.get('MdAPE', float('nan')):.5f} D16 {dv.get('D16', float('nan')):.5f} | {b}")
