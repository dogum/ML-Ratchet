"""Round 2: the tab_gbm arm's released recipe (r2) on a round-2 bank, through round 1's own scripts unchanged.

usage: python gbm2.py <bank> <out_dir>
  1. arms_tab/src/build_tab.py <out_dir>/tab_train.npz 100 1   (one window per train run, seeds 11/12, K 100)
  2. arms_tab/src/train_gbm.py <out_dir> <cfg> with cfg {"data": table, "monotone": false} (r2 = r0 without the
     monotone constraints; every other setting is train_gbm.py's default, as in round 1)
Both scripts run in this process after r2.use_bank(<bank>), so bake.load_bank reads the round-2 bank.
"""
import json
import os
import runpy
import sys
import time

import r2

bank, out = sys.argv[1], os.path.abspath(sys.argv[2])
r2.use_bank(bank)
SRC = os.path.join(r2.BAKEOFF, 'arms_tab', 'src')
sys.path.insert(0, SRC)
os.makedirs(out, exist_ok=True)
c0 = time.process_time()
tab = os.path.join(out, 'tab_train.npz')
if not os.path.exists(tab):
    sys.argv = ['build_tab.py', tab, '100', '1']
    runpy.run_path(os.path.join(SRC, 'build_tab.py'), run_name='__main__')
c1 = time.process_time()
cfg = os.path.join(out, 'cfg.json')
json.dump({'data': tab, 'monotone': False, 'note': 'r2 recipe: r0 without monotone constraints'}, open(cfg, 'w'))
sys.argv = ['train_gbm.py', out, cfg]
runpy.run_path(os.path.join(SRC, 'train_gbm.py'), run_name='__main__')
json.dump({'bank': bank, 'table_cpu_s': round(c1 - c0, 1), 'train_cpu_s': round(time.process_time() - c1, 1)},
          open(os.path.join(out, 'gbm2.json'), 'w'), indent=1)
