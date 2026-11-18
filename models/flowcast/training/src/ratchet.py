"""Ratchet bookkeeping.
  python3 ratchet.py score <ckpt> <val_ver>     sim score on the frozen set of physics version val_ver (800 basins)
  python3 ratchet.py decide <rung> <incumbent> <val_ver> "<change>"   apply the rule and log it in ratchet.json
Rule: keep if sim median NSE (frozen set val_ver) > incumbent + 0.01 AND dev median NSE >= incumbent x 0.95
(dev = results/<ckpt>_dev.parquet: 52 dev basins, WY2001-2014, ungauged)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import torch
torch.set_num_threads(1)
import bankdata as bd
from train import sim_eval, build_net

RJ = f'{ROOT}/ratchet.json'


def load_ckpt(name):
    ck = torch.load(f'{ROOT}/ckpt/{name}.pt', weights_only=False)
    net = build_net(ck['cfg']); net.load_state_dict(ck['state']); net.eval()
    return net, ck['cfg']


def score(name, ver):
    fn = f'{ROOT}/results/sim_{name}_{ver}.json'
    if os.path.exists(fn):
        return json.load(open(fn))
    net, cfg = load_ckpt(name)
    B = bd.add_sigma(bd.load_bank(ver, 'val'))
    ev = sim_eval(net, B, cfg.get('nextday', False))
    jdump(ev, fn)
    return ev


def dev(name):
    d = pd.read_parquet(f'{ROOT}/results/{name}_dev.parquet')
    return {'nse_median': float(d.nse.median()), 'kge_median': float(d.kge.median()), 'fhv_median': float(d.fhv.median()),
            'cov90_median': float(d.cov90.median()), 'frac_nse_lt0': float((d.nse < 0).mean()),
            'peak_abs_err_median': float(d.peak_abs_err_pct_median.median())}


def decide(rung, inc, ver, change):
    R = json.load(open(RJ)) if os.path.exists(RJ) else {'rule': __doc__, 'rungs': []}
    s_new, s_inc = score(rung, ver), score(inc, ver)
    d_new, d_inc = dev(rung), dev(inc)
    sim_ok = s_new['nse_median'] > s_inc['nse_median'] + 0.01
    dev_ok = d_new['nse_median'] >= d_inc['nse_median'] - 0.05 * abs(d_inc['nse_median'])
    keep = bool(sim_ok and dev_ok)
    rec = {'rung': rung, 'incumbent': inc, 'frozen_set': ver, 'change': change,
           'sim_nse': s_new['nse_median'], 'sim_nse_incumbent': s_inc['nse_median'], 'sim_kge': s_new['kge_median'],
           'sim_cov90': s_new['cov90'], 'dev': d_new, 'dev_incumbent': d_inc, 'sim_ok': sim_ok, 'dev_ok': dev_ok,
           'kept': keep, 'time': time.strftime('%Y-%m-%d %H:%M')}
    R['rungs'].append(rec)
    jdump(R, RJ)
    print(json.dumps({k: v for k, v in rec.items() if k not in ('dev', 'dev_incumbent')}, indent=1))
    print('dev', d_new, '\ninc', d_inc)
    return keep


if __name__ == '__main__':
    if sys.argv[1] == 'score':
        ev = score(sys.argv[2], sys.argv[3]); print({k: v for k, v in ev.items() if k != 'nse_all'})
    elif sys.argv[1] == 'decide':
        decide(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
