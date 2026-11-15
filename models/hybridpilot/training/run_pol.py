"""score a HybridPilot configuration on a set (matched-SOC DP), save data/pol_<set>_<name>.pkl
usage: run_pol.py SET NAME CKPT 'json params'   (params: preview_s, nav: case|0|1, knobs)"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E
from hp_env import ROOT, PTAG

if __name__ == '__main__':
    setname, name, ckpt, params = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
    pol = E.score_policy(setname, ckpt, params)
    pickle.dump(pol, open(f'{ROOT}/data/pol_{setname}_{name}{PTAG}.pkl', 'wb'))
    print(name, len(pol), 'skill', E.skill(pickle.load(open(E.base_path(setname), 'rb')), pol)[0])
