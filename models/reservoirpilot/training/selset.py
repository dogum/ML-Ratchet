"""Selection sets (checkpoint choice only; never the ratchet's frozen set). 200 sim tasks per generator, seed 7000+,
SOP scores only. Out: data/sel_<gen>.pkl"""
import sys, pickle, warnings
import numpy as np
warnings.filterwarnings('ignore')
import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
from gen import make_task
from task import simulate, components
for gen in sys.argv[1:]:
    rng = np.random.default_rng(7000 + int(gen[1:])); tasks = []
    for i in range(200):
        tk = make_task(rng, gen, eval_len=int(rng.integers(1000, 3654)))
        R, S = simulate('sop', tk['I'], tk['D'], tk['Qmax'], tk['cap'], tk['S0'])
        tk['scores'] = {'sop': components(R, S, tk['D'], tk['Qmax'], tk['cap'], tk['S0'])}; tasks.append(tk)
    pickle.dump(tasks, open(f'{_paths.WORK}/data/sel_{gen}.pkl', 'wb')); print(gen, 'done')
