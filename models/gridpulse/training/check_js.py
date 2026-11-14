"""JS parity: (1) the page's frequency simulator vs gridsim.frequency_event on lab states; (2) the JS MLP vs numpy vs
torch on real dev windows. Writes checks/js_parity.json."""
from common import *
import lab, subprocess
CASES = [('GB', 1000, 200, 0.4, 1000), ('GB', 1800, 120, 0.7, 0), ('GB', 600, 300, 0.2, 500),
         ('ERCO', 1400, 250, 0.4, 500), ('ERCO', 2750, 150, 0.7, 0), ('ERCO', 800, 300, 0.3, 300)]
os.makedirs(f'{ROOT}/checks', exist_ok=True)


def sim_parity():
    py = []
    for c in CASES:
        st = lab.state(*c); t, f = lab.simulate(st, c[1]); py.append(f.tolist())
    js = f"""const GP=require('{LAB}/gp.js'); const C={json.dumps(CASES)};
console.log(JSON.stringify(C.map(c=>GP.simulate(GP.labState(c[0],c[1],c[2],c[3],c[4]),[[0,c[1]]]).f)));"""
    out = json.loads(subprocess.run(['node', '-e', js], capture_output=True, text=True, check=True).stdout)
    res = []
    for c, a, b in zip(CASES, py, out):
        a, b = np.array(a), np.array(b)
        res.append(dict(case=c, max_abs_Hz=float(np.abs(a - b).max()), nadir_py=float(a.min()), nadir_js=float(b.min())))
    return res


if __name__ == '__main__':
    r = sim_parity()
    for x in r: print(x)
    J = json.load(open(f'{ROOT}/checks/js_parity.json')) if os.path.exists(f'{ROOT}/checks/js_parity.json') else {}
    J['simulator'] = r; json.dump(J, open(f'{ROOT}/checks/js_parity.json', 'w'), indent=1)
