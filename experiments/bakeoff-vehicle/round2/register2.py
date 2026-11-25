"""Round 2: register (freeze) an arm in the ROUND-2 registry (round2/registry/) with the harness's own register.py.

python register2.py <arm_dir> [--replace]       register a round-2 arm or a v1.0 reference arm (from its repo folder)
python register2.py --list                      every round-2 record, re-hashed
python register2.py --pairs                     write registry/prereg_pairs.json (PREREG section 5's pairs) from the
                                                registered arms (fixed before exam-2)
v1.0's harness/registry is never written. Registration is refused once exam-2 has started (EXAM2_RUNNING / DONE).
"""
import json
import os
import sys

R2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, R2)
import paths as P  # noqa: E402
import config as C  # noqa: E402
C.RESULTS = P.RESULTS
C.REGISTRY = P.REGISTRY
os.makedirs(C.RESULTS, exist_ok=True)
os.makedirs(C.REGISTRY, exist_ok=True)
P.setup()
import bake  # noqa: E402
bake.LOCK = os.path.join(C.RESULTS, 'EXAM2_RUNNING.json')
bake.DONE = os.path.join(C.RESULTS, 'EXAM2_DONE.json')
import register as REG  # noqa: E402
P.patch_registry(REG)          # registry arm_dir (worktree / cloud path) -> this checkout


class _Paths(set):
    """the audit's record of files read under the bake-off tree, minus round 2's data and path layer (they play the
    part v1.0's harness/bank, harness/real and harness/*.py played, which the audit exempted)."""
    def add(self, p):
        if not (p.startswith(P.WORK + os.sep) or os.path.dirname(p) == P.R2):
            set.add(self, p)


_Audit = REG.Audit


class Audit(_Audit):
    def __enter__(self):
        r = _Audit.__enter__(self)
        self.paths = _Paths()
        return r


REG.Audit = Audit

FAMILY_PAIRS = [   # (round-2 arm, comparison row, why) -- PREREG section 5
    ('r2_d1_gru', 'trace_gru', 'D1 vs v1.0 GRU: data + ensembling'),
    ('r2_d2_transformer', 'trace_transformer', 'D2 vs v1.0 transformer: data + ensembling'),
    ('r2_d3_mlp', 'tab_mlp', 'D3 vs v1.0 MLP control: data'),
    ('r2_p1_gru', 'r2_d1_gru', 'P1 vs D1: physics / bank'),
    ('r2_p2_transformer', 'r2_d2_transformer', 'P2 vs D2: physics / bank'),
    ('r2_p3_mlp', 'r2_d3_mlp', 'P3 vs D3: physics / bank'),
]


def pairs():
    regs = REG.registered()
    out = [[a, b, w] for a, b, w in FAMILY_PAIRS if a in regs and (b in regs)]
    r2 = [k for k, r in regs.items() if r['meta'].get('group') == 'round2' and not k.endswith('_conformal')]
    for k in r2:
        out.append([k, 'enet_v21', 'round-2 arm vs EnergyNet v2.1'])
    x = [k for k in r2 if regs[k]['meta'].get('family') == 'ensemble' and k.startswith('r2_x')]
    for k in x:
        for m in json.load(open(os.path.join(regs[k]['arm_dir'], 'members.json'))):
            mid = json.load(open(os.path.join(regs[k]['arm_dir'], os.path.dirname(m[1]), 'ratchet.json')))['arm_id']
            out.append([k, mid, 'X vs its member'])
    json.dump(out, open(os.path.join(C.REGISTRY, 'prereg_pairs.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == '--list':
        for k, r in REG.registered().items():
            ok, h, ch = REG.verify(r)
            print(f'{k:28s} {r["hash"][:12]} {"OK" if ok else "CHANGED " + str(ch)} {os.path.relpath(r["arm_dir"], P.BAKEOFF)}')
    elif a[0] == '--pairs':
        pairs()
    else:
        sys.exit(REG.register(a[0], '--replace' in a))
