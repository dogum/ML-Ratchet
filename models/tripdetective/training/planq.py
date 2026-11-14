"""Run a sequence of ratchet steps: rescore every logged rung on a new frozen set, or train + score one rung.
usage: python planq.py <plan.json>  (steps: ["rescore", val] | ["rung", name, val]; rungs as in trainq.py)"""
from common import *
import time, subprocess
from trainq import ready, PY

if __name__ == '__main__':
    plan = json.load(open(sys.argv[1])); R = plan['rungs']
    done = jload(f'{ROOT}/logs/{os.path.basename(sys.argv[1])}.done', [])
    for st in plan['steps']:
        key = ':'.join(st)
        if key in done: continue
        if st[0] == 'rescore':
            while not ready(st[1], 6): time.sleep(30)
            print('rescore', st[1], time.strftime('%H:%M'), flush=True)
            subprocess.run([PY, 'ratchet.py', '--rescore', st[1]], stdout=open(f'{ROOT}/logs/rescore_{st[1]}.log', 'w'), stderr=subprocess.STDOUT, check=True)
        else:
            name, val = st[1], st[2]; r = R[name]
            while not ready(st[2], 6): time.sleep(30)
            if not os.path.exists(f'{ROOT}/ckpt/{name}.pt'):
                while not ready(r['bank'], r['nchunks']): time.sleep(30)
                print('train', name, time.strftime('%H:%M'), flush=True)
                subprocess.run([PY, 'train.py', r['bank'], name, str(r.get('steps', 2500))] + [f'{k}={v}' for k, v in r.get('opts', {}).items()],
                               stdout=open(f'{ROOT}/logs/{name}.train.log', 'w'), stderr=subprocess.STDOUT, check=True)
            print('ratchet', name, time.strftime('%H:%M'), flush=True)
            subprocess.run([PY, 'ratchet.py', f'{name}.pt', val, r['note']], stdout=open(f'{ROOT}/logs/{name}.ratchet.log', 'w'), stderr=subprocess.STDOUT, check=True)
            print([l for l in open(f'{ROOT}/logs/{name}.ratchet.log').read().split('\n') if 'ACCEPTED' in l or 'rejected' in l], flush=True)
        done.append(key); jsave(done, f'{ROOT}/logs/{os.path.basename(sys.argv[1])}.done')
    print('planq done', flush=True)
