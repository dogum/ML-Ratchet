"""Train a list of rungs on the GPU one after another as their banks become ready, then score them through the ratchet in
ladder order (the training order does not matter; the decision order does).
usage: python trainq.py <plan.json>   plan: {"train_order": [names], "ratchet_order": [names], "val": tag,
                                              "rungs": {name: {"bank": tag, "nchunks": n, "steps": n, "opts": {k: v}, "note": str}}}"""
from common import *
import time, glob, subprocess

PY = sys.executable


def ready(bank, n):
    return len(glob.glob(f'{ROOT}/bank/{bank}/chunk_*.npz')) >= n and not any(
        int(os.path.basename(f)[6:9]) < n for f in glob.glob(f'{ROOT}/bank/{bank}/chunk_*.lock'))


if __name__ == '__main__':
    plan = json.load(open(sys.argv[1]))
    R = plan['rungs']
    for name in plan['train_order']:
        r = R[name]
        if os.path.exists(f'{ROOT}/ckpt/{name}.pt'): continue
        while not ready(r['bank'], r['nchunks']): time.sleep(30)
        opts = [f'{k}={v}' for k, v in r.get('opts', {}).items()]
        print('train', name, time.strftime('%H:%M'), flush=True)
        subprocess.run([PY, 'train.py', r['bank'], name, str(r.get('steps', 2500))] + opts,
                       stdout=open(f'{ROOT}/logs/{name}.train.log', 'w'), stderr=subprocess.STDOUT, check=True)
    for name in plan.get('ratchet_order', []):
        print('ratchet', name, time.strftime('%H:%M'), flush=True)
        subprocess.run([PY, 'ratchet.py', f'{name}.pt', plan['val'], R[name]['note']],
                       stdout=open(f'{ROOT}/logs/{name}.ratchet.log', 'w'), stderr=subprocess.STDOUT, check=True)
        print(open(f'{ROOT}/logs/{name}.ratchet.log').read().split('\n')[-2], flush=True)
    print('trainq done', flush=True)
