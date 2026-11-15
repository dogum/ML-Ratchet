"""A rung that changes only how a trained network's outputs are used (v2.1): copy a checkpoint and set options.
usage: python ckpt_variant.py <src.pt> <dst.pt> k=v [k=v ...]     e.g. offstop=1 (the engine-stop constraint)"""
from common import *
import torch

if __name__ == '__main__':
    st = torch.load(f'{ROOT}/ckpt/{sys.argv[1]}', weights_only=False, map_location='cpu')
    st['opts'] = dict(st.get('opts', {}), **dict(x.split('=', 1) for x in sys.argv[3:]))
    st['variant_of'] = sys.argv[1]
    torch.save(st, f'{ROOT}/ckpt/{sys.argv[2]}')
    print(sys.argv[2], st['opts'])
