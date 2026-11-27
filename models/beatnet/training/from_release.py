"""Rebuild a torch checkpoint from the numpy release (BatchNorm already folded into the convolutions): every BatchNorm is
set to the identity, so in eval mode the network gives the release's outputs. Used to re-score the released v1.0 rung (r5)
under split v2 / physics v5 (27 Nov 2026), whose torch checkpoint is not in this repository. Scoring only: train-mode
BatchNorm statistics of the original are lost, so this file is never a warm start.
Usage: from_release.py <weights.npz> <out.pt>"""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC); import model as M

def rebuild(npz, out):
    z = np.load(npz); meta = json.loads(str(z['meta'])); net = M.BeatNet(cin=meta['cin'], nout=z[f"h{meta['n_head'] - 1}_w"].shape[0]); sd = net.state_dict()
    for i in range(meta['n_blocks']):
        for c, b in (('c1', 'b1'), ('c2', 'b2'), ('sk', 'bs')):
            sd[f'body.{i}.{c}.weight'] = torch.from_numpy(z[f'b{i}_{c}_w']); sd[f'body.{i}.{c}.bias'] = torch.from_numpy(z[f'b{i}_{c}_b'])
            bn = getattr(net.body[i], b); n = bn.num_features
            sd[f'body.{i}.{b}.weight'] = torch.full((n,), float(np.sqrt(1 + bn.eps))); sd[f'body.{i}.{b}.bias'] = torch.zeros(n)
            sd[f'body.{i}.{b}.running_mean'] = torch.zeros(n); sd[f'body.{i}.{b}.running_var'] = torch.ones(n)
    lin = [k for k, m in enumerate(net.head) if hasattr(m, 'weight')]
    for j, k in enumerate(lin): sd[f'head.{k}.weight'] = torch.from_numpy(z[f'h{j}_w']); sd[f'head.{k}.bias'] = torch.from_numpy(z[f'h{j}_b'])
    net.load_state_dict(sd); torch.save({'state': net.state_dict(), 'cin': meta['cin'], 'nout': net.head[-1].out_features, 'args': dict(from_release=npz)}, out); return meta

if __name__ == '__main__': print(rebuild(sys.argv[1], sys.argv[2])['ckpt'], '->', sys.argv[2])
