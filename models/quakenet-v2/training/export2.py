"""Export the released graph network to numpy (release/quakenet2_weights.npz) and check numpy == torch on dev samples.
Usage: export2.py <ckpt name>"""
import os, sys, json, numpy as np, torch
from _paths import H; sys.path.insert(0, H + 'src'); sys.path.insert(0, H + 'release'); sys.path.insert(2, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # model root: quakenet2.py
import score2 as SC, train2 as T2

def linears(seq):
    return [m for m in seq if isinstance(m, torch.nn.Linear)]

if __name__ == '__main__':
    name = sys.argv[1]; net, a = SC.load_net(name); W = {}
    def put(prefix, seq):
        for j, l in enumerate(linears(seq)): W[f'{prefix}_{j}_w'] = l.weight.detach().numpy(); W[f'{prefix}_{j}_b'] = l.bias.detach().numpy()
    put('enc', net.enc); put('out', net.out)
    for r, (m, u) in enumerate(zip(net.msg, net.upd)): put(f'msg{r}', m); put(f'upd{r}', u)
    if net.use_emb: W['proj_w'] = net.proj.weight.detach().numpy(); W['proj_b'] = net.proj.bias.detach().numpy()
    f = json.load(open(H + 'data/sigma_factors.json'))['v2']
    meta = dict(ckpt=name, rounds=len(net.msg), h=net.h, use_emb=bool(a.get('emb')), use_cc=bool(a.get('cc')), use_time=not a.get('no_time'), sigma_factor=f,
                params=int(sum(p.numel() for p in net.parameters())))
    np.savez(H + 'release/quakenet2_weights.npz', meta=json.dumps(meta), **{k: v.astype(np.float32) for k, v in W.items()})
    import quakenet2 as Q2
    g = Q2.GraphNet({k: v.astype(np.float32) for k, v in W.items()}, meta)
    D = T2.load(H + 'data/dev_samples.npz'); F = T2.featurize(D, True); rows = np.arange(len(D['K'])); diff = 0
    (xn, xe, m, em, y), _ = T2.make_batch(F, D, rows, np.zeros((len(rows), 10)), meta['use_time'], meta['use_cc'])
    with torch.no_grad(): ot = net(xn, xe, m, em if meta['use_emb'] else None).numpy()
    for i, s in enumerate(rows):
        K = int(D['K'][s]); on = g(xn[i, :K].numpy(), xe[i, :K, :K].numpy(), em[i, :K].numpy()); diff = max(diff, float(np.abs(on - ot[i]).max()))
    print('numpy vs torch, max |difference| over', len(rows), 'dev samples:', diff); json.dump(dict(max_abs_diff=diff, n=len(rows)), open(H + 'data/export_check.json', 'w'))
