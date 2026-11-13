"""Export a checkpoint to numpy weights (BatchNorm folded into each convolution), verify numpy == torch, and write the
browser weight pack (float16, base64). usage: python3 export.py <ckpt> [sigk_json]"""
from common import *
import torch, base64
import evaluate as E
from model import REG

REL = f'{ROOT}/release'


def fold(block):
    w = block.c.weight.detach().numpy().astype(np.float64); b = block.c.bias.detach().numpy().astype(np.float64)
    bn = block.bn
    g = bn.weight.detach().numpy(); be = bn.bias.detach().numpy(); mu = bn.running_mean.numpy(); var = bn.running_var.numpy()
    s = g / np.sqrt(var + bn.eps)
    return (w * s[:, None, None]).astype(np.float32), ((b - mu) * s + be).astype(np.float32), int(block.c.stride[0])


def export(ckpt, extra=None):
    net = E.load_net(f'{ROOT}/ckpt/{ckpt}')
    W = {}
    blocks = [net.stem] + list(net.blocks)
    for i, bl in enumerate(blocks):
        w, b, s = fold(bl); W[f'c{i}_w'] = w; W[f'c{i}_b'] = b; W[f'c{i}_s'] = np.array(s)
    for name, mod in [('trip', net.trip), ('car', net.car), ('led', net.h_led)]:
        lin = [m for m in mod if isinstance(m, torch.nn.Linear)]
        for j, m in enumerate(lin):
            W[f'{name}{j}_w'] = m.weight.detach().numpy(); W[f'{name}{j}_b'] = m.bias.detach().numpy()
    for name, m in [('arch', net.h_arch), ('strat', net.h_strat), ('reg', net.h_reg)]:
        W[f'{name}_w'] = m.weight.detach().numpy(); W[f'{name}_b'] = m.bias.detach().numpy()
    W['n_blocks'] = np.array(len(blocks)); W['down'] = np.array(net.down)
    meta = dict(ckpt=ckpt, archs=ARCHS, strats=STRATS, reg=REG, ledger=LEDGER, feat=int(net.feat), **(extra or {}))
    W['meta'] = np.array(json.dumps(meta))
    os.makedirs(REL, exist_ok=True)
    np.savez(f'{REL}/tripdetective_weights.npz', **W)
    return net, W, meta


def verify(net, n=6):
    sys.path.insert(0, REL)
    import importlib, tripdetective as TD
    importlib.reload(TD)
    td = TD.TripDetective(f'{REL}/tripdetective_weights.npz')
    import simbank
    b = simbank.load_bank('val_v2' if os.path.exists(f'{ROOT}/bank/val_v2/chunk_000.npz') else 'val_v1')
    idx = np.where(b['veh'] == b['veh'][0])[0][:n]
    inp = E.bank_inputs(b, idx)
    Pt = E.predict(net, inp, np.zeros(len(idx), int))
    o = b['offs']
    Pn = td.predict_raw([b['raw'][o[j]:o[j + 1]].T.astype(np.float32) for j in idx], [b['flags'][j] for j in idx])
    d = max(float(np.abs(Pt['mu'] - Pn['mu']).max()), float(np.abs(Pt['arch'] - Pn['arch']).max()), float(np.abs(Pt['led_mu'] - Pn['led_mu']).max() / 100))
    print('numpy vs torch max abs diff', d)
    return d


def web_pack(W, path):
    keys = [k for k in W if k not in ('meta', 'n_blocks', 'down') and not k.endswith('_s')]
    man, parts, off = [], [], 0
    for k in keys:
        a = np.asarray(W[k], np.float32); man.append(dict(k=k, shape=list(a.shape), off=off)); off += a.size; parts.append(a.ravel())
    allw = np.concatenate(parts).astype(np.float16)
    pack = dict(b64=base64.b64encode(allw.tobytes()).decode(), man=man,
                meta=dict(n_blocks=int(W['n_blocks']), strides=[int(W[f'c{i}_s']) for i in range(int(W['n_blocks']))], down=int(W['down']),
                          **json.loads(str(W['meta']))))
    json.dump(pack, open(path, 'w'))
    print('web pack', path, os.path.getsize(path) // 1024, 'kB')


if __name__ == '__main__':
    extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    net, W, meta = export(sys.argv[1], extra)
    verify(net)
    web_pack(W, f'{ROOT}/web/weights.json')
