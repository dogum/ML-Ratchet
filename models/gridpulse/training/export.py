"""Export a rung to numpy (.npz, release) and JSON (page), with the dev-fitted sigma widening factors.
Checks torch == numpy == JS on real dev windows; writes checks/js_parity.json['network']."""
from common import *
import torch, evaluate as EV, subprocess, net as NET
torch.set_num_threads(1)


def export(rung, widen, out_npz, out_json):
    net, s = EV.load(rung)
    sd = {k: v.numpy() for k, v in net.state_dict().items()}
    W = {}
    i = 0
    while f'body.{2 * i}.weight' in sd:
        W[f'b{i}_w'] = sd[f'body.{2 * i}.weight']; W[f'b{i}_b'] = sd[f'body.{2 * i}.bias']; i += 1
    W.update(mu_w=sd['mu.weight'], mu_b=sd['mu.bias'], ls_w=sd['ls.weight'], ls_b=sd['ls.bias'],
             xm=s['xm'], xs=s['xs'], ym=s['ym'], ys=s['ys'], widen=np.asarray(widen, np.float32),
             ctx=np.array(bool(s.get('ctx', False))))
    np.savez(out_npz, **W)
    import base64
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype='<f4').ravel().tobytes()).decode()
    J = dict(layers=[dict(w=b64(W[f'b{j}_w']), b=W[f'b{j}_b'].tolist(), n=int(W[f'b{j}_b'].shape[0])) for j in range(i)],
             mu_w=b64(W['mu_w']), mu_b=W['mu_b'].tolist(), ls_w=b64(W['ls_w']),
             ls_b=W['ls_b'].tolist(), xm=W['xm'].tolist(), xs=W['xs'].tolist(), ym=W['ym'].tolist(), ys=W['ys'].tolist(),
             widen=list(map(float, widen)), nout=int(W['mu_b'].shape[0]), ctx=bool(s.get('ctx', False)), rung=rung)
    json.dump(J, open(out_json, 'w'), separators=(',', ':'))
    return W


def parity(rung, npz, js_json):
    import gridpulse_np as GPN
    W = dict(np.load(npz))
    wins = [EV.real_gb(e)['w'][0] for e in GB_DEV]
    rows = []
    for N in (5, 10, 60):
        w = np.array(wins); dem = np.array([EV.DEV_DEMAND[e] for e in GB_DEV])
        mu_t, sg_t = EV.predict(rung, w, np.full(4, N), np.ones(4), np.full(4, 50.0), dem)
        mu_n, sg_n = GPN.forward(W, w, np.full(4, N), np.ones(4), np.full(4, 50.0), dem)
        code = f"""const GP=require('{LAB}/gp.js'); const M=require('{js_json}'); M.widen=M.widen.map(()=>1);
const W={json.dumps(w.tolist())}, D={json.dumps(dem.tolist())};
console.log(JSON.stringify(W.map((w,i)=>GP.predict(M,w,{N},true,50,D[i]).mu)));"""
        mu_j = np.array(json.loads(subprocess.run(['node', '-e', code], capture_output=True, text=True, check=True).stdout))
        rows.append(dict(N=N, torch_vs_numpy=float(np.abs(mu_t - mu_n).max()), numpy_vs_js=float(np.abs(mu_n - mu_j).max())))
    return rows


if __name__ == '__main__':
    rung = sys.argv[1]; widen = json.loads(sys.argv[2]) if len(sys.argv) > 2 else [1] * 7
    os.makedirs(f'{ROOT}/release', exist_ok=True)
    export(rung, widen, f'{ROOT}/release/gridpulse_weights.npz', f'{ROOT}/web/model.json')
    r = parity(rung, f'{ROOT}/release/gridpulse_weights.npz', f'{ROOT}/web/model.json')
    print(r)
    J = json.load(open(f'{ROOT}/checks/js_parity.json')); J['network'] = dict(rung=rung, rows=r)
    json.dump(J, open(f'{ROOT}/checks/js_parity.json', 'w'), indent=1)
