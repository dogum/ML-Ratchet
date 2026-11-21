"""Export a checkpoint to the numpy release (release/flowcast_weights.npz) and web JSON; verify numpy == torch.
usage: python3 export.py <ckpt> <k50> <k90>"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import torch
import evalreal as ER
sys.path.insert(0, MODEL)
import importlib
import flowcast as FCm


def main(name, k50, k90):
    net, cfg = ER.load_net(name)
    sd = {k: v.numpy().astype(np.float32) for k, v in net.state_dict().items()}
    W = {'w_ih': sd['lstm.weight_ih_l0'], 'w_hh': sd['lstm.weight_hh_l0'], 'b_ih': sd['lstm.bias_ih_l0'],
         'b_hh': sd['lstm.bias_hh_l0'], 'head_w': sd['head.weight'], 'head_b': sd['head.bias'],
         'st_mu': sd['st_mu'], 'st_sd': sd['st_sd'], 'k_widen': np.array([k50, k90], np.float32)}
    np.savez(f'{ROOT}/release/flowcast_weights.npz', **W)
    importlib.reload(FCm)
    m = FCm.FlowCast(f'{ROOT}/release/flowcast_weights.npz')
    import waterdata as wd
    errs = []
    for b in wd.basins('camels', split='dev')[:4]:
        idx, mean_t, q_t = ER.run_basin(net, False, b, '2003-10-01', '2005-09-30')
        f = wd.forcing(b, '2002-10-01', '2005-10-01')
        out = m.predict(f, wd.spec(b), '2003-10-01', '2005-09-30')
        errs.append(float(np.max(np.abs(out['Q'].values - mean_t)) / max(np.max(np.abs(mean_t)), 1e-9)))
    print('max relative |numpy - torch| on 4 dev basins x 2 years:', max(errs))
    jdump({'ckpt': name, 'max_rel_err_numpy_vs_torch': max(errs), 'k_widen': [k50, k90]}, f'{ROOT}/results/export_check.json')
    # web weights (float32 rounded)
    web = {k: np.round(v.astype(np.float64), 6).tolist() for k, v in W.items()}
    json.dump(web, open(f'{ROOT}/web/weights.json', 'w'))


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
