"""Export the final checkpoint to the numpy release, check numpy == torch, write the demo examples.
python3 export.py <ckpt>"""
import sys, os, json, shutil, glob, numpy as np, torch
from common import *
import evaluate as E

REL = f'{ROOT}/release'


def main(ck):
    nets = E.load(ck)
    ex = json.load(open(f'{DATA}/exam/{ck}.json'))
    arrs = {}
    for k, net in nets.items():
        for n, v in net.state_dict().items():
            arrs[f'{k}.{n}'] = v.numpy().astype(np.float32)
    np.savez(f'{REL}/demandcast_weights.npz', **arrs)
    nparam = {k: int(sum(p.numel() for p in net.parameters())) for k, net in nets.items()}
    meta = dict(ckpt=ck, sigma_scale=ex['sigma_scale_dev'], n_params=nparam,
                inputs={'d': int(nets['d'].xm.shape[0]), 'w': int(nets['w'].xm.shape[0]), 's': int(nets['s'].xm.shape[0])},
                issue='10:00 local standard time on D-1; target = 24 h of D in local standard time',
                lst_offset_h=LST)
    json.dump(meta, open(f'{REL}/demandcast_meta.json', 'w'), indent=1)
    os.makedirs(f'{REL}/calib', exist_ok=True)
    for f in ('demand_dev.json', 'renew_dev.json'):
        shutil.copy(f'{DATA}/calib/{f}', f'{REL}/calib/{f}')
    # examples: fixed random draw, two exam days per grid (actual weather)
    B = dict(np.load(f'{DATA}/banks/real_exam.npz'))
    rng = np.random.default_rng(16)
    pick = []
    for ba in BAS:
        ix = np.where((B['meta_ba'] == ba) & (B['meta_fc'] == 0))[0]
        pick += sorted(rng.choice(ix, 2, replace=False).tolist())
    keys = ('xd', 'pd_total', 'd_true', 'df', 'xw', 'pw', 'yw', 'capw', 'xs', 'ps', 'ys', 'caps', 'night', 'hasw', 'hass')
    E2 = {k: B[k][pick] for k in keys}
    E2['ba'] = B['meta_ba'][pick]; E2['day'] = B['meta_day'][pick]
    os.makedirs(f'{REL}/examples', exist_ok=True)
    np.savez_compressed(f'{REL}/examples/exam_days.npz', **E2)
    # numpy == torch
    sys.path.insert(0, REL)
    import demandcast as DCm
    m = DCm.DemandCast()
    tp = E.predict(nets, E2)
    nd = m.predict_demand(E2['xd'], E2['pd_total'])
    dmu = np.max(np.abs(np.log(nd['mw'] / E2['pd_total']) - tp['d'][0]))
    nw = m.predict_cf('w', E2['xw'], E2['pw']); ns = m.predict_cf('s', E2['xs'], E2['ps'], E2['night'])
    dw = np.max(np.abs(nw['cf'] - tp['w'][0])); ds = np.max(np.abs(ns['cf'] - tp['s'][0]))
    sig = np.max(np.abs(nd['log_sigma'] / ex['sigma_scale_dev']['d'] - tp['d'][1]))
    chk = dict(max_abs_diff_demand_logratio=float(dmu), max_abs_diff_wind_cf=float(dw), max_abs_diff_solar_cf=float(ds),
               max_abs_diff_demand_sigma=float(sig), n_params=nparam)
    json.dump(chk, open(f'{ROOT}/checks/numpy_vs_torch.json', 'w'), indent=1)
    print(chk)
    # training scripts
    os.makedirs(f'{REL}/training', exist_ok=True)
    for f in glob.glob(f'{SRC}/*.py'):
        shutil.copy(f, f'{REL}/training/')
    shutil.copy(f'{ROOT}/ratchet.json', f'{REL}/training/ratchet.json')
    for f in glob.glob(f'{ROOT}/checks/*.json'):
        os.makedirs(f'{REL}/training/checks', exist_ok=True); shutil.copy(f, f'{REL}/training/checks/')


if __name__ == '__main__':
    main(sys.argv[1])
