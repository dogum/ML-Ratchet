"""Assemble release/: weights (BatchNorm folded, numpy == torch check), demo examples, training scripts; zip it and test
the demo from an unzipped copy. Usage: python3 make_release.py"""
import _paths as _P  # portable paths (see _paths.py)
import os, sys, json, shutil, glob, subprocess, zipfile, numpy as np
import evaluate as E, events as EV, realdata as RD, export as EX
ROOT = E.ROOT; REL = os.path.join(ROOT, 'release'); OUT = _P.OUT + '/swellcast-v1.0.zip'

if __name__ == '__main__':
    fin = json.load(open(os.path.join(ROOT, 'final.json'))); sf = fin['sigma_factor']
    extra = dict(sigma_factor={k: sf[k] for k in ('logD', 't0', 'dir', 'loghs')}, det_thr=fin['det_thr'])
    W = os.path.join(REL, 'swellcast_weights.npz'); net = EX.export(fin['ckpt'], W, extra)
    sys.path.insert(0, REL); import swellcast as SC, pipeline as PL, torch, model as M
    X = EV.windows(EV.load('dev', 'confirmed')[:8]); f = M.features(X)
    with torch.no_grad(): ot = net(torch.from_numpy(f)).numpy()
    on = SC.SwellCast(W).forward(f); dif = float(np.abs(on - ot).max()); print('numpy vs torch max abs diff', dif)
    # examples
    ex = os.path.join(REL, 'examples'); shutil.rmtree(ex, ignore_errors=True); os.makedirs(ex + '/raw')
    for k in 'wdj': shutil.copy(os.path.join(ROOT, f'data/ndbc/raw/51004{k}2018.txt.gz'), ex + '/raw/')
    ol = [e for e in EV.load('exam', 'confirmed') if e['name'] == 'OLIVIA' and e['buoy'] == '51004'][0]
    json.dump(dict(window_centre_hour=ol['tc'], truth=dict(D=ol['D'] / 1e3, t0=ol['t0_rel'], dir=ol['dir'], name='OLIVIA', sid=ol['sid'])), open(ex + '/raw/olivia.json', 'w'))
    rec = PL.read_ndbc(*(os.path.join(ex, f'raw/51004{k}2018.txt.gz') for k in 'wdj')); w1, tc1 = PL.window(rec, ol['tc']); (w2, tc2) = RD.window('51004', ol['tc']); w2 = np.stack(w2)
    pdif = float(np.nanmax(np.abs(np.log10(np.maximum(w1[0], 1e-3)) - np.log10(np.maximum(w2[0], 1e-3))))); print('pipeline vs training window, max |dlog10 E|', pdif, 'same grid centre', tc1 == tc2)
    ev = EV.load('exam', 'confirmed'); rng = np.random.default_rng(11)
    for i in rng.choice(len(ev), 5, replace=False):
        e = ev[i]; (x, tc) = RD.window(e['buoy'], e['tc'])
        np.savez_compressed(os.path.join(ex, f"{e['buoy']}_{e['name'].lower()}_{e['year']}.npz"), x=np.stack(x), centre_hour=tc,
                            truth=json.dumps(dict(name=e['name'], buoy=e['buoy'], basin=e['basin'], year=e['year'], D=e['D'] / 1e3, t0=e['t0_rel'], dir=e['dir'])))
    tr = os.path.join(REL, 'training'); shutil.rmtree(tr, ignore_errors=True); os.makedirs(tr)
    for f in glob.glob(os.path.join(ROOT, 'src/*.py')): shutil.copy(f, tr)
    for f in ('ratchet.json', 'final.json', 'physics_check.json', 'physics_real.json', 'calib_v2_record.json', 'calib_dev.json', 'twin_dev.json', 'twin_net_r1.json', 'twin_net_r2.json', 'twin_net_r3.json', 'twin_net_r4.json', 'baseline.json', 'baseline_det.json', 'sigma_factor.json'):
        if os.path.exists(os.path.join(ROOT, f)): shutil.copy(os.path.join(ROOT, f), tr)
    json.dump(dict(numpy_vs_torch_max_abs=dif, pipeline_vs_training_window_max_dlog10E=pdif), open(os.path.join(tr, 'release_checks.json'), 'w'), indent=1)
    shutil.rmtree(os.path.join(REL, '__pycache__'), ignore_errors=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
        for dp, dn, fn in os.walk(REL):
            dn[:] = [d for d in dn if d != '__pycache__']
            for f in fn: p = os.path.join(dp, f); z.write(p, os.path.join('swellcast-v1.0', os.path.relpath(p, REL)))
    print('zip', OUT, round(os.path.getsize(OUT) / 1e6, 2), 'MB')
    t = os.path.join(ROOT, 'logs/unzip_test'); shutil.rmtree(t, ignore_errors=True); os.makedirs(t)
    zipfile.ZipFile(OUT).extractall(t); r = subprocess.run([sys.executable, 'demo.py'], cwd=os.path.join(t, 'swellcast-v1.0'), capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])
