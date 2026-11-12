"""Assemble the v2 release folder and zip it."""
import _paths as _P  # portable paths (see _paths.py)
import os, shutil, glob, zipfile
R = _P.WORK + '/release'; T = os.path.join(R, 'training')
SCRIPTS = ['gwphys.py', 'build_bank.py', 'sampler.py', 'model.py', 'train.py', 'evaluate.py', 'realeval.py', 'ratchet.py', 'cascade_tune.py',
           'export.py', 'export_js.py', 'release_build.py', 'bank_phys.py', 'pe_sampler.py', 'flow.py', 'train_pe.py', 'eval_pe.py', 'export_pe.py',
           'bns.py', 'bns_train.py', 'bns_real.py', 'bns_ratchet.py', 'export_bns.py', 'viz.py', 'model_card.py']
shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
for s in SCRIPTS: shutil.copy(f'{_P.SRC}/{s}', T)
shutil.copy(_P.WORK + '/ratchet.json', os.path.join(T, 'ratchet_log.json'))
shutil.copy(_P.WORK + '/bns_rungs.json', os.path.join(T, 'bns_ratchet_log.json'))
shutil.copy(_P.WORK + '/cascade_tune.json', os.path.join(T, 'cascade_veto_tuning.json'))
files = ['chirpnet.py', 'demo.py', 'README.md', 'MODEL_CARD.md', 'chirpnet_weights.npz', 'chirpnet_coinc_weights.npz', 'chirpnet_pe_weights.npz', 'chirpnet_bns_weights.npz']
out = _P.OUT + '/chirpnet-v2.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files: z.write(os.path.join(R, f), f'chirpnet-v2/{f}')
    for f in sorted(glob.glob(os.path.join(R, 'data', '*.hdf5'))): z.write(f, f'chirpnet-v2/data/{os.path.basename(f)}')
    for f in sorted(os.listdir(T)): z.write(os.path.join(T, f), f'chirpnet-v2/training/{f}')
print(out, '%.1f MB' % (os.path.getsize(out) / 1e6))
