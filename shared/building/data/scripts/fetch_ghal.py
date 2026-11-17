"""Selective extraction from figshare 29297999 (Ghalamsiah 2025, CC BY 4.0) via HTTP range reads. Resumable."""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import json, os, sys, zipfile, fsspec, shutil
D = f'{BLDG_ROOT}/data/raw/faults/ghalamsiah2025'
a = json.load(open(f'{BLDG_ROOT}/data/tmp/fs_29297999.json')); json.dump(a, open(f'{D}/figshare_29297999.json', 'w'), indent=1)
u = a['files'][0]['download_url']
skip_data = tuple(sys.argv[1:]) or ('04_G36-1wk', '06_G36-Degrad')
fs = fsspec.filesystem('https', client_kwargs={'trust_env': True})
z = zipfile.ZipFile(fs.open(u, block_size=4 * 2**20))
for i in z.infolist():
    if i.is_dir(): continue
    rel = i.filename.split('/', 1)[1]
    top = rel.split('/')[0]
    if top in skip_data and not rel.lower().endswith(('.pdf', '.xls', '.xlsx', '.txt')): continue
    out = f'{D}/{rel}'
    if os.path.exists(out) and os.path.getsize(out) == i.file_size: continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with z.open(i) as src, open(out + '.tmp', 'wb') as dst: shutil.copyfileobj(src, dst, 2**20)
    os.replace(out + '.tmp', out); print(rel, i.file_size, flush=True)
print('DONE')
