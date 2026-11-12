"""Download the CWRU Bearing Data Center files listed on its four data pages, build a catalogue, and store each recording
as float32 npz (DE, FE, BA channels when present, the logged rpm). The raw .mat is deleted after conversion to save disk.
Source: https://engineering.case.edu/bearingdatacenter (Case Western Reserve University)."""
import _paths as _P  # portable paths (see _paths.py)
import re, os, json, sys, time, subprocess, numpy as np, scipy.io as sio
D = _P.WORK + '/data/'; PAGES = {'normal': ('normal-baseline-data', None, None),
    'de12': ('12k-drive-end-bearing-fault-data', 'DE', 12000), 'de48': ('48k-drive-end-bearing-fault-data', 'DE', 48000),
    'fe12': ('12k-fan-end-bearing-fault-data', 'FE', 12000)}
NOM_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

def catalogue():
    cat = []
    for s, (page, loc, fs) in PAGES.items():
        h = open(D + 'pages/' + page + '.html').read()
        for url, txt in re.findall(r'href="(https://engineering.case.edu/sites/default/files/\d+\.mat)"[^>]*>([^<]+)<', h):
            num = int(url.split('/')[-1][:-4]); txt = txt.strip()
            if s == 'normal':
                cat.append(dict(id=num, set='normal', cls='N', size=0.0, pos=None, load=int(txt.split('_')[1]), fault_end=None, url=url, label=txt)); continue
            m = re.match(r'(IR|B|OR)(\d{3})(?:@(\d+))?_(\d)', txt)
            cls = {'IR': 'IR', 'B': 'B', 'OR': 'OR'}[m.group(1)]
            cat.append(dict(id=num, set=s, cls=cls, size=int(m.group(2)) / 1000, pos=int(m.group(3)) if m.group(3) else None, load=int(m.group(4)),
                            fault_end=loc, fs=fs, url=url, label=txt))
    return cat

def convert(path, num):
    m = sio.loadmat(path); keys = [k for k in m if not k.startswith('__')]; out = {}; note = []
    pre = 'X%03d' % num
    for ch in ('DE', 'FE', 'BA'):
        ks = [k for k in keys if k.endswith('_%s_time' % ch)]
        mine = [k for k in ks if k.startswith(pre)]
        if not ks: continue
        if not mine: note.append('no %s var with own number; used %s' % (ch, ks[0]))
        out[ch] = m[(mine or ks)[0]].ravel().astype(np.float32)
    rk = [k for k in keys if k.endswith('RPM')]; rpm = float(np.ravel(m[rk[0]])[0]) if rk else float('nan')
    if len([k for k in keys if k.endswith('_DE_time')]) > 1: note.append('several DE vars: ' + ','.join(k for k in keys if k.endswith('_DE_time')))
    return out, rpm, note

if __name__ == '__main__':
    cat = catalogue(); os.makedirs(D + 'cwru', exist_ok=True); print(len(cat), 'files listed')
    for c in cat:
        f = D + 'cwru/%d.npz' % c['id']
        if os.path.exists(f):
            z = np.load(f); c.update(json.loads(str(z['meta']))); continue
        tmp = D + 'cwru/%d.mat' % c['id']
        for k in range(5):
            r = subprocess.run(['curl', '-sS', '-m', '180', '-o', tmp, c['url']])
            try:
                out, rpm, note = convert(tmp, c['id']); break
            except Exception as e:
                print('retry', c['id'], e, flush=True); time.sleep(5)
        meta = dict(rpm_logged=rpm, n={k: len(v) for k, v in out.items()}, note=note); c.update(meta)
        np.savez(f, meta=json.dumps(meta), **out); os.remove(tmp); print(c['id'], c['label'], c['set'], meta, flush=True)
    json.dump(cat, open(D + 'cwru_catalogue.json', 'w'), indent=1)
