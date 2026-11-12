"""ChirpNet's O3 exam: the released v2 cascade and posterior network, unchanged, on eight O3 events named in advance
(models/chirpnet/NOTEBOOK.md, 30 Oct 2026), with time-slid real O3 background from the same files.

Same protocol as the O1/O2 exam (realeval.py): 32 s of 4 kHz H1 + L1 strain per event from GWOSC; the event's
statistic is the cascade's best window whose merger time lands within 0.5 s of the catalogue GPS time; background is
L1 slid against H1 by whole seconds (3-25 s, circular within 2-30 s of the file), with windows holding the event in
either detector removed. Truth: the GWOSC catalogue (detector-frame chirp mass = source-frame x (1 + z)).
Virgo is not used (ChirpNet reads two detectors). Nothing here changes the model.

    python o3_exam.py                 # downloads ~16 MB to work/data_o3/, writes logs/o3_exam.json
"""
import _paths as _P  # portable paths (see _paths.py)
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, _P.MODEL)
import chirpnet as C  # noqa: E402

EVENTS = ['GW190408_181802', 'GW190412', 'GW190521', 'GW190828_063405', 'GW191204_171526',
          'GW200129_065458', 'GW200224_222234', 'GW200311_115853']
API = 'https://gwosc.org/api/v2'
DATA = os.path.join(_P.WORK, 'data_o3')
UA = {'User-Agent': 'ml-ratchet-exam/1.0'}
TRAINED_MC = (5.5, 45.0)   # detector-frame chirp-mass range of the training set (MODEL_CARD.md)


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def fetch(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(path + '.part', 'wb') as f:
            f.write(r.read())
        os.replace(path + '.part', path)
    return path


def catalogue(name):
    """The latest catalogue version of an event, its GPS time, parameters and H1/L1 32 s 4 kHz HDF5 files."""
    ev = get(f'{API}/events/{name}')
    vers = sorted(ev['versions'], key=lambda v: v['version'])
    v = next((x for x in reversed(vers) if x['catalog'].startswith(('GWTC-3', 'GWTC-2.1'))), vers[-1])
    d = get(v['detail_url'])
    pars = get(d['parameters_url'])
    rows = pars.get('results', []) if isinstance(pars, dict) else pars
    files = get(d['strain_files_url'])
    files = files.get('results', files) if isinstance(files, dict) else files
    pick = {}
    for f in files:
        if f.get('detector') in ('H1', 'L1') and int(f.get('sample_rate_kHz', f.get('sample_rate', 4))) in (4, 4096) \
                and int(f.get('duration', 32)) == 32 and str(f.get('file_format', f.get('format', 'hdf5'))).lower().startswith('hdf'):
            pick[f['detector']] = f.get('download_url') or f.get('url')
    return {'catalog': d['catalog'], 'version': d['version'], 'gps': float(d['gps']), 'params': rows, 'files': pick}


def param(sets, key):
    """The preferred parameter-estimation set's value of a parameter: median and 90% bounds."""
    sets = sets if isinstance(sets, list) else []
    pref = [s for s in sets if s.get('is_preferred')] + [s for s in sets if 'pe' in s.get('name', '').lower()]
    for s in pref:
        for r in s.get('parameters', []):
            if r.get('name') == key and r.get('best') is not None:
                b = r['best']
                return {'median': b, 'lo': b - abs(r.get('lower_error') or 0), 'hi': b + abs(r.get('upper_error') or 0)}
    return None


def whitened(path):
    x, t0 = C.load_gwosc(path)
    w, _ = C.gate(C.whiten(x, C.FS_IN))
    return w, t0


def windows(H, L, starts):
    return np.stack([np.stack([H[s:s + C.WIN], L[s:s + C.WIN]]) for s in starts]).astype(np.float32)


def run(net, x, bs=64):
    o, c = zip(*(net.statistic(x[i:i + bs]) for i in range(0, len(x), bs)))
    return np.concatenate(o), np.concatenate(c)


def main():
    net = C.ChirpNet(os.path.join(_P.MODEL, 'chirpnet_weights.npz'))
    pe = C.ChirpPE(os.path.join(_P.MODEL, 'chirpnet_pe_weights.npz'))
    thr = net.meta['threshold']
    res, bg = {'events': {}, 'protocol': __doc__.split('\n\n')[0]}, []
    for name in EVENTS:
        t_start = time.time()
        cat = catalogue(name)
        paths = {det: fetch(url, os.path.join(DATA, os.path.basename(url))) for det, url in cat['files'].items()}
        if set(paths) != {'H1', 'L1'}:
            res['events'][name] = {'error': f'H1/L1 32 s files not found: {sorted(paths)}'}
            print(name, 'missing files'); continue
        H, t0 = whitened(paths['H1']); L, _ = whitened(paths['L1'])
        n = min(len(H), len(L)); tev = cat['gps'] - t0
        starts = np.arange(2 * C.FS, n - 2 * C.FS - C.WIN, int(0.0625 * C.FS))
        o, chk = run(net, windows(H, L, starts))
        t_m = starts / C.FS + o[:, 5]
        near = np.abs(t_m - tev) < 0.5
        k = int(np.argmax(np.where(near, o[:, 0], -1e9))) if near.any() else int(np.argmax(o[:, 0]))
        hits = net.scan(H, L, gps_start=t0, whitened=True)
        hit = next((h for h in hits if abs(h['gps'] - cat['gps']) < 0.5), None)
        mcs, z = param(cat['params'], 'chirp_mass_source'), param(cat['params'], 'redshift')
        mc_det = mcs['median'] * (1 + z['median']) if mcs and z and mcs['median'] is not None and z['median'] is not None else None
        e = {'catalog': f"{cat['catalog']} v{cat['version']}", 'gps': cat['gps'], 'score': float(o[k, 0]),
             'checker': None if np.isnan(chk[k]) else float(chk[k]), 'found': hit is not None, 'threshold': thr,
             'mc_det_catalog': mc_det, 'in_training_range': bool(mc_det and TRAINED_MC[0] <= mc_det <= TRAINED_MC[1])}
        if hit:
            e.update({'mc': hit['chirp_mass'], 'mc_68': hit['chirp_mass_68'], 'dt_ms': 1000 * (hit['gps'] - cat['gps']),
                      'h1_alone': hit['h1_alone'], 'l1_alone': hit['l1_alone'], 'far_o1o2_bg_per_hour': hit['far_per_hour']})
            if mc_det:
                e['mc_rel_err'] = hit['chirp_mass'] / mc_det - 1
                e['mc_in_68'] = bool(hit['chirp_mass_68'][0] <= mc_det <= hit['chirp_mass_68'][1])
            s = pe.weigh(*(C.load_gwosc(paths[d])[0] for d in ('H1', 'L1')), t0, hit['gps'])['summary_90']
            e['posterior_90'] = {k2: list(map(float, v)) for k2, v in s.items()}
            inside = {}
            for ours, theirs in (('m1_source', 'mass_1_source'), ('m2_source', 'mass_2_source'), ('chirp_mass_source', 'chirp_mass_source'),
                                 ('chi_eff', 'chi_eff'), ('luminosity_distance_mpc', 'luminosity_distance')):
                ref = param(cat['params'], theirs)
                if ref and ref['median'] is not None and ours in s:
                    inside[ours] = {'catalog': ref['median'], 'inside_90': bool(s[ours][0] <= ref['median'] <= s[ours][2])}
            e['posterior_vs_catalog'] = inside
        # time-slid background from this event's own files
        core = slice(2 * C.FS, 30 * C.FS); hc, lc = H[core], L[core]; m = len(hc); te = tev - 2
        for ks in range(3, 26):
            ls = np.roll(lc, ks * C.FS); tl = (te + ks) % (m / C.FS)
            keep = [s for s in range(0, m - C.WIN, 256)
                    if not (s / C.FS - 0.5 < te < s / C.FS + 2.5) and not (s / C.FS - 0.5 < tl < s / C.FS + 2.5)]
            bg.append(run(net, windows(hc, ls, keep))[0][:, 0])
        res['events'][name] = e
        print(f"{name:16s} score {e['score']:6.2f} found {e['found']!s:5s} Mc {e.get('mc', float('nan')):6.1f} "
              f"(catalog {mc_det or float('nan'):6.1f}) {time.time() - t_start:5.0f} s", flush=True)
    b = np.concatenate(bg); live = len(b) * 256 / C.FS
    res['background'] = {'n_windows': int(len(b)), 'livetime_s': float(live), 'max': float(b.max()),
                         'q999': float(np.quantile(b, 0.999)), 'n_above_threshold': int(np.sum(b >= thr))}
    for e in res['events'].values():
        if 'score' in e:
            e['n_o3_bg_louder'] = int(np.sum(b >= e['score']))
            e['far_o3_bg_per_hour_ub'] = float((e['n_o3_bg_louder'] + 1) / (live / 3600))
    out = os.path.join(_P.SRC, 'logs', 'o3_exam.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print(f"background: {len(b)} windows, {live / 3600:.2f} h, loudest {b.max():.2f}, above threshold {res['background']['n_above_threshold']}")
    print('wrote', out)


if __name__ == '__main__':
    main()
