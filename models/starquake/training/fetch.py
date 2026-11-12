"""Download Kepler long-cadence (29.4 min) light curves from the public MAST bucket (stpubdata on S3) and keep only what
StarQuake reads: cadence number, time, PDCSAP flux and quality flag, per quarter. Each FITS file is read and deleted."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, time, re, concurrent.futures as cf, numpy as np, urllib.request
from astropy.io import fits
D = _P.WORK + '/data/'; LC = D + 'lc/'; os.makedirs(LC, exist_ok=True)
S3 = 'https://stpubdata.s3.amazonaws.com/'

def get(url, path=None):
    for a in range(4):
        try:
            if path: urllib.request.urlretrieve(url, path); return True
            return urllib.request.urlopen(url, timeout=60).read().decode()
        except Exception as e:
            err = e; time.sleep(2 + 4 * a)
    raise err

def one(kic):
    out = f'{LC}{kic}.npz'
    if os.path.exists(out): return kic, 'have'
    k = f'{kic:09d}'; pre = f'kepler/public/lightcurves/{k[:4]}/{k}/'
    try:
        keys = [x for x in re.findall(r'<Key>([^<]+)</Key>', get(S3 + '?list-type=2&prefix=' + pre)) if x.endswith('_llc.fits')]
    except Exception as e:
        return kic, 'list fail ' + str(e)
    C = {a: [] for a in ('cad', 'time', 'flux', 'q', 'quarter')}; hdr = {}
    for key in sorted(keys):
        tmp = f'{LC}.{kic}_{os.path.basename(key)}'
        try:
            get(S3 + key, tmp)
            with fits.open(tmp, memmap=False) as h:
                d = h[1].data; qq = h[0].header['QUARTER']
                C['cad'].append(np.asarray(d['CADENCENO'], np.int32)); C['time'].append(np.asarray(d['TIME'], np.float64)); C['flux'].append(np.asarray(d['PDCSAP_FLUX'], np.float32))
                C['q'].append(np.asarray(d['SAP_QUALITY'], np.int32)); C['quarter'].append(np.full(len(d), qq, np.int8))
                if not hdr: hdr = {a: h[0].header.get(a) for a in ('KEPMAG', 'TEFF', 'LOGG', 'RADIUS', 'RA_OBJ', 'DEC_OBJ')}
        except Exception as e:
            print('bad', kic, key, e, flush=True)
        finally:
            if os.path.exists(tmp): os.remove(tmp)
    if not C['cad']: return kic, 'no data'
    np.savez_compressed(out, **{a: np.concatenate(v) for a, v in C.items()}, hdr=json.dumps(hdr)); return kic, 'ok %d' % len(keys)

if __name__ == '__main__':
    sp = json.load(open(D + 'split.json')); kics = [k for g in sys.argv[1].split(',') for k in sp[g]]; t = time.time(); bad = []
    with cf.ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as ex:
        for n, (k, st) in enumerate(ex.map(one, kics), 1):
            if not st.startswith(('ok', 'have')): bad.append((k, st))
            if n % 25 == 0: print(n, len(kics), '%.0fs' % (time.time() - t), 'bad', len(bad), flush=True)
    print('done', len(kics), 'bad', bad, '%.0fs' % (time.time() - t), flush=True)
