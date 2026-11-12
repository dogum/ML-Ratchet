"""Download TESS SPOC 2-minute light curves and keep only what TransitNet reads: time, PDCSAP flux and error, quality,
flux-weighted centroids, and the star's TIC header values. Each FITS is read with astropy and deleted."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, time, concurrent.futures as cf, numpy as np, urllib.request
from astropy.io import fits
D = _P.WORK + '/data/'; LC = D + 'lc/'; os.makedirs(LC, exist_ok=True)
URL = 'https://mast.stsci.edu/api/v0.1/Download/file?uri='
HDR = ('TESSMAG', 'TEFF', 'LOGG', 'RADIUS', 'MH', 'SECTOR', 'CAMERA', 'CCD', 'CROWDSAP', 'FLFRCSAP', 'RA_OBJ', 'DEC_OBJ')

def one(job):
    tic, sector, uri = job; out = f'{LC}{tic}_s{sector:03d}.npz'
    if os.path.exists(out): return out, 'have'
    tmp = f'{LC}.{tic}_{sector}.fits'
    for a in range(3):
        try:
            urllib.request.urlretrieve(URL + uri, tmp); break
        except Exception as e:
            err = str(e); time.sleep(2 + 3 * a)
    else:
        return out, 'fail ' + err
    try:
        with fits.open(tmp, memmap=False) as h:
            d = h[1].data; hd = {**{k: h[0].header.get(k) for k in HDR}, **{k: h[1].header.get(k) for k in ('CROWDSAP', 'FLFRCSAP', 'CDPP1_0', 'CDPP2_0')}}
            np.savez_compressed(out, time=np.asarray(d['TIME'], np.float64), flux=np.asarray(d['PDCSAP_FLUX'], np.float32), err=np.asarray(d['PDCSAP_FLUX_ERR'], np.float32),
                                q=np.asarray(d['QUALITY'], np.int32), c1=np.asarray(d['MOM_CENTR1'], np.float32), c2=np.asarray(d['MOM_CENTR2'], np.float32),
                                hdr=json.dumps({k: (v if isinstance(v, (int, float, str)) or v is None else str(v)) for k, v in hd.items()}))
        return out, 'ok'
    except Exception as e:
        return out, 'bad ' + str(e)
    finally:
        if os.path.exists(tmp): os.remove(tmp)

def run(jobs, workers=8):
    t = time.time(); n = 0; bad = []
    with cf.ThreadPoolExecutor(workers) as ex:
        for out, st in ex.map(one, jobs):
            n += 1
            if not st.startswith(('ok', 'have')): bad.append((out, st))
            if n % 200 == 0: print(n, len(jobs), '%.0fs' % (time.time() - t), 'bad', len(bad), flush=True)
    print('done', n, 'bad', len(bad), '%.0fs' % (time.time() - t), bad[:5], flush=True)

if __name__ == '__main__':
    jobs = [tuple(j) for j in json.load(open(sys.argv[1]))]; run(jobs, int(sys.argv[2]) if len(sys.argv) > 2 else 8)
