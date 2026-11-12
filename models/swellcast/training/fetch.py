"""Download NDBC historical spectral wave files (swden = energy density, swdir = mean direction alpha1, swr1 = r1) and
store each buoy-year on a fixed hourly x frequency grid:
  E  (n_hours, NF) energy density m^2/Hz (NaN = missing)
  C, S (n_hours, NF) r1*cos(alpha1), r1*sin(alpha1), alpha1 = compass direction waves come FROM (NaN = missing)
Source: https://www.ndbc.noaa.gov/data/historical/{swden,swdir,swr1}/<id>{w,d,j}<year>.txt.gz (NOAA NDBC, public domain)."""
import os, sys, gzip, io, time, subprocess, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); RAW = os.path.join(ROOT, 'data/ndbc/raw'); OUT = os.path.join(ROOT, 'data/buoy')
FGRID = 0.0325 + 0.005 * np.arange(34)          # 0.0325 ... 0.1975 Hz (native NDBC bins up to 0.0925, interpolated above)
BUOYS = ['51101', '51001', '51004', '51000', '51002', '51003', '46047', '46028', '46011', '46025', '46086', '46069', '46026', '46042', '46059',
         '46006', '46002', '41047', '41046', '41040', '41010', '41001', '44014', '44008', '44011', '42060', '32012']
DIRS = {'w': 'swden', 'd': 'swdir', 'j': 'swr1'}

def get(b, k, y):
    os.makedirs(RAW, exist_ok=True); fn = os.path.join(RAW, f'{b}{k}{y}.txt.gz')
    if not os.path.exists(fn):
        url = f'https://www.ndbc.noaa.gov/data/historical/{DIRS[k]}/{b}{k}{y}.txt.gz'
        r = subprocess.run(['curl', '-sS', '-f', '-m', '120', '-o', fn + '.part', url], capture_output=True)
        if r.returncode != 0: return None
        os.rename(fn + '.part', fn); time.sleep(0.3)
    return fn

def parse(fn, year):
    """-> hour index within year (int), freqs, values (n, nf) with NaN for missing"""
    txt = gzip.open(fn, 'rt', errors='ignore').read().splitlines()
    hdr = [l for l in txt if l.startswith('#') or l.startswith('YY') or l.startswith('YYYY')]
    h = (hdr[0] if hdr else txt[0]).replace('#', '').split()
    nt = 5 if h[4].lower() in ('mm', 'mn') else 4
    fr = np.array([float(x) for x in h[nt:]])
    rows = [l.split() for l in txt if l and not (l.startswith('#') or l[0].isalpha())]
    rows = [r for r in rows if len(r) == nt + len(fr)]
    if not rows: return None
    A = np.array(rows, float); yy = A[:, 0].astype(int); yy = np.where(yy < 100, yy + 1900, yy)
    t = (np.array([np.datetime64(f'{a:04d}-{b:02d}-{c:02d}') for a, b, c in zip(yy, A[:, 1].astype(int), A[:, 2].astype(int))]).astype('datetime64[h]').astype(np.int64)
         + A[:, 3] + (A[:, 4] / 60 if nt == 5 else 0))
    V = A[:, nt:]; V[(V >= 98.9) & (V <= 99.01)] = np.nan; V[V >= 998.9] = np.nan
    return np.round(t).astype(np.int64), fr, V

def to_grid(fr, V, logint):
    out = np.full((len(V), len(FGRID)), np.nan)
    for i, v in enumerate(V):
        ok = np.isfinite(v) & (fr > 0.025)
        if ok.sum() < 5: continue
        x = np.log(np.maximum(v[ok], 1e-4)) if logint else v[ok]; y = np.interp(FGRID, fr[ok], x, left=np.nan, right=np.nan)
        out[i] = np.exp(y) if logint else y
    return out

def buoy_year(b, y):
    out = os.path.join(OUT, f'{b}_{y}.npz')
    if os.path.exists(out): return 'have'
    fw = get(b, 'w', y)
    if fw is None: return 'no swden'
    pw = parse(fw, y)
    if pw is None: return 'empty'
    h0 = np.datetime64(f'{y}-01-01', 'h').astype(np.int64); n = int(np.datetime64(f'{y + 1}-01-01', 'h').astype(np.int64) - h0)
    E = np.full((n, len(FGRID)), np.nan, np.float32); C = E.copy(); S = E.copy()
    t, fr, V = pw; k = (t >= h0) & (t < h0 + n); E[t[k] - h0] = to_grid(fr, V[k], True)
    fd, fj = get(b, 'd', y), get(b, 'j', y)
    if fd and fj:
        pd_, pj = parse(fd, y), parse(fj, y)
        if pd_ is not None and pj is not None:
            td, frd, D = pd_; tj, frj, R = pj
            if np.nanmax(R) > 1.5: R = R / 100.0
            R = np.clip(R, 0, 1); common, idd, idj = np.intersect1d(td, tj, return_indices=True)
            a = np.radians(D[idd]); r = R[idj]; kk = (common >= h0) & (common < h0 + n)
            C[common[kk] - h0] = to_grid(frd, (r * np.cos(a))[kk], False); S[common[kk] - h0] = to_grid(frd, (r * np.sin(a))[kk], False)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(out, E=E.astype(np.float32), C=C.astype(np.float16), S=S.astype(np.float16), h0=h0, f=FGRID)
    for f in (fw, fd, fj):
        if f and os.path.exists(f): os.remove(f)
    return f'ok {np.isfinite(E[:, 5]).mean():.2f} dir {np.isfinite(C[:, 5]).mean():.2f}'

if __name__ == '__main__':
    sys.path.insert(0, HERE); import catalog; av = catalog.years_available()
    y0, y1 = int(sys.argv[1]) if len(sys.argv) > 1 else 2008, int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    for b in BUOYS:
        for y in range(y0, y1 + 1):
            if y in av.get(b, {}).get('w', set()): print(b, y, buoy_year(b, y), flush=True)
