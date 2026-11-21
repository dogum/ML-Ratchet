"""Stream WildfireSpreadTS (Zenodo 8006177, CC BY 4.0) fire by fire with HTTP range reads; keep compact arrays.

Usage:  python3 -I fetch_wsts.py meta            # one mid-fire day per fire -> proc/wsts/meta/<fire>.npz (static layers)
        python3 -I fetch_wsts.py full [order.txt] # all days per fire -> proc/wsts/fires/<fire>.npz (order: dev first)
Resumable: existing outputs are skipped. Never stores the 48 GB zip; ~3.5 MB per tif in memory at a time.
"""
import sys, os, json, time, collections
import numpy as np, rasterio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsts_member as W
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root

PROC = f'{_DATA}/data/proc/wsts'
WX = ['total precipitation', 'wind speed', 'wind direction', 'minimum temperature', 'maximum temperature',
      'energy release component', 'specific humidity', 'pdsi', 'total_precipitation_surface_sum',
      'forecast wind speed', 'forecast wind direction', 'forecast temperature', 'forecast specific humidity']
WX_KEYS = ['pr_mm', 'vs_ms', 'th_deg', 'tmin_K', 'tmax_K', 'erc', 'sph_kgkg', 'pdsi', 'gfs_tp_mm',
           'gfs_ws_ms', 'gfs_wd_deg', 'gfs_t_C', 'gfs_q_kgkg']
STRIDE = 8  # weather sampled every 8 px (3 km; gridMET native ~4.6 km, GFS ~28 km)


def fires():
    url, mem = W.index()
    d = collections.defaultdict(list)
    for n in mem:
        if n.endswith('.tif'):
            y, f, day = n.split('/'); d[f].append((day[:-4], n))
    return {f: sorted(v) for f, v in d.items()}


def read(name):
    b = W.fetch(name)
    with rasterio.MemoryFile(b) as mf, mf.open() as s:
        return s.read(), list(s.descriptions), s.crs.to_epsg(), tuple(s.transform)[:6]


def header(name):
    """Partial read: the TIFF IFD sits at byte 8, so the first ~64 KB of the deflate stream holds the geo tags."""
    import zlib, struct
    url, mem = W.index(); m = mem[name]
    for i in range(8):
        r = W.S.get(url, headers={'Range': f"bytes={m['off']}-{m['off'] + 30 + 1024 + 65535}"}, timeout=(30, 120))
        if r.status_code == 206 and r.content[:4] == b'PK\x03\x04': break
        time.sleep(10 * (i + 1))
    b = r.content; nl, el = struct.unpack('<HH', b[26:30])
    raw = zlib.decompressobj(-15).decompress(b[30 + nl + el:])
    # parse first IFD
    n = struct.unpack('<H', raw[8:10])[0]; tags = {}
    for i in range(n):
        t, typ, cnt, val = struct.unpack('<HHII', raw[10 + 12 * i: 22 + 12 * i])
        size = {3: 2, 4: 4, 12: 8, 16: 8}.get(typ, 1) * cnt
        if size > 4:
            fmt = {3: 'H', 4: 'I', 12: 'd', 16: 'Q'}.get(typ, 'B')
            tags[t] = struct.unpack('<%d%s' % (cnt, fmt), raw[val: val + size]) if val + size <= len(raw) else None
        else:
            tags[t] = val if typ != 3 else val & 0xFFFF
    W_, H_ = tags[256], tags[257]
    sc, tp = tags[33550], tags[33922]
    gk = tags.get(34735)
    epsg = None
    if gk:
        for j in range(4, len(gk), 4):
            if gk[j] == 3072: epsg = gk[j + 3]
    return dict(width=W_, height=H_, transform=[sc[0], 0.0, tp[3], 0.0, -sc[1], tp[4]], epsg=epsg)


def do_meta(f, days):
    out = f'{PROC}/meta/{f}.json'
    if os.path.exists(out): return
    h = header(days[0][1])
    m = dict(fire=f, year=int(days[0][1].split('/')[0]), days=[d for d, _ in days], **h)
    json.dump(m, open(out, 'w'))
    return
    mid = days[len(days) // 2][1]
    mid = days[len(days) // 2][1]
    a, desc, epsg, tr = read(mid)
    ix = {d: i for i, d in enumerate(desc)}
    af = a[ix['active fire']]
    m = dict(fire=f, year=int(mid.split('/')[0]), days=[d for d, _ in days], epsg=epsg, transform=tr,
             shape=list(a.shape[1:]), meta_day=mid)
    np.savez_compressed(out, meta=json.dumps(m),
                        elev=np.nan_to_num(a[ix['elevation']], nan=-9999).astype(np.int16),
                        slope=a[ix['slope']].astype(np.float16), aspect=a[ix['aspect']].astype(np.float16),
                        lc=np.nan_to_num(a[ix['LC_Type1']], nan=0).astype(np.uint8),
                        ndvi=np.nan_to_num(a[ix['NDVI_last']], nan=-32768).astype(np.int16),
                        af_mid=np.nan_to_num(af, nan=-1).astype(np.int16))


def do_full(f, days):
    out = f'{PROC}/fires/{f}.npz'
    if os.path.exists(out): return
    afs, wx, wxm, nanfrac = [], [], [], []
    for d, n in days:
        a, desc, epsg, tr = read(n)
        ix = {k: i for i, k in enumerate(desc)}
        afs.append(np.nan_to_num(a[ix['active fire']], nan=-1).astype(np.int16))
        nanb = np.full(a.shape[1:], np.nan, np.float32)  # some years lack a band (e.g. GFS precip): NaN
        B = [a[ix[k]] if k in ix else nanb for k in WX]
        w = np.stack([b[STRIDE // 2::STRIDE, STRIDE // 2::STRIDE] for b in B])
        wx.append(w.astype(np.float32))
        with np.errstate(all='ignore'):
            wxm.append([float(np.nanmean(b)) if np.isfinite(b).any() else np.nan for b in B])
        nanfrac.append([float(np.isnan(b).mean()) for b in B])
    wx = np.stack(wx)
    np.savez_compressed(out, days=np.array([d for d, _ in days]), af=np.stack(afs), wx=wx.astype(np.float32),
                        wx_mean=np.array(wxm, np.float32), wx_nanfrac=np.array(nanfrac, np.float32),
                        wx_keys=np.array(WX_KEYS), stride=STRIDE, epsg=epsg, transform=np.array(tr))


if __name__ == '__main__':
    mode = sys.argv[1]
    F = fires()
    order = sorted(F)
    if len(sys.argv) > 2:
        pri = [l.strip() for l in open(sys.argv[2]) if l.strip()]
        order = pri + [f for f in order if f not in set(pri)]
    t0 = time.time()
    for i, f in enumerate(order):
        try:
            (do_meta if mode == 'meta' else do_full)(f, F[f])
        except Exception as e:
            print('FAIL', f, repr(e), flush=True); continue
        if i % 20 == 0: print(i, f, f'{time.time() - t0:.0f}s', flush=True)
    print('DONE', mode, flush=True)
