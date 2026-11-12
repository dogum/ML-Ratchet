"""Paderborn University KAt Bearing DataCenter (Lessmeier et al. 2016), https://groups.uni-paderborn.de/kat/BearingDataCenter/
Download one bearing archive at a time, keep the first NPER recordings of each of the 4 operating conditions:
vibration_1 (housing accelerometer, 64 kHz) resampled to 12 kHz, mean measured speed, and the bearing's description PDF.
The .rar is deleted afterwards. Uses libarchive (system libarchive.so.13) through libarchive-c."""
import _paths as _P  # portable paths (see _paths.py)
import os, sys, json, subprocess, io, numpy as np, scipy.io as sio, scipy.signal as ss, libarchive
D = _P.WORK + '/data/'; URL = 'https://groups.uni-paderborn.de/kat/BearingDataCenter/%s.rar'; NPER = 10
BEARINGS = ['K001', 'K002', 'K003', 'K004', 'K005', 'K006', 'KA01', 'KA03', 'KA05', 'KA06', 'KA07', 'KA08', 'KA09', 'KI01', 'KI03',
            'KI05', 'KI07', 'KI08', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI04', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21', 'KB23', 'KB24', 'KB27']

def one(b):
    out = D + 'pb/%s.npz' % b
    if os.path.exists(out): return
    rar = D + 'pb_raw/%s.rar' % b
    if not os.path.exists(rar) or os.path.getsize(rar) < 1e8:
        subprocess.run(['curl', '-sS', '-m', '600', '-o', rar, URL % b], check=True)
    X, meta = [], []
    with libarchive.file_reader(rar) as a:
        for e in a:
            name = e.pathname.split('/')[-1]
            if name.endswith('.pdf'):
                open(D + 'pb/docs/' + name, 'wb').write(b''.join(e.get_blocks())); continue
            if not name.endswith('.mat'): continue
            parts = name[:-4].split('_')     # N15_M07_F10_K001_1
            if int(parts[-1]) > NPER: continue
            try: m = sio.loadmat(io.BytesIO(b''.join(e.get_blocks())), squeeze_me=True, struct_as_record=False)
            except Exception as ex: print('  unreadable', name, ex, flush=True); continue
            s = m[[k for k in m if not k.startswith('__')][0]]
            Y = {y.Name: np.asarray(y.Data, float) for y in np.atleast_1d(s.Y)}
            v = ss.resample_poly(Y['vibration_1'] - Y['vibration_1'].mean(), 3, 16)     # 64 kHz -> 12 kHz (polyphase, anti-aliased)
            X.append(v[:47000].astype(np.float32)); meta.append(dict(cond='_'.join(parts[:3]), rep=int(parts[-1]), rpm=float(np.mean(Y['speed'])),
                                                                 rpm_sd=float(np.std(Y['speed'])), torque=float(np.mean(Y['torque'])), force=float(np.mean(Y['force']))))
    o = np.argsort([m['cond'] + '%03d' % m['rep'] for m in meta]); X = np.stack([X[i] for i in o]); meta = [meta[i] for i in o]
    np.savez(out, x=X, meta=json.dumps(meta)); os.remove(rar); print(b, X.shape, meta[0], flush=True)

if __name__ == '__main__':
    os.makedirs(D + 'pb/docs', exist_ok=True); os.makedirs(D + 'pb_raw', exist_ok=True)
    for b in (sys.argv[1:] or BEARINGS): one(b)
