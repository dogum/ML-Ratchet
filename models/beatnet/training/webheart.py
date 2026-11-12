"""Pack one reference patient's voxel heart (and noise snippets, recorder filters) for the browser simulator."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, base64, numpy as np, scipy.signal as ss
sys.path.insert(0, _P.SRC); import sim, heart as Hm, strip as S
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

def pack(seed=4, version=2):
    sim.set_version(version); p = sim.Patient(seed, bbb='none'); h = p.h; p.vmap('normal'); p.vdipole(p.vmap('normal'), 1.0)
    lo = (h.pos[0] - h.ijk[0] * Hm.HV).tolist()
    NB = S.noise_bank(); noise = {}
    for k in ('bw', 'em', 'ma'):
        x, r = NB[k]; seg = x[0, 100000:100000 + 360 * 25]; noise[k] = dict(d=b64(np.round(seg * 200).astype(np.int16)), r=float(r[0] * 200))
    hp = ss.butter(1, 0.3, 'highpass', fs=360); lp = ss.butter(2, 80, fs=360)
    return dict(shape=list(h.shape), lo=lo, ijk=b64(h.ijk.astype(np.int16)), cls=base64.b64encode(h.cls.astype(np.uint8).tobytes()).decode(), e=b64(h.e.astype(np.float32)), b=b64(h.b.astype(np.float32)),
                roots={k: int(v) for k, v in p.roots.items()}, root_t={k: float(v) for k, v in p.root_t.items()}, sinus=int(p.sinus), avn=int(p.avn), apex=float(h.apex),
                M=p.M.tolist(), cv={k: float(v) for k, v in p.cv.items()}, rep={k: float(v) for k, v in p.rep.items()}, PR=float(p.PR), atr_gain=float(p.atr_gain), vgain=float(p.vgain),
                scale_target=float(p.scale_target), rec=dict(hp=[list(hp[0]), list(hp[1])], lp=[list(lp[0]), list(lp[1])]), noise=noise, seed=seed, version=version), p

if __name__ == '__main__':
    pk, p = pack(); json.dump(pk, open(_P.LAB + '/heart.json', 'w')); print('heart.json', len(json.dumps(pk)) // 1024, 'KB', p.h.n, 'voxels')
    # reference values for the JS check
    tn = p.vmap('normal'); f = int(np.nonzero(p.vent)[0][len(np.nonzero(p.vent)[0]) // 3]); tv = p.vmap('pvc', f, 3.0)
    ref = dict(focus=f, normal=p.vdipole(tn, 0.8, 'n').tolist(), pvc=p.vdipole(tv, 0.6, ('pvc', f)).tolist(), qw_n=p.qrs_width(tn), qw_v=p.qrs_width(tv), atr=p.adip(p.sinus).tolist(),
               myo=p.cv['myo'], fast=p.cv['fast'])
    json.dump(ref, open(_P.LAB + '/heart_ref.json', 'w'))
