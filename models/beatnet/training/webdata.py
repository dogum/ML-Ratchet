"""Page data: network weights (float32), one 60 s segment of every DS2 record (fixed random draw within the record:
among the 60 s windows that hold the record's rarest beat class, if any), Python predictions on those segments (to check the
JS port), exam / ratchet / physics tables. Writes web/data.json."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, base64, numpy as np, wfdb
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.WORK + '/release'); import common as C
from beatnet import BeatNet
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
SEG = 60 * 360; PADS = 30 * 360
_z = np.load(C.D + 'mitdb_beats.npz'); RH_REC, RH_POS = _z['rec'], _z['pos']; RH = np.load(C.D + 'mitdb_rhythm.npy')        # 60 s shown, 30 s of context either side for the template and RR features

def weights_pack(npz):
    z = np.load(npz); meta = json.loads(str(z['meta'])); man = []; parts = []; off = 0
    for k in z.files:
        if k == 'meta': continue
        a = z[k].astype(np.float32); man.append(dict(k=k, shape=list(a.shape), off=off)); off += a.size; parts.append(a.ravel())
    return dict(f32=True, b64=b64(np.concatenate(parts)), man=man, meta=meta)

def segments(net, seed=17):
    rng = np.random.default_rng(seed); out = []
    for rec in C.DS2:
        r = wfdb.rdrecord(f'{C.D}mitdb/{rec}'); a = wfdb.rdann(f'{C.D}mitdb/{rec}', 'atr'); ch = r.sig_name.index('MLII'); adc = r.d_signal[:, ch] if r.d_signal is not None else None
        x = r.p_signal[:, ch]; pos, cls, sym = C.beats(np.asarray(a.sample), np.asarray(a.symbol)); n = len(x)
        cnt = np.bincount(cls[cls < 4], minlength=4); rare = [c for c in (3, 1, 2) if cnt[c] > 0]; target = min(rare, key=lambda c: cnt[c]) if rare else 0
        starts = np.arange(PADS, n - SEG - PADS, 360 * 5); ok = [s for s in starts if np.any((cls == target) & (pos >= s) & (pos < s + SEG))] if target else list(starts)
        s0 = int(rng.choice(ok)); lo, hi = s0 - PADS, s0 + SEG + PADS; seg = x[lo:hi]; m = (pos >= lo) & (pos < hi); p = pos[m] - lo
        res = net.classify(seg, 360, p); show = (p >= PADS) & (p < PADS + SEG)
        rk = RH[(RH_REC == rec)]; rp = RH_POS[RH_REC == rec]; lut = dict(zip(rp.tolist(), rk.tolist())); rhy = [int(lut.get(int(q), 0)) for q in pos[m]]
        out.append(dict(rec=rec, start_s=s0 / 360, mv=b64(np.round(seg * 200).astype(np.int16)), pos=p.tolist(), cls=cls[m].tolist(), sym=''.join(sym[m]),
                        py_probs=np.round(res['probs'], 6).tolist(), py_qrs=np.round(res['qrs_ms'], 3).tolist(), show=[int(np.nonzero(show)[0][0]), int(np.nonzero(show)[0][-1])],
                        counts=cnt.tolist(), target=int(target), rhy=rhy, py_af=(np.round(res['af_prob'], 5).tolist() if 'af_prob' in res else None)))
        print(rec, 'segment at', round(s0 / 360), 's, target class', 'NSVF'[target], 'beats', int(m.sum()), flush=True)
    return out

if __name__ == '__main__':
    net = BeatNet(_P.WORK + '/release/beatnet_weights.npz'); D = dict(weights=weights_pack(_P.WORK + '/release/beatnet_weights.npz'), records=segments(net))
    for k, f in (('exam', 'exam.json'), ('ratchet', 'ratchet.json'), ('physics', 'physics_check.json')):
        try: D[k] = json.load(open(_P.WORK + '/' + f))
        except FileNotFoundError: pass
    try: D.update(json.load(open(_P.WORK + '/fails.json')))
    except FileNotFoundError: pass
    if 'exam' in D:
        for s in ('ds2', 'ds1rest', 'svdb', 'dev'): D['exam'][s].pop('reliability', None)
    if 'ratchet' in D: D['ratchet'] = [{k: r[k] for k in ('ckpt', 'note', 'objective', 'accepted', 'why', 'dev', 'exam') if k in r} for r in D['ratchet']]
    json.dump(D, open(_P.LAB + '/data.json', 'w')); print('data.json', len(json.dumps(D)) // 1024, 'KB')
