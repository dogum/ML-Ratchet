"""Dated supplementary exam (27 Nov 2026; audit queue): the released v1.1 network re-scored with NDBC direction files
parsed with field sentinels (99 kept in swdir and swr1, as pipeline.read_ndbc(field_sentinels=True)).

fetch.py deleted the raw files after gridding, so for every buoy-year an exam or dev window touches, the swdir and swr1
files are downloaded again. Each is parsed twice through fetch.parse: the old way (keep_99=False), which must reproduce
the stored data/buoy grid (a check that the files did not change), and with keep_99=True, written to data/buoy_fs/.
Energy (swden) is unchanged and copied. Events, labels and splits are the frozen data/events.json; only the network's
input windows change. Writes work/exam_v111_fs.json. Nothing here feeds any choice."""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as _P
sys.path.insert(0, _P.MODEL)
import fetch, realdata as RD, events as EV, exam_v11 as X, swellcast as SC
FS = os.path.join(_P.WORK, 'data/buoy_fs')

def dir_grids(fd, fj, h0, n, keep):
    C = np.full((n, len(fetch.FGRID)), np.nan, np.float32); S = C.copy()
    pd_, pj = fetch.parse(fd, 0, keep), fetch.parse(fj, 0, keep)
    if pd_ is None or pj is None: return C, S
    td, frd, D = pd_; tj, frj, R = pj
    if np.nanmax(R) > 1.5: R = R / 100.0
    R = np.clip(R, 0, 1); common, idd, idj = np.intersect1d(td, tj, return_indices=True)
    a = np.radians(D[idd]); r = R[idj]; kk = (common >= h0) & (common < h0 + n)
    C[common[kk] - h0] = fetch.to_grid(frd, (r * np.cos(a))[kk], False); S[common[kk] - h0] = fetch.to_grid(frd, (r * np.sin(a))[kk], False)
    return C, S

def rebuild(b, y):
    out = os.path.join(FS, f'{b}_{y}.npz')
    if os.path.exists(out): return json.load(open(out + '.json'))
    z = np.load(os.path.join(RD.BUOY, f'{b}_{y}.npz')); h0 = int(z['h0']); E = z['E']; n = len(E)
    fd, fj = fetch.get(b, 'd', y), fetch.get(b, 'j', y); info = dict(buoy=b, year=y)
    if not (fd and fj):
        info['status'] = 'no direction files'; C, S = z['C'].astype(np.float32), z['S'].astype(np.float32)
    else:
        C0, S0 = dir_grids(fd, fj, h0, n, False); C1, S1 = dir_grids(fd, fj, h0, n, True)
        old = z['C'].astype(np.float32); same = np.allclose(np.nan_to_num(C0.astype(np.float16).astype(np.float32), nan=9), np.nan_to_num(old, nan=9), atol=1e-3)
        info.update(status='ok', old_parse_reproduces_store=bool(same), finite_before=int(np.isfinite(C0).sum()), finite_after=int(np.isfinite(C1).sum()))
        C, S = C1, S1
    os.makedirs(FS, exist_ok=True); np.savez_compressed(out, E=E, C=C.astype(np.float16), S=S.astype(np.float16), h0=h0, f=z['f'])
    json.dump(info, open(out + '.json', 'w')); return info

def needed():
    s = set()
    for sp in ('exam', 'dev'):
        for e in EV.load(sp, None):
            tc = int(np.floor(e['tc'] / 3) * 3)
            for h in (tc - 120, tc + 119):
                s.add((e['buoy'], int(np.datetime64(int(h), 'h').astype('datetime64[Y]').astype(int) + 1970)))
    return sorted(s)

def score_all():
    net = SC.SwellCast(); out = {}
    for split, vis in (('exam', 'confirmed'), ('exam', True), ('exam', None), ('dev', 'confirmed')):
        ev = EV.load(split, vis); Xw = EV.windows(ev); out[f'{split}_{vis}'] = X.score(net.predict(Xw), ev)
    return out

if __name__ == '__main__':
    t = time.time(); infos = []
    for b, y in needed():
        if not os.path.exists(os.path.join(RD.BUOY, f'{b}_{y}.npz')): continue
        infos.append(rebuild(b, y)); print(infos[-1], flush=True)
    res = dict(run_utc=time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()), buoy_years=len(infos),
               old_parse_reproduces_store=sum(i.get('old_parse_reproduces_store', False) for i in infos),
               no_direction_files=sum(i['status'] != 'ok' for i in infos),
               direction_cells_before=sum(i.get('finite_before', 0) for i in infos), direction_cells_after=sum(i.get('finite_after', 0) for i in infos))
    res['generic_99'] = score_all(); RD.BUOY = FS; RD._cache.clear(); res['field_sentinels'] = score_all()
    json.dump(res, open(os.path.join(_P.WORK, 'exam_v111_fs.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('generic_99', 'field_sentinels')}))
    for k in res['generic_99']: print(k, 'old', {a: round(b, 3) for a, b in res['generic_99'][k].items()}, '\n   new', {a: round(b, 3) for a, b in res['field_sentinels'][k].items()})
    print('%.0f s' % (time.time() - t))
