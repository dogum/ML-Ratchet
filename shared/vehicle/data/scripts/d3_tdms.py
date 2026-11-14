"""Minimal NI TDMS reader (numpy only) for the D3 "Analysis.tdms" files.

npTDMS is not a dependency of this repo, so this reads the subset of TDMS 2.0 that the D3 files use: metadata
segments, new/appended object lists, contiguous or interleaved raw data, numeric, boolean and timestamp channels
(string channels are skipped). Format: NI "TDMS File Format Internal Structure" (ni.com white paper).

    groups = read_tdms(path)        # {group: {channel: np.ndarray}}, plus props via read_tdms(path, props=True)
"""
import struct
import numpy as np

_DT = {1: 'i1', 2: '<i2', 3: '<i4', 4: '<i8', 5: 'u1', 6: '<u2', 7: '<u4', 8: '<u8', 9: '<f4', 10: '<f8',
       0x19: '<f4', 0x1A: '<f8', 0x21: 'u1'}
_SIZE = {1: 1, 2: 2, 3: 4, 4: 8, 5: 1, 6: 2, 7: 4, 8: 8, 9: 4, 10: 8, 0x19: 4, 0x1A: 8, 0x21: 1, 0x44: 16,
         0x08000C: 8, 0x10000D: 16}
K_META, K_RAW, K_NEWOBJ, K_INTERLEAVED, K_BIGEND, K_DAQMX = 1 << 1, 1 << 3, 1 << 2, 1 << 5, 1 << 6, 1 << 7


def _str(b, o):
    n = struct.unpack_from('<I', b, o)[0]
    return b[o + 4:o + 4 + n].decode('utf-8', 'replace'), o + 4 + n


def _prop_value(b, o, dt):
    if dt == 0x20:
        return _str(b, o)
    if dt == 0x44:
        frac, sec = struct.unpack_from('<Qq', b, o)
        return sec + frac / 2 ** 64, o + 16
    if dt in _DT:
        v = np.frombuffer(b, _DT[dt], 1, o)[0]
        return v.item(), o + _SIZE[dt]
    if dt in _SIZE:
        return None, o + _SIZE[dt]
    raise ValueError(f'TDMS property type {dt:#x} not supported')


def _split_path(p):
    parts = [s.strip("'").replace("''", "'") for s in p.split("'/'")]
    parts = [s.lstrip('/').strip("'") for s in parts]
    return tuple(x for x in parts if x != '')


def read_tdms(path, props=False):
    b = open(path, 'rb').read()
    N = len(b)
    o = 0
    order = []                      # active object list: [path, ...] with raw data
    index = {}                      # path -> (dtype code, n values per chunk, total bytes (strings))
    pieces = {}                     # path -> list of arrays
    properties = {}
    while o + 28 <= N:
        if b[o:o + 4] != b'TDSm':
            break
        toc, ver, nxt, rawoff = struct.unpack_from('<IIQQ', b, o + 4)
        if toc & K_BIGEND:
            raise ValueError('big-endian TDMS not supported')
        seg_start = o + 28
        seg_end = N if nxt == 0xFFFFFFFFFFFFFFFF else min(N, seg_start + nxt)
        if toc & K_META:
            m = seg_start
            nobj = struct.unpack_from('<I', b, m)[0]; m += 4
            if toc & K_NEWOBJ:
                order = []
            for _ in range(nobj):
                p, m = _str(b, m)
                ri = struct.unpack_from('<I', b, m)[0]; m += 4
                if ri == 0xFFFFFFFF:
                    pass
                elif ri == 0:
                    if p not in order and p in index:
                        order.append(p)
                elif ri in (0x69120000, 0x69130000):
                    raise ValueError('DAQmx raw data not supported')
                else:
                    dt, dim, nv = struct.unpack_from('<IIQ', b, m)
                    tot = struct.unpack_from('<Q', b, m + 16)[0] if dt == 0x20 else None
                    m += ri - 4                # the length counts its own 4 bytes
                    index[p] = (dt, nv, tot)
                    if p not in order:
                        order.append(p)
                npr = struct.unpack_from('<I', b, m)[0]; m += 4
                for _ in range(npr):
                    k, m = _str(b, m)
                    dt = struct.unpack_from('<I', b, m)[0]; m += 4
                    v, m = _prop_value(b, m, dt)
                    properties.setdefault(p, {})[k] = v
        if toc & K_RAW and order:
            r0 = seg_start + rawoff
            chans = [(p, index[p]) for p in order]
            chunk = 0
            for p, (dt, nv, tot) in chans:
                chunk += tot if dt == 0x20 else nv * _SIZE.get(dt, 0)
            nchunks = (seg_end - r0) // chunk if chunk else 0
            if toc & K_INTERLEAVED:
                nv = chans[0][1][1] * nchunks
                if any(c[1][0] not in _DT for c in chans):
                    raise ValueError('interleaved non-numeric TDMS data not supported')
                rec = np.dtype([(f'c{i}', _DT[c[1][0]]) for i, c in enumerate(chans)])
                arr = np.frombuffer(b, rec, nv, r0)
                for i, (p, _) in enumerate(chans):
                    pieces.setdefault(p, []).append(np.array(arr[f'c{i}']))
            else:
                q = r0
                for _ in range(nchunks):
                    for p, (dt, nv, tot) in chans:
                        if dt == 0x20:
                            q += tot; continue
                        sz = _SIZE.get(dt, 0)
                        if dt in _DT:
                            pieces.setdefault(p, []).append(np.frombuffer(b, _DT[dt], nv, q))
                        elif dt == 0x44:
                            raw = np.frombuffer(b, '<u8', 2 * nv, q).reshape(-1, 2)
                            pieces.setdefault(p, []).append(raw[:, 1].view('<i8') + raw[:, 0] / 2 ** 64)
                        q += nv * sz
        o = seg_end
    out = {}
    for p, ps in pieces.items():
        sp = _split_path(p)
        if len(sp) == 2:
            out.setdefault(sp[0], {})[sp[1]] = np.concatenate(ps)
    if props:
        return out, {_split_path(k): v for k, v in properties.items()}
    return out
