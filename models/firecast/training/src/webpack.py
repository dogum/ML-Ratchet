"""Pack numpy weights for the page (float16 + manifest)."""
import base64, json, sys
import numpy as np


def pack(npz, meta=None):
    z = np.load(npz)
    man, parts, off = [], [], 0
    for k in sorted(z.files):
        a = z[k].astype(np.float32)
        if a.ndim == 0:
            a = a.reshape(1)
        man.append(dict(k=k, shape=list(a.shape), off=off))
        parts.append(a.ravel().astype(np.float16))
        off += a.size
    b = np.concatenate(parts).astype('<f2').tobytes()
    return dict(b64=base64.b64encode(b).decode(), man=man, meta=meta or {})


def f16_npz(npz, out):
    z = np.load(npz)
    np.savez(out, **{k: z[k].astype(np.float16).astype(np.float32) for k in z.files})
