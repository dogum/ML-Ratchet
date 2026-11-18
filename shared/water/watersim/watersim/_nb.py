"""numba shim: use numba's njit when installed, otherwise run the same kernels as plain Python (slow but correct)."""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

try:  # pragma: no cover - depends on the environment
    from numba import njit as _njit
    HAVE_NUMBA = True

    def njit(*args, **kw):
        kw.setdefault('cache', True)
        kw.setdefault('fastmath', False)
        if args and callable(args[0]):
            return _njit(**kw)(args[0])
        return _njit(*args, **kw)
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kw):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


def _source_hash():
    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha1()
    for root, _, files in os.walk(here):
        if '__pycache__' in root:
            continue
        for f in sorted(files):
            if f.endswith('.py'):
                with open(os.path.join(root, f), 'rb') as fh:
                    h.update(f.encode() + fh.read())
    return h.hexdigest()


def clear_cache():
    """Delete numba's on-disk caches of this package (numba does not track cross-module changes)."""
    here = os.path.dirname(os.path.abspath(__file__))
    n = 0
    for root, _, files in os.walk(here):
        for f in files:
            if f.endswith('.nbi') or f.endswith('.nbc'):
                try:
                    os.remove(os.path.join(root, f))
                    n += 1
                except OSError:
                    pass
    return n


def check_cache():
    """Clear the numba cache when any source file of the package changed since it was written."""
    here = os.path.dirname(os.path.abspath(__file__))
    stamp = os.path.join(here, '__pycache__', 'watersim_src.sha1')
    try:
        cur = _source_hash()
        old = open(stamp).read().strip() if os.path.exists(stamp) else ''
        if cur != old:
            clear_cache()
            os.makedirs(os.path.dirname(stamp), exist_ok=True)
            with open(stamp, 'w') as fh:
                fh.write(cur)
    except OSError:
        pass
