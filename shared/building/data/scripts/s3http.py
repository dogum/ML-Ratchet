"""Small helpers for reading public OEDI S3 objects over HTTPS (through the proxy) with column projection."""
import re, time, urllib.parse, requests, fsspec
S3 = 'https://oedi-data-lake.s3.amazonaws.com'
BS = 'nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2025'
_fs = None
def fs():
    global _fs
    if _fs is None:
        _fs = fsspec.filesystem('https', client_kwargs={'trust_env': True}, block_size=4 * 2**20)
    return _fs
def url(key): return f'{S3}/{urllib.parse.quote(key)}'
def list_keys(prefix, delimiter=None, max_keys=None):
    """All keys (and sizes) under prefix, following continuation tokens (ListObjectsV2)."""
    out, pref, tok = [], [], None
    while True:
        p = {'list-type': '2', 'prefix': prefix}
        if delimiter: p['delimiter'] = delimiter
        if tok: p['continuation-token'] = tok
        for a in range(5):
            try:
                r = requests.get(S3 + '/', params=p, timeout=60); r.raise_for_status(); break
            except Exception:
                time.sleep(2 + 3 * a)
        t = r.text
        out += [(k, int(s)) for k, s in re.findall(r'<Key>(.*?)</Key>.*?<Size>(.*?)</Size>', t)]
        pref += re.findall(r'<Prefix>(.*?)</Prefix>', t)[1:]
        m = re.search(r'<NextContinuationToken>(.*?)</NextContinuationToken>', t)
        if not m or (max_keys and len(out) >= max_keys): break
        tok = m.group(1)
    return (out, pref) if delimiter else out
