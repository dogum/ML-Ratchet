"""Fetch single members of the remote WSTS zip by HTTP range (local header + deflate stream). Data only."""
import json, zlib, struct, requests, time
import os
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
IDX = f'{_DATA}/data/raw/wsts/zip_index.json'
_idx = None
def index():
    global _idx
    if _idx is None:
        d = json.load(open(IDX)); _idx = (d['url'], {m['name']: m for m in d['members']})
    return _idx
S = requests.Session()
def fetch(name, retries=6):
    url, mem = index(); m = mem[name]
    start = m['off']; end = start + 30 + 1024 + m['csize']  # local header + generous name/extra
    for i in range(retries):
        try:
            r = S.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=(30, 300)); r.raise_for_status()
            b = r.content
            assert b[:4] == b'PK\x03\x04'
            nl, el = struct.unpack('<HH', b[26:30])
            data = b[30 + nl + el: 30 + nl + el + m['csize']]
            assert len(data) == m['csize']
            out = zlib.decompress(data, -15) if m['ctype'] == 8 else data
            assert len(out) == m['size']
            return out
        except Exception as e:
            if i == retries - 1: raise
            time.sleep(5 * (i + 1))
