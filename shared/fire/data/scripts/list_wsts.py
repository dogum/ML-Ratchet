"""List members of the remote WildfireSpreadTS zip (Zenodo 8006177) via HTTP range reads. Data only."""
import sys, zipfile, json, os
sys.path.insert(0, os.path.dirname(__file__))
from httpzip import HttpFile
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
URL = 'https://zenodo.org/api/records/8006177/files/WildfireSpreadTS.zip/content'
out = f'{_DATA}/data/raw/wsts/zip_index.json'
os.makedirs(os.path.dirname(out), exist_ok=True)
f = HttpFile(URL)
z = zipfile.ZipFile(io.BufferedReader(f, 1 << 20) if False else f)
rows = [dict(name=i.filename, csize=i.compress_size, size=i.file_size, off=i.header_offset, ctype=i.compress_type) for i in z.infolist()]
json.dump(dict(url=f.url, size=f.size, members=rows), open(out, 'w'))
print(len(rows), f.size)
