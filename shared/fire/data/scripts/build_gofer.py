"""GOFER-Combined (Zenodo 14642378, CC BY 4.0): hourly GOES perimeters for 28 CA fires 2019-2021 -> proc/gofer/."""
import os, warnings, shutil
import numpy as np, pandas as pd, pyogrio, shapely
from pyproj import Transformer
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
warnings.filterwarnings('ignore')
X = f'{_DATA}/data/raw/gofer/x/GOFER'
OUT = f'{_DATA}/data/proc/gofer'; os.makedirs(OUT, exist_ok=True)
TR = Transformer.from_crs(4326, 5070, always_xy=True)
f = lambda g: shapely.to_wkb(shapely.simplify(shapely.transform(g, lambda xy: np.column_stack(TR.transform(xy[:, 0], xy[:, 1]))), 15.0))
for lay in ['GOFERC_fireProg', 'GOFERC_fireIg']:
    m, t = pyogrio.raw.read_arrow(f'{X}/GOFER_Combined/{lay}.shp')
    d = t.to_pandas(); g = shapely.from_wkb(d.pop(m['geometry_name'] or 'wkb_geometry'))
    d['wkb'] = [f(x) if x is not None else None for x in g]
    d['tUTC'] = pd.to_datetime(d['tUTC'])
    d.to_parquet(f'{OUT}/{lay[7:]}.parquet', compression='zstd'); print(lay, len(d))
shutil.copy(f'{X}/fireData.csv', f'{OUT}/fireData.csv')
s = pd.read_csv(f'{X}/GOFER_Combined/GOFERC_summary.csv'); s.to_parquet(f'{OUT}/summary.parquet'); print('summary', len(s))
