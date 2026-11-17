"""Download OEDI AMY2018 county weather for a list of (state, county gisjoin) pairs into refdata/stock_weather.
ResStock 2025 R1 key: weather/state={ST}/{county}_2018.csv (documented in the OEDI AGENT_GUIDE / README)."""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import os
import subprocess
import sys

BASE = ('https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/'
        '2025/resstock_amy2018_release_1/weather/state={st}/{c}_2018.csv')
OUT = f'{BLDG_ROOT}/buildingsim/refdata/stock_weather'


def fetch(pairs):
    os.makedirs(OUT, exist_ok=True)
    got = []
    for st, c in pairs:
        f = os.path.join(OUT, f'{c}_2018.csv')
        if not os.path.exists(f) or os.path.getsize(f) < 100000:
            r = subprocess.run(['curl', '-sS', '-f', '-m', '120', '-o', f, BASE.format(st=st, c=c)])
            if r.returncode != 0:
                print('fail', st, c, file=sys.stderr)
                if os.path.exists(f):
                    os.remove(f)
                continue
        got.append(f)
    return got
