"""Globe-LFMC 2.0 (Yebra et al. 2024, Sci. Data; figshare 10.6084/m9.figshare.25413790, CC0) -> proc/lfmc/lfmc_us.parquet
(CONUS samples only). Streams the xlsx in read-only mode."""
import os, openpyxl, pandas as pd, numpy as np
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
os.makedirs(f'{_DATA}/data/proc/lfmc', exist_ok=True)
wb = openpyxl.load_workbook(f'{_DATA}/data/raw/lfmc/Globe-LFMC-2.0_final.xlsx', read_only=True)
print(wb.sheetnames)
for ws in wb.worksheets:
    it = ws.iter_rows(values_only=True)
    hdr = next(it); print(ws.title, hdr)
    if 'LFMC' not in str(hdr): continue
    rows = [r for r in it if r and any(r)]
    d = pd.DataFrame(rows, columns=[str(h) for h in hdr])
    print(len(d)); print(d.head(3).T)
    for c in d.columns:
        if d[c].dtype == object:
            try: d[c] = pd.to_numeric(d[c])
            except Exception: d[c] = d[c].astype('string')
    d.to_parquet(f'{_DATA}/data/proc/lfmc/lfmc_all.parquet')
