"""Extract ResStock 2025 R1 / ComStock 2025 R3 baseline (upgrade 0) metadata + annual results, whole stock.

Reads the national parquet over HTTPS with column projection (batches of columns, strings dictionary-encoded),
writes data/proc/stock/<name>_meta.parquet. Resumable: per-batch parts in data/tmp/stock_parts/.
Usage: python3 -I stock_meta.py resstock|comstock
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import sys, os, re, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
import s3http as s

OUT = f'{BLDG_ROOT}/data/proc/stock'
TMP = f'{BLDG_ROOT}/data/tmp/stock_parts'
SRC = {
    'resstock': s.BS + '/resstock_amy2018_release_1/metadata_and_annual_results/national/full/parquet/upgrade0.parquet',
    'comstock': s.BS + '/comstock_amy2018_release_3/metadata_and_annual_results_aggregates/national/full/parquet/upgrade0_agg.parquet',
}
DROP = re.compile(r'savings|intensity|emissions|^applicability|^in\.simulation_control|_bill_(max|min|median_high|median_low)\.\.')


def keep(c):
    if c in ('bldg_id', 'weight', 'completed_status', 'calc.weighted.sqft..ft2', 'dataset', 'upgrade'):
        return True
    if not (c.startswith('in.') or c.startswith('out.')):
        return False
    return not DROP.search(c)


def main(name):
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    f = s.fs().open(s.url(SRC[name]))
    pf = pq.ParquetFile(f)
    cols = [c for c in pf.schema_arrow.names if keep(c)]
    print(name, pf.metadata.num_rows, 'rows', len(cols), 'of', pf.metadata.num_columns, 'columns kept', flush=True)
    B = 30
    parts = []
    for i in range(0, len(cols), B):
        part = f'{TMP}/{name}_{i:04d}.parquet'
        parts.append(part)
        if os.path.exists(part):
            continue
        t = pf.read(columns=cols[i:i + B])
        arrs = []
        for c in t.column_names:
            a = t[c].combine_chunks()
            if pa.types.is_string(a.type) or pa.types.is_large_string(a.type):
                a = pc.dictionary_encode(a)
            elif pa.types.is_float64(a.type) and c.startswith('out.'):
                a = a.cast(pa.float32())
            arrs.append(a)
        pq.write_table(pa.table(arrs, names=t.column_names), part + '.tmp', compression='zstd')
        os.replace(part + '.tmp', part)
        del t, arrs; gc.collect()
        print('batch', i, flush=True)
    tabs = [pq.read_table(p) for p in parts]
    t = tabs[0]
    for x in tabs[1:]:
        for c in x.column_names:
            t = t.append_column(c, x[c])
    pq.write_table(t, f'{OUT}/{name}_meta.parquet', compression='zstd', row_group_size=100_000)
    print('wrote', f'{OUT}/{name}_meta.parquet', t.num_rows, t.num_columns, os.path.getsize(f'{OUT}/{name}_meta.parquet'), flush=True)
    for p in parts:
        os.remove(p)


if __name__ == '__main__':
    main(sys.argv[1])
