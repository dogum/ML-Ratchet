import fsspec, pyarrow.parquet as pq, sys
fs = fsspec.filesystem('http')
u = 'https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/stable/core_epacems__hourly_emissions.parquet'
f = fs.open(u, block_size=2**20)
m = pq.ParquetFile(f)
md = m.metadata
print(md.num_rows, md.num_row_groups); print(m.schema_arrow)
for i in list(range(0, md.num_row_groups, max(1, md.num_row_groups//15)))[:20]:
    rg = md.row_group(i)
    st = {}
    for j in range(rg.num_columns):
        c = rg.column(j)
        if c.path_in_schema in ('year','state','operating_datetime_utc','plant_id_eia') and c.statistics is not None:
            st[c.path_in_schema] = (c.statistics.min, c.statistics.max)
    print(i, rg.num_rows, rg.total_byte_size, st)
