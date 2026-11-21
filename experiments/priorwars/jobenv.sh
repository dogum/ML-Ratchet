# sourced by runner.sh before every job (edit freely; picked up at the next job)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MALLOC_ARENA_MAX=2
export ARROW_DEFAULT_MEMORY_POOL=system     # pyarrow's jemalloc reserves ~1 GB of address space it never uses
export TABPFN_DISABLE_TELEMETRY=1
CAP=2147483648                               # 2 GiB address-space cap (Amendment A2)
MEMWAIT=5200000000                           # wait while usage_in_bytes (RSS + ~1.4 GB reclaimable page cache) is above this, i.e. others hold > ~3.8 GB RSS
# large blocks via mmap so they are unmapped on free (keeps the address space from creeping up across fits)
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576
