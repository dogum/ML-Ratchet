"""RoadData rebuild: download the raw sources, then run data/scripts/ in order (HANDOFF.md P0.3).

usage: python -I shared/vehicle/data/build.py [--only a,b] [--from stage] [--jobs N] [--offline] [--list]
       (or `mlratchet data vehicle ...`)

Resumable: a download that is already in raw/ is not fetched again (partial files resume from .part); each stage skips
the work its outputs show is done. Safe for tracked files: the stages that would write tracked tables (EPA, fueleconomy,
cycles, HD, splits, baselines, QC) run in a shadow root under data/work/rebuild/, and their outputs are compared with
the shipped tables. data/proc/splits.json is never written.

Stages (in order):
  fetch      raw downloads: VED, USGS 3DEP tiles, IEM ASOS weather, EPA test cars, fueleconomy.gov, EPA schedules,
             eCFR 1037 App. A/D, FASTSim 2.1.5 cycles, ZTBus (Zenodo mirror), Mersin (Zenodo)
  stage1     VED .7z -> weekly CSVs -> proc/ved_stage1/ (parallel over CSV shards)
  survey     proc/ved_signal_availability.csv, proc/ved_oat_sample.parquet
  splits_ved make_splits_ved.py must reproduce the frozen VED split; proc/splits_ved_part.json = frozen VED part
  dem        proc/dem_annarbor_2m.npz (2 m mosaic of the four 1 m lidar tiles)
  stage2     proc/ved/trips/ (16 vehicle buckets in parallel; per-trip resumable)
  index      proc/ved/index.parquet
  tables     shadow: epa_testcars, fe_labels, cycles, hd, make_splits_all, baselines, qc
  refdata    roadsim/refdata/ from the same raw files, then roadsim/tools/epa_candidates.py; shadow build_cycles and
             build_generator_calib are compared with roadsim/roadsim/data/
  verify     splits.json sha256, rebuilt-vs-shipped comparison, a dev-trip load
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

VEH = Path(os.environ.get('ML_RATCHET_VEHICLE') or Path(__file__).resolve().parents[1])
DATA = VEH / 'data'
RAW = DATA / 'raw'
PROC = DATA / 'proc'
SCRIPTS = DATA / 'scripts'
WORK = DATA / 'work'                 # git-ignored (the repo ignores every work/ folder)
SHADOW = WORK / 'rebuild' / 'vehicle'
RS_SHADOW = WORK / 'rebuild' / 'roadsim'
LOGS = WORK / 'logs'
REFDATA = VEH / 'roadsim' / 'refdata'
STATE = WORK / 'build_state.json'
SPLITS_SHA = 'd917df78b099167ceadd7c5cd384a9fd9c45f2e14ca69fa3927c98b9b3233ad9'
UA = 'mlratchet-data-rebuild/1.0 (+https://github.com/; research data rebuild)'
PY = sys.executable
STAGES = ['fetch', 'stage1', 'survey', 'splits_ved', 'dem', 'stage2', 'index', 'tables', 'refdata', 'verify']

# ---------------------------------------------------------------- sources
VED = 'https://raw.githubusercontent.com/gsoh/VED/master/Data/'
VED_PARTS = ['VED_DynamicData_Part1.7z', 'VED_DynamicData_Part2.7z']
VED_STATIC = {'VED_Static_Data_ICE&HEV.xlsx': 'VED_Static_Data_ICE%26HEV.xlsx',
              'VED_Static_Data_PHEV&EV.xlsx': 'VED_Static_Data_PHEV%26EV.xlsx'}
DEM_TILES = ['x26y468', 'x26y469', 'x27y468', 'x27y469']           # as dem_mosaic.py
DEM_URL = ('https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/MI_31Co_Washtenaw_2016/TIFF/'
           'USGS_one_meter_{t}_MI_31Co_Washtenaw_2016.tif')
# Routine + special METARs (report_type 3 and 4) reproduce the shipped trip T_amb exactly (300 dev trips; routine only
# is off by up to 1.3 degC). The window 2017-10-31..2018-11-13 reproduces checks/qc_summary.json's 8,830 joint hours.
ASOS = ('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={s}&data=tmpc&data=dwpc&data=alti&data=sknt'
        '&data=drct&data=wxcodes&data=p01i&year1=2017&month1=10&day1=31&year2=2018&month2=11&day2=13&tz=Etc%2FUTC'
        '&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=3&report_type=4')
EPA_SF = 'https://www.epa.gov/sites/default/files/'
EPA_SY = 'https://www.epa.gov/system/files/documents/'
EPA_TESTCARS = {   # model years 2008-2026, the files proc/epa/testcars_tests.parquet was built from
    **{f'{y:02d}tstcar.csv': f'{EPA_SF}2016-07/{y:02d}tstcar.csv' for y in range(8, 17)},
    '09tstcar-laterelease.csv': f'{EPA_SF}2016-07/09tstcar-laterelease.csv',
    '17tstcar-2018-05-30.xlsx': f'{EPA_SF}2018-06/17tstcar-2018-05-30.xlsx',
    '18tstcar-2018-10-24.xlsx': f'{EPA_SF}2018-10/18tstcar-2018-10-24.xlsx',
    '19tstcar-2020-10-02.xlsx': f'{EPA_SF}2020-10/19tstcar-2020-10-02.xlsx',
    '20tstcar-2021-03-02.xlsx': f'{EPA_SF}2021-03/20tstcar-2021-03-02.xlsx',
    '21-tstcar-2022-04-15.xlsx': f'{EPA_SY}2022-04/21-tstcar-2022-04-15.xlsx',
    '22-testcar-2023-06-13.xlsx': f'{EPA_SY}2023-06/22-testcar-2023-06-13.xlsx',
    '23-testcar-2024-05-17_0.xlsx': f'{EPA_SY}2024-05/23-testcar-2024-05-17_0.xlsx',
    '24-testcar-2025-05.xlsx': f'{EPA_SY}2025-05/24-testcar-2025-05.xlsx',
    '25-testcar-2026-01-21.xlsx': f'{EPA_SY}2026-01/25-testcar-2026-01-21.xlsx',
    '26-testcar-2026-01-21.xlsx': f'{EPA_SY}2026-01/26-testcar-2026-01-21.xlsx'}
FE_ZIP = 'https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip'
EPA_DDS = {   # epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules
    **{f: f'{EPA_SF}2015-10/{f}' for f in ['uddscol.txt', 'ftpcol.txt', 'hwycol.txt', 'sc03col.txt', 'la92dds.txt',
                                           'la92shortdds.txt', 'im240col.txt', 'ececol.txt', 'eudccol.txt', 'eudclcol.txt',
                                           'j1015col.txt', 'jpn10col.txt', 'jpn15col.txt']},
    'ftpmc1b.txt': f'{EPA_SF}2021-01/ftpmc1b.txt', 'la92col.txt': f'{EPA_SF}2021-01/la92col.txt',
    'us06col.txt': 'https://www.epa.gov/system/files/other-files/2025-03/us06col.txt',
    'epa-new-york-city-cycle.txt': 'https://www.epa.gov/system/files/other-files/2025-03/epa-new-york-city-cycle.txt'}
ECFR_TITLES = 'https://www.ecfr.gov/api/versioner/v1/titles.json'
ECFR = 'https://www.ecfr.gov/api/versioner/v1/full/{date}/title-40.xml?part=1037&appendix=Appendix%20{a}%20to%20Part%201037'
FASTSIM_JSON = 'https://pypi.org/pypi/fastsim/2.1.5/json'
FASTSIM_CYCLES = ['wltc_3b.csv', 'longHaulDriveCycle.csv']
ZTBUS = 'https://zenodo.org/api/records/17470816/files/{f}/content'      # mirror of doi:10.3929/ethz-b-000626723
MERSIN = 'https://zenodo.org/api/records/22716221/files/mersin_33M_bus_dataset.xlsx/content'


# ---------------------------------------------------------------- helpers
def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {'timings_s': {}, 'blocked': {}, 'downloads': {}}


def save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


class Blocked(Exception):
    pass


def http_get(url, dest: Path, tries=5, gz=False):
    """Download url to dest (via dest.part, resuming). Raises Blocked on 401/403/429/451 or after `tries` failures."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    last = None
    for k in range(tries):
        have = part.stat().st_size if part.exists() and not gz else 0
        hdr = {'User-Agent': UA}
        if have: hdr['Range'] = f'bytes={have}-'
        if gz: hdr['Accept-Encoding'] = 'gzip'
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=120) as r:
                mode = 'ab' if (have and r.status == 206) else 'wb'
                if gz and r.headers.get('Content-Encoding') == 'gzip':
                    body = gzip.decompress(r.read())
                    part.write_bytes(body)
                else:
                    with open(part, mode) as f:
                        shutil.copyfileobj(r, f, 1 << 20)
            os.replace(part, dest)
            return dest.stat().st_size
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:          # already complete
                os.replace(part, dest); return dest.stat().st_size
            if e.code in (401, 403, 451):
                raise Blocked(f'HTTP {e.code} {e.reason}')
            last = f'HTTP {e.code} {e.reason}'
            if e.code == 429:           # rate limited: back off politely, give up after `tries`
                time.sleep(60 * (k + 1)); continue
        except Exception as e:   # timeouts, resets
            last = repr(e)
        time.sleep(5 * (k + 1))
    raise Blocked(f'failed after {tries} tries: {last}')


def run_py(args, log_name, env_extra=None, cwd=None):
    """Run a Python script (isolated mode); stdout+stderr to work/logs/<log_name>.log. Returns the return code."""
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
    env.pop('ML_RATCHET_VEHICLE', None)
    env.update(env_extra or {})
    with open(LOGS / f'{log_name}.log', 'a', encoding='utf-8') as f:
        f.write(f'\n==== {time.ctime()} {args}\n'); f.flush()
        return subprocess.run([PY, '-I', '-X', 'utf8', *map(str, args)], stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd).returncode


def must(rc, what):
    if rc != 0:
        raise SystemExit(f'{what} failed (rc {rc}); see {LOGS}')


def link_dir(target: Path, link: Path):
    """Directory junction (Windows) or symlink pointing at target."""
    if link.exists() or link.is_symlink():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def unlink_dir(link: Path):
    """Remove a junction/symlink without touching its target."""
    if os.name == 'nt' and link.exists():
        os.rmdir(link)              # removes the junction itself only
    elif link.is_symlink():
        link.unlink()


def sha256(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- stage: fetch
def fetch_jobs(st):
    """(name, dest, url-or-callable, gz) for every raw file still missing."""
    jobs = []
    for p in VED_PARTS:
        jobs.append(('ved', RAW / 'ved' / p, VED + p, False))
    for local, remote in VED_STATIC.items():
        jobs.append(('ved', RAW / 'ved' / local, VED + remote, False))
    dem_done = (PROC / 'dem_annarbor_2m.npz').exists()
    for t in DEM_TILES:
        if not dem_done and not (PROC / 'dem_tmp' / f'{t}.npz').exists():
            jobs.append(('dem', RAW / 'elev' / f'{t}.tif', DEM_URL.format(t=t), False))
    for s in ['ARB', 'YIP', 'DTW']:
        jobs.append(('weather', RAW / 'weather' / f'asos_{s}.csv', ASOS.format(s=s), False))
    for f, u in EPA_TESTCARS.items():
        jobs.append(('epa', RAW / 'epa' / f, u, False))
    jobs.append(('fe', RAW / 'fe' / 'vehicles.csv.zip', FE_ZIP, False))
    for f, u in EPA_DDS.items():
        jobs.append(('cycles', RAW / 'cycles' / f, u, False))
    for a in 'AD':
        jobs.append(('ecfr', RAW / 'cycles' / f'cfr1037_app{a}.xml', ('ecfr', a), True))
    jobs.append(('fastsim', RAW / 'cycles' / 'fastsim-2.1.5.whl', ('fastsim',), False))
    jobs.append(('ztbus', RAW / 'hd' / 'ztbus' / 'metaData.csv', ZTBUS.format(f='metaData.csv'), False))
    jobs.append(('ztbus', RAW / 'hd' / 'ztbus' / 'ZTBus_samples.zip', ZTBUS.format(f='ZTBus_samples.zip'), False))
    jobs.append(('mersin', RAW / 'hd' / 'mersin' / 'mersin_33M_bus_dataset.xlsx', MERSIN, False))
    return [j for j in jobs if not j[1].exists()]


def resolve(url):
    if isinstance(url, str):
        return url
    if url[0] == 'ecfr':
        with urllib.request.urlopen(urllib.request.Request(ECFR_TITLES, headers={'User-Agent': UA}), timeout=60) as r:
            t = next(x for x in json.load(r)['titles'] if x['number'] == 40)
        return ECFR.format(date=t['latest_issue_date'], a=url[1])
    if url[0] == 'fastsim':
        with urllib.request.urlopen(urllib.request.Request(FASTSIM_JSON, headers={'User-Agent': UA}), timeout=60) as r:
            files = json.load(r)['urls']
        whl = [f for f in files if f['filename'].endswith('.whl')]
        return min(whl, key=lambda f: f['size'])['url']       # any wheel carries the same resources
    raise ValueError(url)


def unpack():
    """Derived raw files: vehicles.csv, FASTSim cycle CSVs, ZTBus sample missions."""
    z = RAW / 'fe' / 'vehicles.csv.zip'
    if z.exists() and not (RAW / 'fe' / 'vehicles.csv').exists():
        with zipfile.ZipFile(z) as f:
            name = next(n for n in f.namelist() if n.lower().endswith('vehicles.csv'))
            (RAW / 'fe' / 'vehicles.csv').write_bytes(f.read(name))
    w = RAW / 'cycles' / 'fastsim-2.1.5.whl'
    out = RAW / 'cycles' / 'fastsim2_cycles'
    if w.exists() and not all((out / c).exists() for c in FASTSIM_CYCLES):
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(w) as f:
            for c in FASTSIM_CYCLES:
                name = next(n for n in f.namelist() if n.endswith(f'resources/cycles/{c}'))
                (out / c).write_bytes(f.read(name))
    z = RAW / 'hd' / 'ztbus' / 'ZTBus_samples.zip'
    out = RAW / 'hd' / 'ztbus' / 'samples'
    if z.exists() and not (out / '.ok').exists():
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as f:
            for n in f.namelist():
                b = os.path.basename(n)
                if b.endswith('.csv') and b != 'metaData.csv' and not b.startswith('.'):
                    (out / b).write_bytes(f.read(n))
        (out / '.ok').write_text('')


def stage_fetch(a, st):
    jobs = fetch_jobs(st)
    log(f'fetch: {len(jobs)} files to download')

    def one(j):
        src, dest, url, gz = j
        try:
            u = resolve(url)
            t = time.time(); n = http_get(u, dest, gz=gz)
            if src == 'dem':
                Path(str(dest) + '.ok').write_text('')
            log(f'  {src:8s} {dest.name}  {n / 1e6:8.1f} MB  {time.time() - t:5.0f} s')
            return src, dest, u, n, None
        except Blocked as e:
            log(f'  BLOCKED {src} {dest.name}: {e}')
            return src, dest, str(url), None, str(e)

    # IEM asks for one request at a time: its jobs run sequentially, the rest six at a time
    iem = [j for j in jobs if 'mesonet.agron.iastate.edu' in str(j[2])]
    rest = [j for j in jobs if j not in iem]

    def seq():
        out = []
        for j in iem:
            out.append(one(j)); time.sleep(2)
        return out

    with cf.ThreadPoolExecutor(6) as ex:
        f_iem = ex.submit(seq)
        for src, dest, u, n, err in list(ex.map(one, rest)) + f_iem.result():
            key = str(dest.relative_to(RAW)).replace('\\', '/')
            if err:
                st['blocked'][key] = {'url': u, 'error': err, 'when': time.strftime('%Y-%m-%d %H:%M')}
            else:
                st['blocked'].pop(key, None)
                st['downloads'][key] = {'url': u, 'bytes': n, 'when': time.strftime('%Y-%m-%d %H:%M')}
    save_state(st)
    unpack()
    if st['blocked']:
        log('fetch: blocked sources (recorded in work/build_state.json):', ', '.join(st['blocked']))


# ---------------------------------------------------------------- VED stages
def _done1():
    f = PROC / 'ved_stage1' / 'stage1_done.txt'
    return set(f.read_text().split()) if f.exists() else set()


def stage_stage1(a, st):
    import py7zr
    done = _done1()
    for k, p in enumerate(VED_PARTS, 1):
        arc = RAW / 'ved' / p
        if not arc.exists():
            raise SystemExit(f'missing {arc} (fetch blocked?)')
        with py7zr.SevenZipFile(arc) as z:
            names = [n for n in z.getnames() if n.lower().endswith('.csv')]
        todo = [n for n in names if Path(n).stem not in done]
        log(f'stage1: {p}: {len(names)} weekly files, {len(todo)} to convert')
        if not todo:
            continue
        xdir = RAW / 'ved' / f'x{k}'
        xdir.mkdir(parents=True, exist_ok=True)
        have = {q.name for q in xdir.rglob('*.csv')}
        need = [n for n in todo if Path(n).name not in have]
        if need:
            with py7zr.SevenZipFile(arc) as z:
                z.extract(path=xdir, targets=need)
        # shard the CSVs so that several stage-1 processes run at once
        csvs = sorted(q for q in xdir.rglob('*.csv'))
        nj = max(1, min(a.jobs, len(csvs)))
        shards = [xdir / f's{j:02d}' for j in range(nj)]
        for i, c in enumerate(csvs):
            d = shards[i % nj]; d.mkdir(exist_ok=True)
            if c.parent != d:
                os.replace(c, d / c.name)
        with cf.ThreadPoolExecutor(nj) as ex:
            rcs = list(ex.map(lambda d: run_py([SCRIPTS / 'ved_stage1.py', d], f'stage1_{d.parent.name}_{d.name}'), shards))
        must(max(rcs), 'ved_stage1.py')
        shutil.rmtree(xdir, ignore_errors=True)     # CSVs were deleted by stage 1; the .7z is kept
    done = _done1()
    log(f'stage1: {len(done)} weekly files converted')


def stage_survey(a, st):
    if (PROC / 'ved_signal_availability.csv').exists() and (PROC / 'ved_oat_sample.parquet').exists():
        log('survey: done'); return
    must(run_py([SCRIPTS / 'ved_survey.py'], 'survey'), 'ved_survey.py')


def stage_splits_ved(a, st):
    frozen = json.loads((PROC / 'splits.json').read_text())['VED']
    part = PROC / 'splits_ved_part.json'
    must(run_py([SCRIPTS / 'make_splits_ved.py'], 'splits_ved'), 'make_splits_ved.py')
    got = json.loads(part.read_text())['VED']
    diff = {k: (frozen['vehicles'].get(k), got['vehicles'].get(k))
            for k in set(frozen['vehicles']) | set(got['vehicles']) if frozen['vehicles'].get(k) != got['vehicles'].get(k)}
    st['splits_ved_reproduced'] = not diff
    st['splits_ved_diff'] = diff
    # stage 2 always uses the frozen VED part
    part.write_text(json.dumps({'VED': frozen}, indent=1))
    save_state(st)
    if diff:
        log(f'splits_ved: WARNING make_splits_ved.py differs from the frozen split on {len(diff)} vehicles: {diff}; '
            'the frozen split is used')
    else:
        log(f'splits_ved: make_splits_ved.py reproduces the frozen VED split ({len(frozen["vehicles"])} vehicles)')


def stage_dem(a, st):
    if (PROC / 'dem_annarbor_2m.npz').exists():
        log('dem: done'); return
    must(run_py([SCRIPTS / 'dem_mosaic.py'], 'dem'), 'dem_mosaic.py')
    shutil.rmtree(PROC / 'dem_tmp', ignore_errors=True)


def stage_stage2(a, st):
    todo = [b for b in range(16) if not (PROC / 'ved' / f'stage2_b{b:02d}.done').exists()]
    log(f'stage2: {len(todo)} of 16 buckets to build, {a.jobs} at a time')
    with cf.ThreadPoolExecutor(min(a.jobs, 16)) as ex:
        futs = {ex.submit(run_py, [SCRIPTS / 'ved_stage2.py', b], f'stage2_b{b:02d}'): b for b in todo}
        bad = []
        for f in cf.as_completed(futs):
            b = futs[f]
            log(f'  bucket {b:02d}: rc {f.result()}')
            if f.result(): bad.append(b)
    if bad:
        raise SystemExit(f'stage2 failed for buckets {bad}; see {LOGS}')


def stage_index(a, st):
    idx = PROC / 'ved' / 'index.parquet'
    marks = list((PROC / 'ved').glob('stage2_b*.done')) + [SCRIPTS / 'ved_index.py']
    if idx.exists() and all(idx.stat().st_mtime > m.stat().st_mtime for m in marks):
        log('index: up to date'); return
    must(run_py([SCRIPTS / 'ved_index.py'], 'index'), 'ved_index.py')


# ---------------------------------------------------------------- shadow tables
def make_shadow():
    """A vehicle root whose proc/ is private (tables are rebuilt there) and whose raw/, proc/ved/ and realdata/ are
    junctions to the real ones."""
    (SHADOW / 'data' / 'proc').mkdir(parents=True, exist_ok=True)
    link_dir(VEH / 'realdata', SHADOW / 'realdata')
    link_dir(RAW, SHADOW / 'data' / 'raw')
    link_dir(PROC / 'ved', SHADOW / 'data' / 'proc' / 'ved')
    for d in ['epa', 'fe', 'cycles', 'hd', 'baselines', 'checks']:
        (SHADOW / 'data' / 'proc' / d).mkdir(exist_ok=True)
    for f in ['splits_ved_part.json', 'ved_signal_availability.csv']:
        shutil.copy2(PROC / f, SHADOW / 'data' / 'proc' / f)
    shutil.copy2(PROC / 'splits.json', SHADOW / 'data' / 'proc' / 'splits.json')


def stage_tables(a, st):
    make_shadow()
    env = {'ML_RATCHET_VEHICLE': str(SHADOW)}
    sp = SHADOW / 'data' / 'proc'
    for s in ['epa_testcars.py', 'fe_labels.py', 'cycles.py', 'hd.py', 'make_splits_all.py']:
        t = time.time()
        must(run_py([SCRIPTS / s], f'tables_{s[:-3]}', env), s)
        log(f'tables: {s} {time.time() - t:.0f} s')
    # the rebuilt split file is only compared, never used: put the frozen tables back before the baselines
    rebuilt = json.loads((sp / 'splits.json').read_text())
    shutil.copy2(sp / 'splits.json', sp / 'splits.rebuilt.json')
    frozen = json.loads((PROC / 'splits.json').read_text())
    st['splits_rebuilt_content_sha'] = rebuilt.get('sha256_of_content_without_this_field')
    st['splits_frozen_content_sha'] = frozen.get('sha256_of_content_without_this_field')
    st['splits_rebuilt_equal'] = rebuilt == frozen
    if rebuilt != frozen:
        for f in ['epa/testcars_split.parquet', 'fe/fe_split.parquet']:
            shutil.copy2(sp / f, sp / (f[:-8] + '.rebuilt.parquet'))
            shutil.copy2(PROC / f, sp / f)
        log('tables: WARNING the rebuilt splits.json differs from the frozen one; frozen split tables used downstream')
    shutil.copy2(PROC / 'splits.json', sp / 'splits.json')
    save_state(st)
    for s in ['baselines.py', 'qc.py']:
        t = time.time()
        must(run_py([SCRIPTS / s], f'tables_{s[:-3]}', env), s)
        log(f'tables: {s} {time.time() - t:.0f} s')


# ---------------------------------------------------------------- roadsim refdata
REF_CYCLES = ['uddscol.txt', 'hwycol.txt', 'us06col.txt', 'sc03col.txt', 'epa-new-york-city-cycle.txt', 'la92col.txt']


def stage_refdata(a, st):
    """roadsim/refdata/ from the RoadData raw files (copies, no second download)."""
    cp = []
    for f in EPA_TESTCARS:
        if f.endswith('.xlsx'): cp.append((RAW / 'epa' / f, REFDATA / 'epa_testcar' / f))
    cp.append((RAW / 'fe' / 'vehicles.csv', REFDATA / 'fueleconomy' / 'vehicles.csv'))
    cp += [(RAW / 'cycles' / f, REFDATA / 'cycles' / f) for f in REF_CYCLES]
    cp += [(RAW / 'cycles' / f'cfr1037_app{x}.xml', REFDATA / 'ecfr' / f'app_{x}_1037.xml') for x in 'AD']
    for s, d in cp:
        if not s.exists():
            log(f'refdata: missing {s}'); continue
        if not d.exists() or d.stat().st_size != s.stat().st_size:
            d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(s, d)
    tools = VEH / 'roadsim' / 'tools'
    must(run_py([tools / 'epa_candidates.py'], 'refdata_epa_candidates'), 'epa_candidates.py')
    # shadow runs of the tools that write tracked roadsim data; outputs compared in verify
    (RS_SHADOW / 'tools').mkdir(parents=True, exist_ok=True)
    (RS_SHADOW / 'roadsim' / 'data').mkdir(parents=True, exist_ok=True)
    link_dir(REFDATA, RS_SHADOW / 'refdata')
    for s in ['build_cycles.py', 'build_generator_calib.py']:
        shutil.copy2(tools / s, RS_SHADOW / 'tools' / s)
        rc = run_py([RS_SHADOW / 'tools' / s], f'refdata_{s[:-3]}')
        log(f'refdata: shadow {s} rc {rc}')
    log('refdata: ready (build_maps.py needs refdata/alpha/, the EPA ALPHA packages, which this build does not fetch)')


# ---------------------------------------------------------------- verify
def _cmp_frames(x, y):
    import numpy as np, pandas as pd
    out = {}
    if x.shape != y.shape:
        out['shape'] = [list(x.shape), list(y.shape)]
    if list(x.columns) != list(y.columns):
        out['columns_only_shipped'] = sorted(set(map(str, x.columns)) - set(map(str, y.columns)))
        out['columns_only_rebuilt'] = sorted(set(map(str, y.columns)) - set(map(str, x.columns)))
    if len(x) != len(y):
        return out
    worst = {}
    for c in [c for c in x.columns if c in y.columns]:
        a, b = x[c], y[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(a):
            av = a.to_numpy(dtype=float, na_value=np.nan); bv = b.to_numpy(dtype=float, na_value=np.nan)
            nan_mis = int(np.sum(np.isnan(av) != np.isnan(bv)))
            m = ~np.isnan(av) & ~np.isnan(bv)
            rel = np.abs(av[m] - bv[m]) / np.maximum(np.abs(av[m]), 1e-12) if m.any() else np.zeros(0)
            r = float(rel.max()) if len(rel) else 0.0
            if nan_mis or r > 1e-9:
                worst[str(c)] = {'max_rel': r, 'n_rel_gt_1e-6': int(np.sum(rel > 1e-6)), 'nan_mismatch': nan_mis}
        else:
            na, nb = a.isna().to_numpy(), b.isna().to_numpy()
            both = ~na & ~nb
            sa, sb = a[both].astype(str).to_numpy(), b[both].astype(str).to_numpy()
            ne, nm = int(np.sum(sa != sb)), int(np.sum(na != nb))
            if ne or nm:
                ex = [[x, y] for x, y in zip(sa[sa != sb][:3], sb[sa != sb][:3])]
                worst[str(c)] = {'n_unequal': ne, 'null_mismatch': nm, 'examples': ex}
    if worst:
        out['columns'] = worst
    return out


def _cmp_json(a, b, path='', out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append((f'{path}/{k}', a.get(k, '<missing>'), b.get(k, '<missing>')))
            else:
                _cmp_json(a[k], b[k], f'{path}/{k}', out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _cmp_json(x, y, f'{path}[{i}]', out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if a != b and abs(a - b) > 1e-9 * max(abs(a), abs(b), 1e-12):
            out.append((path, a, b))
    elif a != b:
        out.append((path, a, b))
    return out


def compare_file(shipped: Path, rebuilt: Path):
    import numpy as np, pandas as pd
    if not rebuilt.exists():
        return {'status': 'not rebuilt'}
    if shipped.read_bytes() == rebuilt.read_bytes():
        return {'status': 'identical'}
    s = shipped.suffix
    try:
        if s == '.parquet':
            x, y = pd.read_parquet(shipped), pd.read_parquet(rebuilt)
            if not isinstance(x.index, pd.RangeIndex):
                d = _cmp_frames(x.reset_index(), y.reset_index())
            else:
                d = _cmp_frames(x, y)
        elif s == '.csv':
            d = _cmp_frames(pd.read_csv(shipped), pd.read_csv(rebuilt))
        elif s == '.json':
            j = _cmp_json(json.loads(shipped.read_text()), json.loads(rebuilt.read_text()))
            d = {'n_diff': len(j), 'diffs': [list(map(lambda v: v if isinstance(v, (int, float, str, type(None))) else str(v)[:200], t)) for t in j[:200]]} if j else {}
        elif s == '.npz':
            with np.load(shipped) as A, np.load(rebuilt) as B:
                d = {}
                for k in sorted(set(A.files) | set(B.files)):
                    if k not in A.files or k not in B.files:
                        d[k] = 'missing'; continue
                    a, b = A[k], B[k]
                    if a.shape != b.shape:
                        d[k] = f'shape {a.shape} vs {b.shape}'
                    elif a.dtype.kind in 'fc':
                        m = np.isfinite(a) & np.isfinite(b)
                        r = float(np.max(np.abs(a[m] - b[m]) / np.maximum(np.abs(a[m]), 1e-12))) if m.any() else 0.0
                        nm = int(np.sum(np.isfinite(a) != np.isfinite(b)))
                        if r > 1e-6 or nm: d[k] = {'max_rel': r, 'nan_mismatch': nm}
                    elif not np.array_equal(a, b):
                        d[k] = 'unequal'
        else:
            d = {'bytes': 'differ'}
    except Exception as e:
        return {'status': 'error', 'error': repr(e)}
    return {'status': 'equal (formatting only)' if not d else 'DIFFERENT', **d}


def stage_verify(a, st):
    rep = {}
    h = sha256(PROC / 'splits.json')
    rep['splits_sha256'] = h
    rep['splits_sha256_ok'] = h == SPLITS_SHA
    rep['splits_rebuilt_equal_to_frozen'] = st.get('splits_rebuilt_equal')
    rep['splits_ved_reproduced'] = st.get('splits_ved_reproduced')
    log(f'verify: splits.json sha256 {h[:8]}… {"OK" if h == SPLITS_SHA else "CHANGED"}')
    # rebuilt tracked tables vs shipped
    sp = SHADOW / 'data' / 'proc'
    files = {}
    if sp.exists():
        for d in ['epa', 'fe', 'cycles', 'hd', 'baselines', 'checks']:
            for f in sorted((PROC / d).rglob('*')):
                if f.is_file():
                    r = str(f.relative_to(PROC)).replace('\\', '/')
                    files[r] = compare_file(f, sp / r)
        files['splits.json (rebuilt)'] = compare_file(PROC / 'splits.json', sp / 'splits.rebuilt.json')
    rs = VEH / 'roadsim' / 'roadsim' / 'data'
    for f in ['cycles.npz', 'cycles_meta.json', 'generator_calib.json']:
        files[f'roadsim/data/{f}'] = compare_file(rs / f, RS_SHADOW / 'roadsim' / 'data' / f)
    rep['files'] = files
    groups = {}
    for k, v in files.items():
        groups.setdefault(v['status'], []).append(k)
    for s_, ks in sorted(groups.items()):
        log(f'verify: {s_:24s} {len(ks):4d}  ' + ('' if s_ in ('identical', 'equal (formatting only)') else ', '.join(ks[:20])))
    # baselines headline numbers
    bs = PROC / 'baselines' / 'summary.json'; br = sp / 'baselines' / 'summary.json'
    if br.exists():
        A, B = json.loads(bs.read_text()), json.loads(br.read_text())
        rows = []
        for m in ['pred_label', 'pred_vsp', 'pred_class']:
            for vt in sorted(set(A['VED_exam'][m]) | set(B['VED_exam'][m])):
                x, y = A['VED_exam'][m].get(vt, {}), B['VED_exam'][m].get(vt, {})
                rows.append([m, vt, x.get('n_trips'), y.get('n_trips'), x.get('trip_MdAPE'), y.get('trip_MdAPE'),
                             x.get('vehicle_total_MdAPE'), y.get('vehicle_total_MdAPE')])
        rep['baselines_VED_exam'] = {'columns': ['method', 'type', 'n_shipped', 'n_rebuilt', 'MdAPE_shipped', 'MdAPE_rebuilt',
                                                 'veh_MdAPE_shipped', 'veh_MdAPE_rebuilt'], 'rows': rows}
        for r in rows:
            log('verify: baseline %-10s %-4s n %s/%s  MdAPE %.4f/%.4f  veh %.4f/%.4f' % tuple(
                [r[0], r[1], r[2], r[3]] + [float('nan') if v is None else v for v in r[4:]]))
    # a dev load through the public loader
    code = ('import sys; sys.path.insert(0, sys.argv[1]); import numpy as np, realdata as rd\n'
            'n = 0\n'
            'for m, a in rd.load_trips("dev", label_ok=True, limit=5, seed=0):\n'
            '    assert np.isfinite(a["grade"]).all() and np.isfinite(a["T_amb"]).all(), m["trip_id"]\n'
            '    n += 1; print(m["trip_id"], m["architecture"], len(a["t"]), "s", round(float(np.nanmax(np.abs(a["grade"]))), 3),\n'
            '          round(float(np.nanmean(a["T_amb"])), 1), "C", round(m["fuel_g"] or 0, 1), "g")\n'
            'print("dev trips loaded", n, "of", len(rd.load_trips("dev", meta_only=True)))\n')
    r = subprocess.run([PY, '-I', '-c', code, str(VEH)], capture_output=True, text=True,
                       env={k: v for k, v in os.environ.items() if k != 'ML_RATCHET_VEHICLE'})
    rep['dev_load'] = (r.stdout + r.stderr).strip().splitlines()[-8:]
    log('verify: dev load\n  ' + '\n  '.join(rep['dev_load']))
    rep['dem_present'] = (PROC / 'dem_annarbor_2m.npz').exists()
    rep['blocked'] = st.get('blocked', {})
    rep['timings_s'] = st.get('timings_s', {})
    (WORK / 'rebuild_report.json').write_text(json.dumps(rep, indent=1, default=str))
    log(f'verify: report in {WORK / "rebuild_report.json"}')
    if not rep['splits_sha256_ok']:
        raise SystemExit('splits.json changed')


# ---------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(prog='mlratchet data vehicle', description=__doc__.split('\n')[0])
    p.add_argument('--only', default='', help='comma-separated stages')
    p.add_argument('--from', dest='start', default='', help='start at this stage')
    p.add_argument('--jobs', type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    p.add_argument('--offline', action='store_true', help='skip the fetch stage')
    p.add_argument('--list', action='store_true', help='list the stages and what is missing')
    a = p.parse_args(argv)
    if a.list:
        print(__doc__)
        print('missing downloads:', [str(j[1].relative_to(RAW)) for j in fetch_jobs(load_state())])
        return 0
    stages = STAGES[STAGES.index(a.start):] if a.start else STAGES
    if a.only:
        stages = [s for s in a.only.split(',') if s]
    if a.offline:
        stages = [s for s in stages if s != 'fetch']
    bad = set(stages) - set(STAGES)
    if bad:
        p.error(f'unknown stages {sorted(bad)}; known: {STAGES}')
    WORK.mkdir(parents=True, exist_ok=True)
    st = load_state()
    before = sha256(PROC / 'splits.json')
    for s in stages:
        t = time.time()
        log(f'== {s}')
        globals()[f'stage_{s}'](a, st)
        st.setdefault('timings_s', {})[s] = round(time.time() - t, 1)
        save_state(st)
        log(f'== {s} done in {time.time() - t:.0f} s')
        assert sha256(PROC / 'splits.json') == before, 'splits.json was modified'
    return 0


if __name__ == '__main__':
    sys.exit(main())
