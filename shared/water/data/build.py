"""WaterData: unpack the shipped data or rebuild it from the sources, then verify against the shipped hashes.

usage: python -I shared/water/data/build.py [--list] [--from-zip PATH] [--only a,b] [--from stage] [--offline]
                                            [--write-manifest]
       (or `mlratchet data water ...`)

Data root: $ML_RATCHET_WATER, default shared/water. Everything below lives under it and is git-ignored:
data/proc/ (except the frozen splits.json), data/raw/, data/logs/, data/work/, checks/out/, checks/ref/*.csv and
watersim/.refenv/ (the isolated R environment for the airGR/TUWmodel checks).

Two routes to the same data root:
  --from-zip watersim-waterdata-v1.0.zip   what the owner shipped (10 MB): data/proc/attributes, specs and baselines
      (3,278 basins' attributes, per-unit baseline scores, calibrated GR4J parameters), the per-basin dev calibrations in
      checks/out/, and the airGR/TUWmodel reference CSVs in checks/ref/. Verified against data/MANIFEST.json. Code in
      the zip is never unpacked (the repository's copy is the code); data/proc/splits.json is never overwritten.
      The zip has NO time series: Caravan forcing and Q, NWIS, NWM, SNOTEL and reservoir records need the stages
      below.
  (default)  run data/scripts/ in the order of waterdata/DATA.md section 9. Every fetcher is resumable (it skips files
      on disk), so an interrupted stage can be re-run.

Stages (in order), with their cost on 18 Nov 2026:
  caravan    fetch_caravan.py: streams Caravan v1.6 (Zenodo 15529786, Caravan-nc.tar.gz, 24.8 GB, CC BY 4.0) once,
             never storing the tarball; keeps 1.7 GB of parquet (data/proc/caravan) plus attributes and licences.
             About 2-3 hours. Restart-safe, but a restart re-reads the stream up to the first missing basin.
  nwis       fetch_nwis.py (USGS site info, daily values 1980-today, annual peaks for 671 gauges; about 2,000 small
             requests), plus the CAMELS-531 list from the neuralhydrology repository; then build_attributes.py,
             build_nwis.py, derive_area.py, build_attributes.py, build_nwis.py (the implied-area fix, DATA.md
             gotcha 1).
  nwm        fetch_nwm.py: NOAA NWM v3.0 retrospective CHANOBS at the CAMELS gauges, one HTTP range request per hour
             (about 385,000 requests of 32 KB from noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com; many hours), then
             build_nwm.py.
  snotel     fetch_snotel.py (NRCS report generator, 934 stations), build_snotel.py.
  reservoirs ResOpsUS v2 zip (Zenodo 6612040, about 0.3 GB, read in place) and the ResOpsUS+CARS GRanD attribute CSVs
             (Zenodo 15978041, two CSVs read from the remote zip by HTTP range), then build_reservoirs.py.
  cdec       fetch_cdec.py: 18 California reservoirs from cdec.water.ca.gov.
  gr4j       run_gr4j.py: per-basin GR4J calibration on every CAMELS-US basin (hours on one core; needs numba).
  baselines  run_baselines_flow.py, run_baselines_peaks.py, run_baselines_detective.py, run_baselines_snow.py,
             run_baselines_reservoir.py, run_baselines_reservoir_recent.py, build_summary.py.
  verify     every shipped file vs data/MANIFEST.json (sha256), the frozen splits' two hashes, a loader smoke test.

The splits are frozen (seed 20261118) and tracked in git; make_splits.py is never run here (it refuses to rewrite a
frozen section anyway). A rebuild from live hosts is not byte-identical (NWIS revises provisional data; SNOTEL and
CDEC add days), so verify reports files that differ instead of failing on them; only the split file must match.
Failed hosts are listed at the end. Do not route around them: record them in the model's help_wanted.
"""
from __future__ import annotations

import argparse
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

CODE = Path(__file__).resolve().parents[1]                         # shared/water (code, scripts, manifest)
ROOT = Path(os.environ.get('ML_RATCHET_WATER') or CODE)             # data root
DATA = ROOT / 'data'
RAW = DATA / 'raw'
PROC = DATA / 'proc'
WORK = DATA / 'work'
STATE = WORK / 'build_state.json'
SCRIPTS = CODE / 'data' / 'scripts'
MANIFEST = CODE / 'data' / 'MANIFEST.json'
FROZEN = CODE / 'data' / 'proc' / 'splits.json'                    # tracked; the data root's copy must equal it
SPLITS_FILE_SHA = '33f88ca0225180dcce979eceed26bf5157b92455e13bb7b78e135e9fd4d98c6b'
SPLITS_FIELD_SHA = 'ed8109e5213e851a2b8513045e903d738180572bced95d7da1c5118a076aacb2'   # the file's own `sha256` field
DATA_PREFIXES = ('data/proc/', 'checks/out/', 'checks/ref/')
DATA_SUFFIX_CHECKS_REF = '.csv'                                     # only the CSVs of checks/ref are data
UA = 'mlratchet-data-rebuild/1.0 (research data rebuild)'
PY = sys.executable
STAGES = ['caravan', 'nwis', 'nwm', 'snotel', 'reservoirs', 'cdec', 'gr4j', 'baselines', 'verify']
DOWNLOADS = ('caravan', 'nwis', 'nwm', 'snotel', 'reservoirs', 'cdec')

LIST_531 = ('https://raw.githubusercontent.com/neuralhydrology/neuralhydrology/master/examples/06-Finetuning/'
            '531_basin_list.txt')
RESOPS_RECORD = 'https://zenodo.org/api/records/6612040'           # ResOpsUS (Steyaert et al. 2022), CC BY 4.0
CARS_RECORD = 'https://zenodo.org/api/records/15978041'            # ResOpsUS+CARS v1.0 (GRanD attributes), CC BY 4.0
CARS_MEMBERS = ('v1.0__attributes__grand.csv', 'v1.0__attributes__resops.csv')


# ---------------------------------------------------------------- helpers
def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def save_state(st):
    WORK.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def splits_field_ok(raw: bytes) -> tuple[str, str]:
    """(stored, recomputed) sha256 of the split content, as make_splits.py computes it: json.dumps(sort_keys=True)
    of everything but the `sha256` key."""
    d = json.loads(raw)
    stored = d.pop('sha256', None)
    return stored, hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


BLOCKED: dict[str, str] = {}


def download(url: str, dest: Path, tries: int = 3) -> bool:
    """Resumable-by-file download: skip if dest exists; write to .part, then rename. Records failures by host."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    host = url.split('/')[2]
    last = ''
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=300) as r, open(part, 'wb') as f:
                shutil.copyfileobj(r, f, 1 << 20)
            part.replace(dest)
            return True
        except urllib.error.HTTPError as e:
            last = f'HTTP {e.code}'
            if e.code in (401, 403, 404, 451):
                break
        except Exception as e:  # timeouts, TLS, DNS
            last = type(e).__name__
        time.sleep(5 * (a + 1))
    BLOCKED.setdefault(host, f'{last} ({url})')
    log('  failed', url, last)
    return False


def zenodo_files(record_url: str) -> list[dict]:
    """The file list of a Zenodo record ({key, size, links.self}); [] when the API cannot be reached."""
    try:
        req = urllib.request.Request(record_url, headers={'User-Agent': UA, 'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r).get('files', [])
    except Exception as e:
        BLOCKED.setdefault(record_url.split('/')[2], f'{type(e).__name__} ({record_url})')
        return []


def run(args: list, env_extra: dict | None = None) -> int:
    """Run one data script with the data root in the environment, isolated (-I), single thread."""
    env = dict(os.environ, ML_RATCHET_WATER=str(ROOT), OMP_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
               PYTHONIOENCODING='utf-8')
    env.update(env_extra or {})
    log('  run', ' '.join(str(a) for a in args))
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / 'logs').mkdir(exist_ok=True)
    return subprocess.run([PY, '-I', '-X', 'utf8', *map(str, args)], env=env, cwd=str(DATA)).returncode   # UTF-8 mode: the scripts write text without an encoding (cp1252 on Windows breaks)


def steps(*calls) -> int:
    """Run (script, args...) in order; stop at the first failure."""
    for c in calls:
        rc = run([SCRIPTS / c[0], *c[1:]])
        if rc:
            log(f'  {c[0]} failed (exit {rc}); stopping this stage')
            return rc
    return 0


def is_data(rel: str) -> bool:
    if rel.startswith('checks/ref/'):
        return rel.endswith(DATA_SUFFIX_CHECKS_REF)
    return rel.startswith(DATA_PREFIXES)


# ---------------------------------------------------------------- unpack
def stage_unpack(zips: list[Path]) -> list[str]:
    man = json.loads(MANIFEST.read_text())['files']
    seen = []
    for zp in zips:
        n_new = n_same = n_skip = 0
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist() if n.strip('/')]
            tops = {n.split('/')[0] for n in names}
            single = next(iter(tops)) if len(tops) == 1 and all('/' in n for n in names) else None
            top = single + '/' if single and single not in ('data', 'checks', 'watersim', 'waterdata') else ''
            for n in names:
                if n.endswith('/') or '..' in n.split('/') or n.startswith('/'):
                    n_skip += 1
                    continue
                rel = n[len(top):] if n.startswith(top) else n
                if rel.endswith('.part') or not is_data(rel):
                    n_skip += 1
                    continue
                seen.append(rel)
                dest = ROOT / rel
                if rel == 'data/proc/splits.json':           # frozen: compare, never write
                    same = hashlib.sha256(z.read(n)).hexdigest() == SPLITS_FILE_SHA
                    log(f'  {zp.name}: splits.json {"identical to" if same else "DIFFERS from"} the frozen file; not written')
                    continue
                info = z.getinfo(n)
                if dest.exists() and dest.stat().st_size == info.file_size and rel in man and sha256(dest) == man[rel][0]:
                    n_same += 1
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(n) as src, open(dest, 'wb') as f:
                    shutil.copyfileobj(src, f, 1 << 20)
                n_new += 1
        log(f'unpack {zp.name}: {n_new} files written, {n_same} already identical, {n_skip} skipped (directories, code)')
    new = sorted(set(r for r in seen if r not in man and r != 'data/proc/splits.json'))
    if new:
        log(f'  {len(new)} unpacked files are not in MANIFEST.json (e.g. {new[0]}); check them, then --write-manifest')
    return new


def ensure_splits():
    """The data root's proc/splits.json is the tracked frozen file (copied when the root is elsewhere)."""
    sp = PROC / 'splits.json'
    if sp.resolve() != FROZEN.resolve() and (not sp.exists() or sha256(sp) != SPLITS_FILE_SHA):
        sp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FROZEN, sp)
        log(f'splits.json copied from the repository into {sp.parent}')


# ---------------------------------------------------------------- stages
def stage_caravan():
    return steps(('fetch_caravan.py',))


def stage_nwis():
    download(LIST_531, RAW / 'lists' / '531_basin_list.txt')
    return steps(('fetch_nwis.py',), ('build_attributes.py',), ('build_nwis.py',), ('derive_area.py',),
                 ('build_attributes.py',), ('build_nwis.py',))


def stage_nwm():
    return steps(('fetch_nwm.py',), ('build_nwm.py',))


def stage_snotel():
    return steps(('fetch_snotel.py',), ('build_snotel.py',))


def stage_reservoirs():
    """ResOpsUS v2 zip (read in place by build_reservoirs.py) and two GRanD attribute CSVs from the ResOpsUS+CARS zip,
    fetched by HTTP range (scripts/httpzip.py) instead of downloading the whole archive."""
    dest = RAW / 'resopsus' / 'ResOpsUS2.zip'
    if not dest.exists():
        files = [f for f in zenodo_files(RESOPS_RECORD) if f.get('key', '').lower().endswith('.zip')]
        if not files:
            log('  ResOpsUS: no zip found in the Zenodo record; download it by hand into', dest)
            return {'missing': str(dest)}
        download(files[0]['links']['self'], dest)
    cars = RAW / 'resopsus_cars'
    if not all((cars / m).exists() for m in CARS_MEMBERS):
        files = [f for f in zenodo_files(CARS_RECORD) if f.get('key', '').lower().endswith('.zip')]
        if not files:
            log('  ResOpsUS+CARS: record unreachable; put', ', '.join(CARS_MEMBERS), 'in', cars)
            return {'missing': str(cars)}
        sys.path.insert(0, str(SCRIPTS))
        import httpzip                                   # noqa: E402  (data only: reads bytes, runs nothing)
        cars.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(httpzip.HttpFile(files[0]['links']['self'])) as z:
            for n in z.namelist():
                flat = n.replace('/', '__')              # v1.0/attributes/grand.csv -> v1.0__attributes__grand.csv
                if flat in CARS_MEMBERS:
                    (cars / flat).write_bytes(z.read(n))
    if not dest.exists():
        return {'missing': str(dest)}
    return steps(('build_reservoirs.py',))


def stage_cdec():
    return steps(('fetch_cdec.py',))


def stage_gr4j():
    return steps(('run_gr4j.py',))


def stage_baselines():
    return steps(('run_baselines_flow.py',), ('run_baselines_peaks.py',), ('run_baselines_detective.py',),
                 ('run_baselines_snow.py',), ('run_baselines_reservoir.py',), ('run_baselines_reservoir_recent.py',),
                 ('build_summary.py',))


def stage_verify() -> int:
    man = json.loads(MANIFEST.read_text())
    same, diff, missing = [], [], []
    for rel, (h, n) in man['files'].items():
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
        elif p.stat().st_size == n and sha256(p) == h:
            same.append(rel)
        else:
            diff.append(rel)
    log(f'verify: {len(same)} of {len(man["files"])} shipped files identical, {len(diff)} differ, {len(missing)} missing')
    for tag, lst in (('differ', diff), ('missing', missing)):
        for r in lst[:15]:
            log(f'  {tag}: {r}')
        if len(lst) > 15:
            log(f'  ... {len(lst) - 15} more {tag}')
    series = {'Caravan forcing + Q (data/proc/caravan/camels)': PROC / 'caravan' / 'camels',
              'NWIS daily (data/proc/nwis/daily)': PROC / 'nwis' / 'daily',
              'NWM daily (data/proc/nwm/daily_local.parquet)': PROC / 'nwm' / 'daily_local.parquet',
              'SNOTEL (data/proc/snotel/daily)': PROC / 'snotel' / 'daily',
              'ResOpsUS (data/proc/reservoirs/daily)': PROC / 'reservoirs' / 'daily'}
    for k, p in series.items():
        n = len(list(p.glob('*'))) if p.is_dir() else int(p.exists())
        log(f'  time series: {k}: {"absent" if not n else f"{n} files" if p.is_dir() else "present"}')
    sp = PROC / 'splits.json'
    ok_file = sp.exists() and sha256(sp) == SPLITS_FILE_SHA
    stored, recomputed = splits_field_ok(sp.read_bytes()) if sp.exists() else (None, None)
    ok_field = bool(stored) and stored == recomputed == SPLITS_FIELD_SHA
    log(f'splits.json: file sha256 {"OK" if ok_file else "WRONG"}; sha256 field {"OK" if ok_field else "WRONG"} '
        f'({(stored or "none")[:12]}..., recomputed {(recomputed or "none")[:12]}...)')
    rep = {'checked': time.strftime('%Y-%m-%d %H:%M'), 'root': str(ROOT), 'identical': len(same), 'differ': diff,
           'missing': missing, 'splits_ok': ok_file and ok_field}
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'verify.json').write_text(json.dumps(rep, indent=1))
    smoke = ('import sys; sys.path.insert(0, sys.argv[1]); import waterdata as wd; '
             'print("loader:", len(wd.basins("camels", split="dev")), "dev /", len(wd.basins("camels", split="exam")),'
             ' "exam CAMELS-US basins;", len(wd.attributes("camels")), "attribute rows")')
    if not any(m.startswith('data/proc/attributes/') for m in missing):
        r = subprocess.run([PY, '-I', '-c', smoke, str(CODE)], env=dict(os.environ, ML_RATCHET_WATER=str(ROOT)))
        if r.returncode:
            return 1
    return 0 if ok_file and ok_field and not missing and not diff else 1


def write_manifest():
    """Re-hash every shipped data file under the root (data/proc minus the time series, checks/out, checks/ref CSVs)."""
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    skip = ('data/proc/caravan/', 'data/proc/nwis/', 'data/proc/nwm/', 'data/proc/snotel/', 'data/proc/reservoirs/',
            'data/proc/baselines/sims/')                    # rebuilt time series: never in the shipped zip
    files = {}
    for pre in DATA_PREFIXES:
        base = ROOT / pre
        for p in sorted(base.rglob('*')) if base.exists() else []:
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and is_data(rel) and not rel.startswith(skip) and not rel.endswith('.part') \
                    and rel != 'data/proc/splits.json':
                files[rel] = [sha256(p), p.stat().st_size]
    files['data/proc/splits.json'] = [SPLITS_FILE_SHA, FROZEN.stat().st_size]
    files = dict(sorted(files.items()))
    old.update(source='watersim-waterdata-v1.0.zip (18 Nov 2026)', n_files=len(files),
               bytes=sum(v[1] for v in files.values()), files=files)
    MANIFEST.write_text(json.dumps(old, indent=1) + '\n', encoding='utf-8', newline='\n')
    log(f'MANIFEST.json: {len(files)} files, {old["bytes"] / 1e6:.1f} MB')


FUNCS = {s: globals()[f'stage_{s}'] for s in STAGES}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--only', help='comma-separated stages')
    ap.add_argument('--from', dest='start', help='start at this stage')
    ap.add_argument('--from-zip', type=Path, nargs='+', help='unpack the shipped zip instead of downloading, then verify')
    ap.add_argument('--offline', action='store_true', help='skip every stage that downloads (' + ' '.join(DOWNLOADS) + ')')
    ap.add_argument('--write-manifest', action='store_true', help='re-hash the shipped data on disk into data/MANIFEST.json')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args(argv)
    if a.list:
        print(__doc__)
        return 0
    log(f'water data root: {ROOT}')
    ensure_splits()
    if a.write_manifest:
        write_manifest()
        return 0
    if a.from_zip:
        stage_unpack(a.from_zip)
        return stage_verify()
    stages = STAGES
    if a.only:
        stages = [s.strip() for s in a.only.split(',')]
        bad = [s for s in stages if s not in FUNCS]
        if bad:
            ap.error(f'unknown stage(s): {bad}; stages: {STAGES}')
    elif a.start:
        stages = STAGES[STAGES.index(a.start):]
    if a.offline:
        stages = [s for s in stages if s not in DOWNLOADS]
    st = load_state()
    rc = 0
    for s in stages:
        log(f'== {s}')
        t = time.time()
        r = FUNCS[s]()
        st[s] = {'done': time.strftime('%Y-%m-%d %H:%M'), 'seconds': round(time.time() - t)}
        if isinstance(r, dict):
            st[s].update(r)
        elif r:
            st[s]['rc'] = r
        save_state(st)
        if s == 'verify':
            rc = r
    if BLOCKED:
        log('hosts that failed (record them in help_wanted; do not route around them):')
        for h, why in BLOCKED.items():
            log(f'  {h}: {why}')
        st['blocked'] = BLOCKED
        save_state(st)
    return rc


if __name__ == '__main__':
    sys.exit(main())
