"""FireData: unpack the shipped data or rebuild it from the sources, then verify against the shipped hashes.

usage: python -I shared/fire/data/build.py [--list] [--from-zip PATH] [--only a,b] [--from stage] [--offline]
                                           [--write-manifest [--from-zip PATH]]
       (or `mlratchet data fire ...`)

Data root: $ML_RATCHET_FIRE, default shared/fire. Everything below lives under it and is git-ignored:
data/proc/ (except the frozen splits.json), data/raw/, data/tmp/, data/logs/, data/work/, checks/out/,
firesim/refdata/ (USFS GTR PDFs and their text, 39 MB), firesim/demo_out/ and firesim/.refenv/ (the isolated
pyretechnics and xclim environment for the cross-checks; not shipped, build it by hand as in firesim/README.md).

Two routes to the same data root:
  --from-zip firesim-firedata-v1.0.zip   what the owner shipped (135 MB zip, 19 Nov 2026). Unpacked: data/proc/ (fire
      table, FireSize split, exam guard footprints, LANDFIRE fuel groups, the 40 dev fires' landscapes and hourly HRRR,
      baseline summaries and the dev/exam fire-day tables), checks/out/ (pyretechnics values, xclim FWI, VIIRS and
      replay caches), firesim/refdata/ (the GTR-371/153/143 PDFs and text that check_published.py reads) and
      firesim/demo_out/ (the two README figures). Verified against data/MANIFEST.json. Code in the zip is never
      unpacked (the repository's copy is the code); data/proc/splits.json is never overwritten.
      The zip has NO progression data (FEDS-MTBS perimeters, VIIRS pixels), no WildfireSpreadTS, FPA-FOD,
      Globe-LFMC or gridMET, and no landscapes or HRRR for the 48 exam replay fires: those need the stages below.
  (default)  run data/scripts/ in the order of data/DATA.md section 1. Every script is resumable (it skips outputs on
      disk), so an interrupted stage can be re-run. The default route needs the frozen tables from the zip first
      (data/proc/fires.parquet, firesize_split.parquet, guard/, baselines/replay_subset.json): they are outputs of
      make_splits.py and make_replay_subset.py, which are frozen and never run here.

Stages (in order), with their size and time where the handoff recorded them (19 Nov 2026):
  feds       FEDS-MTBS (Zenodo 20187963, CC BY 4.0): FEDS25MTBS.zip and the fires list, then build_feds.py
             -> data/proc/feds/ (12-hourly perimeters, fire lines, VIIRS pixels for 7,739 MTBS fires 2012-2024;
             171 MB processed). Raw zip size and time were not recorded.
  gofer      GOFER-Combined (Zenodo 14642378, CC BY 4.0), hourly GOES perimeters of 28 California fires, then
             build_gofer.py -> data/proc/gofer/. Small.
  wsts       WildfireSpreadTS (Zenodo 8006177, CC BY 4.0) streamed by HTTP range reads, never storing the 48 GB zip:
             list_wsts.py (zip index), fetch_wsts.py meta (static layers), fetch_wsts.py full (daily VIIRS AF and
             weather, dev fires first) -> data/proc/wsts/ (607 fires, 269 MB). About 2 fires per minute: 5 h or more.
  fpafod     FPA-FOD 6th ed. (USFS RDS-2013-0009.6) GeoPackage zip, then build_fpafod.py -> data/proc/fpafod/fod.parquet
             (1.59 M CONUS records 2001-2020). The 0.93 GB extracted GeoPackage is deleted after; the zip is kept.
  lfmc       Globe-LFMC 2.0 (figshare 25413790, CC0) xlsx, then build_lfmc.py -> data/proc/lfmc/lfmc_all.parquet.
             KNOWN GAP (found on import, 21 Nov 2026): gridmet and the moisture baselines read lfmc_us.parquet (CONUS
             only, with `date` and renamed columns), and no shipped script writes it. This stage reports it missing.
  gridmet    build_gridmet.py: gridMET daily at fire centroids, FPA-FOD points and Globe-LFMC sites, 2012-2024, from
             northwestknowledge.net. 12 variables x 13 years of 100-300 MB files, each downloaded, read and deleted
             (tens of GB of transfer, many hours). Needs fpafod and lfmc first.
  landscapes LANDFIRE (lfps.usgs.gov ImageServer) and hourly HRRR (hrrrzarr on AWS Open Data) for the 40 dev fires and
             the 48 exam replay fires: build_landscapes.py, build_weather.py (+ --repair), build_precip.py (+ --repair),
             the order of the sandbox's chain_packages.sh and chain_repair.sh. Dev: 38 MB landscapes, 33 MB HRRR (in
             the zip). Replay fires: not in the zip; time not recorded.
  baselines  the classical baselines, run only for outputs that are MISSING (the shipped ones are frozen: firedata
             STATUS says never re-run the exam baselines over them): baselines_spread.py dev/exam (spread_exam.jsonl,
             89 MB, is not in the zip), summarize_spread.py, build_fire_days.py, baselines_firesize.py,
             baselines_moisture.py (writes the frozen moisture_split.json, seed 20261118 + 3, which the zip lacks).
  verify     every shipped file vs data/MANIFEST.json (sha256), the frozen splits' two hashes, a loader smoke test.

The splits are frozen (seed 20261118, 18 Nov 2026 23:44Z) and tracked in git; make_splits.py and
make_replay_subset.py are never run here (both refuse to rewrite anyway). A rebuild from live hosts may not be
byte-identical, so verify reports files that differ instead of failing on them; only the split file must match.
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

CODE = Path(__file__).resolve().parents[1]                         # shared/fire (code, scripts, manifest)
ROOT = Path(os.environ.get('ML_RATCHET_FIRE') or CODE)              # data root
DATA = ROOT / 'data'
RAW = DATA / 'raw'
PROC = DATA / 'proc'
WORK = DATA / 'work'
STATE = WORK / 'build_state.json'
SCRIPTS = CODE / 'data' / 'scripts'
MANIFEST = CODE / 'data' / 'MANIFEST.json'
FROZEN = CODE / 'data' / 'proc' / 'splits.json'                    # tracked; the data root's copy must equal it
SPLITS_FILE_SHA = '70f655969586d73dfbe6af5e5c3c7c1bd11ac96e011451acfe1bc3132d017fd2'
SPLITS_FIELD_SHA = 'd73e2d9f79e62c48ee17047c1e5e17e4212899c71c656603c27f9b6601b08fe0'   # the file's own `sha256` field
DATA_PREFIXES = ('data/proc/', 'checks/out/', 'firesim/refdata/', 'firesim/demo_out/')
ZIP_SOURCE = 'firesim-firedata-v1.0.zip (19 Nov 2026)'
UA = 'mlratchet-data-rebuild/1.0 (research data rebuild)'
PY = sys.executable
STAGES = ['feds', 'gofer', 'wsts', 'fpafod', 'lfmc', 'gridmet', 'landscapes', 'baselines', 'verify']
DOWNLOADS = ('feds', 'gofer', 'wsts', 'fpafod', 'lfmc', 'gridmet', 'landscapes')
FROZEN_TABLES = ('fires.parquet', 'firesize_split.parquet', 'guard/exam_footprints.parquet',
                 'baselines/replay_subset.json')

FEDS_RECORD = 'https://zenodo.org/api/records/20187963'            # FEDS-MTBS (Chen et al. 2022 method), CC BY 4.0
FEDS_FILES = ('FEDS25MTBS.zip', 'fireslist_FEDS25MTBS_2012-2024.geojson')
GOFER_RECORD = 'https://zenodo.org/api/records/14642378'           # GOFER-Combined (Liu et al.), CC BY 4.0
FPAFOD_ZIP = 'https://www.fs.usda.gov/rds/archive/products/RDS-2013-0009.6/RDS-2013-0009.6_Data_Format2_GPKG.zip'
FPAFOD_PAGE = 'https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6'
LFMC_ARTICLE = 'https://api.figshare.com/v2/articles/25413790'     # Globe-LFMC 2.0 (Yebra et al. 2024), CC0
LFMC_FILE = 'Globe-LFMC-2.0_final.xlsx'


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
    """(stored, recomputed) sha256 of the split content, as make_splits.py computes it: json.dumps(sort_keys=True,
    default=str) of everything but the `sha256` key."""
    d = json.loads(raw)
    stored = d.pop('sha256', None)
    return stored, hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


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


def api_json(url: str):
    """A JSON API answer (Zenodo record, figshare article); None when the host cannot be reached."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except Exception as e:
        BLOCKED.setdefault(url.split('/')[2], f'{type(e).__name__} ({url})')
        return None


def zenodo_files(record_url: str) -> list[dict]:
    """The file list of a Zenodo record ({key, size, links.self}); [] when the API cannot be reached."""
    d = api_json(record_url)
    return (d or {}).get('files', [])


def extract_data(zp: Path, dest: Path):
    """Extract a downloaded archive into its own folder (data only: names with '..' or absolute paths are skipped)."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.endswith('/') or '..' in n.replace('\\', '/').split('/') or n.startswith(('/', '\\')) or ':' in n:
                continue
            out = dest / n
            if out.exists() and out.stat().st_size == z.getinfo(n).file_size:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(out, 'wb') as f:
                shutil.copyfileobj(src, f, 1 << 20)


def run(args: list, env_extra: dict | None = None) -> int:
    """Run one data script with the data root in the environment, isolated (-I), single thread, cwd = data/."""
    env = dict(os.environ, ML_RATCHET_FIRE=str(ROOT), OMP_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
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
    return rel.startswith(DATA_PREFIXES) and not rel.endswith('.part')


def zip_members(z: zipfile.ZipFile):
    """(member name, path relative to the data root) for the zip's data files; code is never returned."""
    names = [n for n in z.namelist() if n.strip('/')]
    tops = {n.split('/')[0] for n in names}
    single = next(iter(tops)) if len(tops) == 1 and all('/' in n for n in names) else None
    top = single + '/' if single and single not in ('data', 'checks', 'firesim', 'firedata') else ''
    for n in names:
        if n.endswith('/') or '..' in n.split('/') or n.startswith('/'):
            continue
        rel = n[len(top):] if n.startswith(top) else n
        if is_data(rel):
            yield n, rel


def need_frozen_tables() -> list[str]:
    missing = [t for t in FROZEN_TABLES if not (PROC / t).exists()]
    if missing:
        log('  frozen tables missing:', ', '.join(missing), '- unpack the release first: mlratchet data fire --from-zip '
            'firesim-firedata-v1.0.zip (make_splits.py / make_replay_subset.py are frozen and never run here)')
    return missing


# ---------------------------------------------------------------- unpack
def stage_unpack(zips: list[Path]) -> list[str]:
    man = json.loads(MANIFEST.read_text())['files'] if MANIFEST.exists() else {}
    seen = []
    for zp in zips:
        n_new = n_same = 0
        with zipfile.ZipFile(zp) as z:
            n_all = sum(1 for n in z.namelist() if not n.endswith('/'))
            for n, rel in zip_members(z):
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
        log(f'unpack {zp.name}: {n_new} files written, {n_same} already identical, '
            f'{n_all - n_new - n_same - 1} skipped (code, docs, splits.json)')
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
def stage_feds():
    raw = RAW / 'feds_mtbs'
    if not all((raw / f).exists() for f in FEDS_FILES):
        files = {f.get('key'): f for f in zenodo_files(FEDS_RECORD)}
        for name in FEDS_FILES:
            if name in files:
                download(files[name]['links']['self'], raw / name)
        if not all((raw / f).exists() for f in FEDS_FILES):
            log('  FEDS-MTBS: put', ', '.join(FEDS_FILES), 'in', raw, f'(record keys seen: {sorted(files)[:8]})')
            return {'missing': str(raw)}
    return steps(('build_feds.py',))


def stage_gofer():
    x = RAW / 'gofer' / 'x'
    if not (x / 'GOFER' / 'fireData.csv').exists():
        zips = [f for f in zenodo_files(GOFER_RECORD) if f.get('key', '').lower().endswith('.zip')]
        if not zips:
            log('  GOFER: no zip found in the Zenodo record; unpack it by hand into', x, '(GOFER/GOFER_Combined/*.shp)')
            return {'missing': str(x)}
        for f in zips:
            dest = RAW / 'gofer' / f['key']
            if download(f['links']['self'], dest):
                extract_data(dest, x)
        if not (x / 'GOFER' / 'fireData.csv').exists():
            return {'missing': str(x / 'GOFER')}
    return steps(('build_gofer.py',))


def stage_wsts():
    if not (RAW / 'wsts' / 'zip_index.json').exists():
        rc = steps(('list_wsts.py',))
        if rc:
            return rc
    # dev fires first, then the rest in name order (the sandbox's proc/wsts/order.txt was "dev first, exam in a
    # seeded order" and was not shipped; the order changes only which fires an interrupted run has, not their content)
    sp = json.loads(FROZEN.read_text())['wsts']
    order = sorted(k for k, v in sp.items() if v.get('split') == 'dev') + sorted(k for k, v in sp.items()
                                                                                if v.get('split') != 'dev')
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'wsts_order.txt').write_text('\n'.join(order) + '\n')
    return steps(('fetch_wsts.py', 'meta'), ('fetch_wsts.py', 'full', WORK / 'wsts_order.txt'))


def stage_fpafod():
    gpkg = RAW / 'fpafod' / 'x' / 'Data' / 'FPA_FOD_20221014.gpkg'
    if not (PROC / 'fpafod' / 'fod.parquet').exists() and not gpkg.exists():
        zp = RAW / 'fpafod' / FPAFOD_ZIP.rsplit('/', 1)[1]
        if not download(FPAFOD_ZIP, zp):
            log('  FPA-FOD: download the GeoPackage zip from', FPAFOD_PAGE, 'into', zp.parent)
            return {'missing': str(zp)}
        extract_data(zp, RAW / 'fpafod' / 'x')
        if not gpkg.exists():
            log('  FPA-FOD: the zip has no Data/FPA_FOD_20221014.gpkg; check the edition')
            return {'missing': str(gpkg)}
    if (PROC / 'fpafod' / 'fod.parquet').exists():
        return 0
    (PROC / 'fpafod').mkdir(parents=True, exist_ok=True)
    return steps(('build_fpafod.py',))


def stage_lfmc():
    xl = RAW / 'lfmc' / LFMC_FILE
    if not xl.exists():
        art = api_json(LFMC_ARTICLE) or {}
        f = next((f for f in art.get('files', []) if f.get('name') == LFMC_FILE), None)
        if not f or not download(f['download_url'], xl):
            log('  Globe-LFMC: put', LFMC_FILE, 'in', xl.parent, '(figshare 10.6084/m9.figshare.25413790)')
            return {'missing': str(xl)}
    if not (PROC / 'lfmc' / 'lfmc_all.parquet').exists():
        rc = steps(('build_lfmc.py',))
        if rc:
            return rc
    if not (PROC / 'lfmc' / 'lfmc_us.parquet').exists():
        log('  lfmc_us.parquet is missing: no shipped script writes it from lfmc_all.parquet (CONUS filter, `date`'
            ' column). gridmet and the moisture baselines need it. Known gap, recorded in shared/fire/STATUS.md.')
        return {'missing': str(PROC / 'lfmc' / 'lfmc_us.parquet')}
    return 0


def stage_gridmet():
    if need_frozen_tables():
        return {'missing': 'frozen tables'}
    for need in (PROC / 'fpafod' / 'fod.parquet', PROC / 'lfmc' / 'lfmc_us.parquet'):
        if not need.exists():
            log('  gridmet needs', need, '(stages fpafod and lfmc)')
            return {'missing': str(need)}
    return steps(('build_gridmet.py',))


def stage_landscapes():
    if need_frozen_tables():
        return {'missing': 'frozen tables'}
    return steps(('build_landscapes.py', 'dev', 'replay'),
                 ('build_weather.py', 'dev', 'replay'), ('build_weather.py', 'dev', 'replay', '--repair'),
                 ('build_precip.py', 'dev', 'replay'), ('build_precip.py', 'dev', 'replay', '--repair'),
                 ('build_precip.py', 'dev', 'replay'))


def stage_baselines():
    """Only outputs that are missing are built: the shipped baseline files are frozen exam numbers."""
    if need_frozen_tables():
        return {'missing': 'frozen tables'}
    B = PROC / 'baselines'
    plan = [(('baselines_spread.py', 'dev'), [B / 'spread_dev.jsonl', B / 'spread_params.json']),
            (('baselines_spread.py', 'exam'), [B / 'spread_exam.jsonl']),
            (('summarize_spread.py',), [B / 'spread_summary.json']),
            (('build_fire_days.py',), [B / 'fire_days_dev.parquet', B / 'fire_days_exam.parquet']),
            (('baselines_firesize.py',), [B / 'firesize_exam.json']),
            (('baselines_moisture.py',), [B / 'moisture_exam.json', PROC / 'moisture_split.json'])]
    for call, outs in plan:
        if all(o.exists() for o in outs):
            log(f'  {" ".join(call)}: outputs present (frozen), not re-run')
            continue
        if any(o.exists() for o in outs):
            log(f'  {" ".join(call)}: some outputs present ({", ".join(o.name for o in outs if o.exists())}); the '
                f'script resumes or skips them')
        rc = steps(call)
        if rc:
            return rc
    return 0


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
    built = {'FEDS-MTBS progression (data/proc/feds/perims)': PROC / 'feds' / 'perims',
             'GOFER (data/proc/gofer)': PROC / 'gofer',
             'WildfireSpreadTS (data/proc/wsts/fires)': PROC / 'wsts' / 'fires',
             'FPA-FOD (data/proc/fpafod/fod.parquet)': PROC / 'fpafod' / 'fod.parquet',
             'Globe-LFMC (data/proc/lfmc)': PROC / 'lfmc',
             'gridMET points (data/proc/weather/gridmet)': PROC / 'weather' / 'gridmet',
             'landscapes, dev + replay (data/proc/landscapes)': PROC / 'landscapes',
             'exam spread rows (data/proc/baselines/spread_exam.jsonl)': PROC / 'baselines' / 'spread_exam.jsonl',
             'reference PDFs (firesim/refdata)': ROOT / 'firesim' / 'refdata',
             'pyretechnics/xclim venv (firesim/.refenv)': ROOT / 'firesim' / '.refenv'}
    for k, p in built.items():
        n = len(list(p.glob('*'))) if p.is_dir() else int(p.exists())
        log(f'  {k}: {"absent" if not n else f"{n} files" if p.is_dir() else "present"}')
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
    smoke = ('import sys; sys.path.insert(0, sys.argv[1])\n'
             'try:\n    import firedata as fd\n'
             'except ImportError as e:\n    print("loader smoke test skipped:", e, "(pip install -e .[fire])"); sys.exit(0)\n'
             'f = fd.fires("dev"); L = fd.landscape(f.Event_ID.iloc[0])\n'
             'print("loader:", len(f), "dev fires,", len(fd.fires("exam")), "exam fires; landscape", '
             'f.Event_ID.iloc[0], L["layers"]["fuel"].shape)')
    if not any(m.startswith(('data/proc/fires.parquet', 'data/proc/landscapes/')) for m in missing):
        r = subprocess.run([PY, '-I', '-c', smoke, str(CODE)], env=dict(os.environ, ML_RATCHET_FIRE=str(ROOT)))
        if r.returncode:
            return 1
    return 0 if ok_file and ok_field and not missing and not diff else 1


def write_manifest(zips: list[Path] | None = None):
    """Hash the shipped data files: straight from the release zip (--from-zip), or, without it, re-hash on disk the
    files the manifest already lists. The frozen splits.json is always the tracked file."""
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    files = {}
    if zips:
        for zp in zips:
            with zipfile.ZipFile(zp) as z:
                for n, rel in zip_members(z):
                    if rel == 'data/proc/splits.json':
                        continue
                    h = hashlib.sha256()
                    with z.open(n) as f:
                        for b in iter(lambda: f.read(1 << 20), b''):
                            h.update(b)
                    files[rel] = [h.hexdigest(), z.getinfo(n).file_size]
    else:
        for rel in old.get('files', {}):
            p = ROOT / rel
            if p.is_file() and rel != 'data/proc/splits.json':
                files[rel] = [sha256(p), p.stat().st_size]
    files['data/proc/splits.json'] = [SPLITS_FILE_SHA, FROZEN.stat().st_size]
    files = dict(sorted(files.items()))
    old.update(source=ZIP_SOURCE, n_files=len(files), bytes=sum(v[1] for v in files.values()), files=files)
    MANIFEST.write_text(json.dumps(old, indent=1) + '\n', encoding='utf-8', newline='\n')
    log(f'MANIFEST.json: {len(files)} files, {old["bytes"] / 1e6:.1f} MB')


FUNCS = {s: globals()[f'stage_{s}'] for s in STAGES}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--only', help='comma-separated stages')
    ap.add_argument('--from', dest='start', help='start at this stage')
    ap.add_argument('--from-zip', type=Path, nargs='+', help='unpack the shipped zip instead of downloading, then verify')
    ap.add_argument('--offline', action='store_true', help='skip every stage that downloads (' + ' '.join(DOWNLOADS) + ')')
    ap.add_argument('--write-manifest', action='store_true',
                    help='hash the shipped data into data/MANIFEST.json (from the zip with --from-zip, else on disk)')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args(argv)
    if a.list:
        print(__doc__)
        return 0
    log(f'fire data root: {ROOT}')
    if a.write_manifest:
        write_manifest(a.from_zip)
        return 0
    ensure_splits()
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
        if a.start not in FUNCS:
            ap.error(f'unknown stage {a.start!r}; stages: {STAGES}')
        stages = STAGES[STAGES.index(a.start):]
    if a.offline:
        stages = [s for s in stages if s not in DOWNLOADS]
    st = load_state()
    rc, failed = 0, []
    for s in stages:
        log(f'== {s}')
        t = time.time()
        r = FUNCS[s]()
        st[s] = {'done': time.strftime('%Y-%m-%d %H:%M'), 'seconds': round(time.time() - t)}
        if isinstance(r, dict):
            st[s].update(r)
            if r.get('missing'):
                failed.append(s)
        elif r:
            st[s]['rc'] = r
            failed.append(s)
            rc = rc or r                                 # the first failing stage's exit code; later stages still run
        save_state(st)
    if failed:
        log('stages that failed or are missing data:', ', '.join(failed))
        rc = rc or 1
    if BLOCKED:
        log('hosts that failed (record them in help_wanted; do not route around them):')
        for h, why in BLOCKED.items():
            log(f'  {h}: {why}')
        st['blocked'] = BLOCKED
        save_state(st)
    return rc


if __name__ == '__main__':
    sys.exit(main())
