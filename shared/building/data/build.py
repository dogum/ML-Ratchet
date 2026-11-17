"""BuildingData: unpack the shipped data or rebuild it from the sources, then verify against the shipped hashes.

usage: python -I shared/building/data/build.py [--list] [--from-zip PATH [PATH ...]] [--only a,b] [--from stage]
                                               [--offline] [--write-manifest]
       (or `mlratchet data building ...`)

Data root: $ML_RATCHET_BUILDING, default shared/building. Everything here lives under it and is git-ignored:
data/proc/ (except the frozen splits.json), data/raw/, data/tmp/, data/work/, buildingsim/refdata/, checks/out/.

Two routes to the same data root:
  --from-zip PATH [PATH ...]   unpack what the owner shipped, then verify every file against data/MANIFEST.json. Takes:
      buildingsim-buildingdata-v1.0.zip  the full release (230 MB; 409 data files): data/proc/**, buildingsim/refdata/**,
                                         the per-unit twin results in checks/out/
      buildingsim-minusdata.zip          the release without data/proc (14 MB): buildingsim/refdata/** (EPW/TMYx weather,
                                         BESTEST weather, priors) and the per-unit twin results in checks/out/ that
                                         checks/check_resstock.py and check_comstock.py summarise
      proc/<folder>.zip                  one data/proc folder per zip, top folder = its name: baselines, eia, faults,
                                         tariffs, tmy, bdg2, other
    Code in a zip is never unpacked (the repository's copy is the code). data/proc/splits.json is never overwritten.
    Leftover partial files (*.part) are skipped.
  (default)  run data/scripts/ in order (DATA.md section 13). Resumable: every download is skipped when its file exists,
    and the scripts skip what they already wrote. Hours, mostly the NREL stock samples (about 1 GB over HTTPS).

Stages (in order):
  fetch      raw files no script downloads: EIA RECS 2020 v7 and CBECS 2018 CSVs (www.eia.gov), the OpenEI URDB bulk
             CSV (apps.openei.org), figshare article 29297999's file list (api.figshare.com). Manual, recorded in
             work/build_state.json: LBNL FDD synthesis (figshare 11752740) -> raw/faults/lbnl_synthesis/, Korean office
             AHU (figshare 25932409) -> raw/faults/korea_office/x/FDD_dataset/, HUE (Harvard Dataverse
             doi:10.7910/DVN/N3HGRN) -> raw/other/hue/, WPuQ 60-min HDF5 (Zenodo 10.5281/zenodo.5642902) -> raw/other/wpuq/
  bdg2       scripts/fetch_bdg2.sh (BDG2 CSVs, CC BY-SA 4.0, media.githubusercontent.com + raw.githubusercontent.com;
             needs bash and curl), build_bdg2.py meters summary
  weather    weather_fetch.py screen; build_bdg2.py points (reads GridData's raw/weather/asos_stations.parquet);
             weather_fetch.py power (power.larc.nasa.gov), asos (mesonet.agron.iastate.edu); build_bdg2.py weather
  stock      stock_meta.py resstock, comstock (oedi-data-lake.s3.amazonaws.com); build_bdg2.py spec; stock_ts.py sample,
             fetch resstock, fetch comstock, weather (MODELLED data; never shipped)
  eia        build_recs_cbecs.py
  faults     fetch_ghal.py (figshare range reads), build_faults.py
  tmy        fetch_tmy.py (climate.onebuilding.org)
  tariffs    build_tariffs.py
  other      build_other.py
  splits     make_splits.py in data/work/splits_rebuild/ (never over the frozen file); its sha256 field must equal the
             frozen file's (7ac15e9f...)
  baselines  baselines.py load detective faults thermostat summary
  priors     buildingsim/tools/build_priors.py -> buildingsim/refdata/priors/ (needs the stock metadata)
  verify     every file vs data/MANIFEST.json (sha256), splits hashes, a loader smoke test

No script fetches buildingsim/refdata/weather/ (17 TMYx cities and the BESTEST Denver TMY, climate.onebuilding.org;
TMYx files carry their own terms, so they are not committed): it comes from buildingsim-minusdata.zip.
The verify stage reports files that differ instead of failing on them: a rebuild from live hosts is not byte-identical
(URDB and NASA POWER change). Only the split file must match exactly. --write-manifest re-hashes what is on disk
(use it once, when a new chunk such as bdg2 has been checked).
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

CODE = Path(__file__).resolve().parents[1]                         # shared/building (code, scripts, manifest)
ROOT = Path(os.environ.get('ML_RATCHET_BUILDING') or CODE)          # data root
DATA = ROOT / 'data'
RAW = DATA / 'raw'
PROC = DATA / 'proc'
WORK = DATA / 'work'
STATE = WORK / 'build_state.json'
SCRIPTS = CODE / 'data' / 'scripts'
MANIFEST = CODE / 'data' / 'MANIFEST.json'
FROZEN = CODE / 'data' / 'proc' / 'splits.json'                    # tracked; the data root's copy must equal it
SPLITS_FILE_SHA = '524aee157fffff7c45f3d119823bb3ccd81e73046bf9f15e8e8af523185fba75'
SPLITS_FIELD_SHA = '7ac15e9f56a1979b0fca9fcb17e8577385901ddef1bab950f22655e5573342d2'   # the file's own `sha256` field
PROC_DIRS = ('bdg2', 'other', 'stock', 'eia', 'faults', 'tariffs', 'tmy', 'baselines')
TWINS = ('checks/out/resstock/', 'checks/out/comstock/', 'checks/out/comstock_heldout12/')
DATA_PREFIXES = ('data/proc/', 'buildingsim/refdata/') + TWINS
UA = 'mlratchet-data-rebuild/1.0 (research data rebuild)'
PY = sys.executable
STAGES = ['fetch', 'bdg2', 'weather', 'stock', 'eia', 'faults', 'tmy', 'tariffs', 'other', 'splits', 'baselines',
          'priors', 'verify']
DOWNLOADS = ('fetch', 'bdg2', 'weather', 'stock', 'faults', 'tmy')

FETCH = {   # documented direct URLs (DATA.md section 2)
    'recs/recs2020_public_v7.csv': 'https://www.eia.gov/consumption/residential/data/2020/csv/recs2020_public_v7.csv',
    'cbecs/cbecs2018_final_public.csv': 'https://www.eia.gov/consumption/commercial/data/2018/xls/cbecs2018_final_public.csv',
    'tariffs/usurdb.csv.gz': 'https://apps.openei.org/USURDB/download/usurdb.csv.gz',
}
MANUAL = {
    'raw/faults/lbnl_synthesis/': 'LBNL FDD data synthesis, CC0, figshare doi:10.6084/m9.figshare.11752740',
    'raw/faults/korea_office/x/FDD_dataset/combined_data.csv': 'Korean office AHU, CC BY 4.0, figshare doi:10.6084/m9.figshare.25932409.v2',
    'raw/other/hue/': 'HUE, CC BY 4.0, Harvard Dataverse doi:10.7910/DVN/N3HGRN',
    'raw/other/wpuq/': 'WPuQ 60-min HDF5 + weather, CC BY 4.0, Zenodo doi:10.5281/zenodo.5642902',
}


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
    """(stored, recomputed) sha256 of the split content, computed as make_splits.py does: json.dumps(sort_keys=True)
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


def run(args: list, env_extra: dict | None = None) -> int:
    """Run one data script with the data root in the environment, isolated (-I), single thread."""
    env = dict(os.environ, ML_RATCHET_BUILDING=str(ROOT), OMP_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
    env.update(env_extra or {})
    log('  run', ' '.join(str(a) for a in args))
    cmd = ['bash', *map(str, args)] if str(args[0]).endswith('.sh') else [PY, '-I', *map(str, args)]
    DATA.mkdir(parents=True, exist_ok=True)
    return subprocess.run(cmd, env=env, cwd=str(DATA)).returncode


def steps(*calls) -> int:
    """Run (script, args...) in order; stop at the first failure."""
    for c in calls:
        rc = run([SCRIPTS / c[0], *c[1:]])
        if rc:
            log(f'  {c[0]} failed (exit {rc}); stopping this stage')
            return rc
    return 0


# ---------------------------------------------------------------- unpack
def member_target(name: str, top: str | None, chunk: str | None) -> str | None:
    """Data-root-relative path for a zip member, or None when it is code, a directory, or not ours to unpack."""
    if name.endswith('/') or '..' in name.split('/') or name.startswith('/'):
        return None
    if chunk:                                        # proc/<folder>.zip: faults/... -> data/proc/faults/...
        rel = 'data/proc/' + name
    else:
        rel = name[len(top):] if top else name
    if rel.endswith('.part') or not rel.startswith(DATA_PREFIXES):
        return None
    return rel


def stage_unpack(zips: list[Path]) -> list[str]:
    man = json.loads(MANIFEST.read_text())['files']
    seen = []
    for zp in zips:
        n_new = n_same = n_skip = 0
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist() if n.strip('/')]
            tops = {n.split('/')[0] for n in names}
            single = next(iter(tops)) if len(tops) == 1 and all('/' in n for n in names) else None
            chunk = single if single in PROC_DIRS else None
            top = single + '/' if single and not chunk and single not in ('data', 'buildingsim', 'checks') else None
            for n in names:
                rel = member_target(n, top, chunk)
                if rel is None:
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
        kind = f'proc chunk {chunk}/' if chunk else 'release'
        log(f'unpack {zp.name} ({kind}): {n_new} files written, {n_same} already identical, {n_skip} skipped (directories, code, *.part)')
    new = sorted(set(r for r in seen if r not in man and r != 'data/proc/splits.json'))
    if new:
        log(f'  {len(new)} unpacked files are not in MANIFEST.json yet (e.g. {new[0]}); check them, then --write-manifest')
    return new


def ensure_splits():
    """The data root's proc/splits.json is the tracked frozen file (copied when the root is elsewhere)."""
    sp = PROC / 'splits.json'
    if sp.resolve() != FROZEN.resolve() and (not sp.exists() or sha256(sp) != SPLITS_FILE_SHA):
        sp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FROZEN, sp)
        log(f'splits.json copied from the repository into {sp.parent}')


# ---------------------------------------------------------------- stages
def stage_fetch():
    for rel, url in FETCH.items():
        download(url, RAW / rel)
    download('https://api.figshare.com/v2/articles/29297999', DATA / 'tmp' / 'fs_29297999.json')
    (RAW / 'faults' / 'ghalamsiah2025').mkdir(parents=True, exist_ok=True)
    missing = {k: v for k, v in MANUAL.items() if not (DATA / k).exists()}
    for k, v in missing.items():
        log(f'  manual download needed: data/{k}  ({v})')
    return {'manual_missing': missing}


def stage_bdg2():
    return run([SCRIPTS / 'fetch_bdg2.sh']) or steps(('build_bdg2.py', 'meters', 'summary'))


def stage_weather():
    return steps(('weather_fetch.py', 'screen'), ('build_bdg2.py', 'points'), ('weather_fetch.py', 'power'),
                 ('weather_fetch.py', 'asos'), ('build_bdg2.py', 'weather'))


def stage_stock():
    return steps(('stock_meta.py', 'resstock'), ('stock_meta.py', 'comstock'), ('build_bdg2.py', 'spec'),
                 ('stock_ts.py', 'sample'), ('stock_ts.py', 'fetch', 'resstock'), ('stock_ts.py', 'fetch', 'comstock'),
                 ('stock_ts.py', 'weather'))


def stage_eia():
    return steps(('build_recs_cbecs.py',))


def stage_faults():
    return steps(('fetch_ghal.py',), ('build_faults.py',))


def stage_tmy():
    return steps(('fetch_tmy.py',))


def stage_tariffs():
    return steps(('build_tariffs.py',))


def stage_other():
    return steps(('build_other.py',))


def stage_splits():
    """make_splits.py in a scratch root (it refuses to run where splits.json exists, and it rewrites
    faults/runs.parquet); its stored sha256 must equal the frozen file's."""
    w = WORK / 'splits_rebuild'
    need = ['bdg2/buildings.parquet', 'eia/recs2020.parquet', 'eia/cbecs2018.parquet', 'faults/runs.parquet']
    miss = [n for n in need if not (PROC / n).exists()]
    if miss:
        log(f'splits: inputs missing ({", ".join(miss)}); not run')
        return {'skipped': miss}
    shutil.rmtree(w, ignore_errors=True)
    for n in need:
        (w / 'data' / 'proc' / n).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROC / n, w / 'data' / 'proc' / n)
    env = dict(os.environ, ML_RATCHET_BUILDING=str(w), OMP_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
    rc = subprocess.run([PY, '-I', str(SCRIPTS / 'make_splits.py')], env=env, cwd=str(w)).returncode
    if rc:
        return {'rc': rc}
    raw = (w / 'data' / 'proc' / 'splits.json').read_bytes()
    stored, _ = splits_field_ok(raw)
    frozen, _ = splits_field_ok(FROZEN.read_bytes())
    log(f'splits: rebuilt sha256 field {stored[:12]}... frozen {frozen[:12]}... '
        + ('MATCH' if stored == frozen else 'DIFFERENT (the frozen file stays)'))
    return {'match': stored == frozen, 'rebuilt': stored}


def stage_baselines():
    return steps(('baselines.py', 'load'), ('baselines.py', 'detective'), ('baselines.py', 'faults'),
                 ('baselines.py', 'thermostat'), ('baselines.py', 'summary'))


def stage_priors():
    return run([CODE / 'buildingsim' / 'tools' / 'build_priors.py'])


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
    for k, v in man.get('pending', {}).items():
        log(f'  pending (not shipped yet): {k} {"present" if (ROOT / k).exists() else "absent"} - {v}')
    sp = PROC / 'splits.json'
    ok_file = sp.exists() and sha256(sp) == SPLITS_FILE_SHA
    stored, recomputed = splits_field_ok(sp.read_bytes()) if sp.exists() else (None, None)
    ok_field = bool(stored) and stored == recomputed and stored.startswith(SPLITS_FIELD_SHA)
    log(f'splits.json: file sha256 {"OK" if ok_file else "WRONG"}; sha256 field {"OK" if ok_field else "WRONG"} '
        f'({(stored or "none")[:12]}..., recomputed {(recomputed or "none")[:12]}...)')
    rep = {'checked': time.strftime('%Y-%m-%d %H:%M'), 'root': str(ROOT), 'identical': len(same), 'differ': diff,
           'missing': missing, 'splits_ok': ok_file and ok_field}
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'verify.json').write_text(json.dumps(rep, indent=1))
    smoke = ('import sys; sys.path.insert(0, sys.argv[1]); import buildingdata as bd; '
             'print("loader:", len(bd.recs(split="dev")), "RECS dev,", len(bd.cbecs(split="dev")), "CBECS dev,", '
             'len(bd.faults(split="exam")), "exam fault runs,", len(bd.tariffs()), "tariffs,", '
             'len(bd.bdg2_meta(split="dev")), "BDG2 dev buildings,", len(bd.other("list")), "other datasets")')
    if not any(m.startswith('data/proc/') for m in missing):
        r = subprocess.run([PY, '-I', '-c', smoke, str(CODE)], env=dict(os.environ, ML_RATCHET_BUILDING=str(ROOT)))
        if r.returncode:
            return 1
    return 0 if ok_file and ok_field and not missing and not diff else 1


def write_manifest():
    """Re-hash every data file under the root (data/proc, buildingsim/refdata, checks/out twin results)."""
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    files = {}
    for pre in DATA_PREFIXES:
        base = ROOT / pre
        for p in sorted(base.rglob('*')) if base.exists() else []:
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and not rel.endswith('.part') and rel != 'data/proc/splits.json':
                files[rel] = [sha256(p), p.stat().st_size]
    files['data/proc/splits.json'] = [SPLITS_FILE_SHA, FROZEN.stat().st_size]
    files = dict(sorted(files.items()))
    old.update(n_files=len(files), bytes=sum(v[1] for v in files.values()), files=files)
    pend = old.get('pending', {})
    for k in list(pend):
        if any(f.startswith(k) for f in files):
            pend.pop(k)
    MANIFEST.write_text(json.dumps(old, indent=1) + '\n')
    log(f'MANIFEST.json: {len(files)} files, {old["bytes"] / 1e6:.1f} MB')


FUNCS = {s: globals()[f'stage_{s}'] for s in STAGES}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--only', help='comma-separated stages')
    ap.add_argument('--from', dest='start', help='start at this stage')
    ap.add_argument('--from-zip', type=Path, nargs='+', help='unpack shipped zips instead of downloading, then verify')
    ap.add_argument('--offline', action='store_true', help='skip every stage that downloads (' + ' '.join(DOWNLOADS) + ')')
    ap.add_argument('--write-manifest', action='store_true', help='re-hash the data on disk into data/MANIFEST.json')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args(argv)
    if a.list:
        print(__doc__)
        return 0
    log(f'building data root: {ROOT}')
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
