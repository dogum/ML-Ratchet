"""GridData rebuild: download the raw sources, run data/scripts/ in order, verify against the shipped v1.0 hashes.

usage: python -I shared/grid/data/build.py [--only a,b] [--from stage] [--offline] [--list] [--from-zip PATH]
       (or `mlratchet data grid ...`)

Two routes to the same data root ($ML_RATCHET_GRID, default shared/grid):
  --from-zip PATH   unpack data/proc/ and gridsim/refdata/ from the v1.0 release zip (gridsim-griddata-v1.0.zip,
                    245 MB) and verify every file against data/MANIFEST.json. Minutes; exact.
  (default)         fetch the raw sources from their public hosts and rebuild proc/ with data/scripts/. Hours
                    (CAMPD: 8 state files of ~0.8 GB; NESO: one SQL query per month and per event window; ERCOT and
                    CAISO price archives; NASA POWER at ~120 points). Resumable: every download is skipped when its file
                    exists, and every script skips the pieces it already wrote.

Stages (in order):
  fetch      raw downloads with no script of their own: PUDL tables, RTS-GMLC gen.csv, MATPOWER cases, Census
             Gazetteer places, NESO package lists, ERCOT monthly Operations Reports (data/sources/ercot_ops_urls.tsv),
             gridsim/refdata (Henry Hub, EIA-923 receipts and heat rates, MATPOWER)
  eia930     scripts/eia930_fetch.py, build_eia930.py         EIA-930 six-month files -> proc/eia930/
  campd      scripts/campd_fetch.py, campd_units.py           CAMPD hourly Texas 2018-2025 -> proc/campd/
  fleet      scripts/build_fleet.py                           proc/fleet/
  neso       scripts/neso_fetch.py inertia|genmix|scan        raw/neso/ (CKAN datastore SQL API)
  gb_events  scripts/gb_events.py detect windows attribute interconnectors build; neso_months.py
  ercot      scripts/ercot_events.py                          proc/frequency/ercot_events.parquet
  prices     scripts/ercot_prices.py, caiso_prices.py         raw/ercot, raw/caiso
  weather    scripts/weather.py points power asos build       proc/weather/
  networks   scripts/networks.py                              proc/networks/
  finalize   scripts/finalize.py                              GB frequency + CAISO tables
  splits     scripts/make_splits.py must reproduce the frozen split (sha256_of_content 895b4e9a...); the shipped
             proc/splits.json is restored byte for byte afterwards either way
  baselines  scripts/baselines.py demand heatrate battery frequency; qc.py
  verify     every file vs data/MANIFEST.json (sha256), splits content hash, a loader smoke test

The verify stage reports files that differ instead of failing on them: a rebuild from live hosts is not byte-identical
(EIA revises EIA-930; NESO and ERCOT files are re-issued). Only the split file must match exactly.
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

CODE = Path(__file__).resolve().parents[1]                     # shared/grid (code, scripts, manifest)
ROOT = Path(os.environ.get('ML_RATCHET_GRID') or CODE)          # data root: data/, gridsim/refdata/, checks/out/
DATA = ROOT / 'data'
RAW = DATA / 'raw'
PROC = DATA / 'proc'
REFDATA = ROOT / 'gridsim' / 'refdata'
SCRIPTS = CODE / 'data' / 'scripts'
WORK = DATA / 'work'                                             # git-ignored
STATE = WORK / 'build_state.json'
MANIFEST = CODE / 'data' / 'MANIFEST.json'
SPLITS_CONTENT_SHA = '895b4e9ab0b3e78a31916a2de557658f236736661800f0f9665b735314b917eb'
SPLITS_FILE_SHA = 'f8787bc2aac0ee13cad980fe11e7ec7a75f07cdc3b446e564437d37f119440ef'
UA = 'mlratchet-data-rebuild/1.0 (research data rebuild)'
PY = sys.executable
STAGES = ['fetch', 'eia930', 'campd', 'fleet', 'neso', 'gb_events', 'ercot', 'prices', 'weather', 'networks',
          'finalize', 'splits', 'baselines', 'verify']

# ---------------------------------------------------------------- sources (DATA.md section 2)
PUDL = 'https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/stable/{t}.parquet'
PUDL_TABLES = ['out_eia__yearly_generators', 'out_eia__yearly_plants', 'core_eia860__scd_generators_wind',
               'core_eia860__scd_generators_solar', 'core_eia860__scd_generators_energy_storage',
               'out_eia923__yearly_generation_fuel_by_generator', 'out_eia923__monthly_fuel_receipts_costs',
               'out_eia861__yearly_balancing_authority_service_territory', 'out_censusdp1tract__counties',
               'core_epa__assn_eia_epacamd']
RTS_GEN = 'https://raw.githubusercontent.com/GridMod/RTS-GMLC/master/RTS_Data/SourceData/gen.csv'
MATPOWER = 'https://raw.githubusercontent.com/MATPOWER/matpower/master/data/{c}.m'
MATPOWER_CASES = ['case9', 'case14', 'case30', 'case39', 'case57', 'case118', 'case300', 'case_RTS_GMLC',
                  'case_ACTIVSg200', 'case_ACTIVSg500', 'case_ACTIVSg2000']
REF_MATPOWER = ['case9', 'case14', 'case30', 'case57', 'case118', 'case300', 'case_ACTIVSg200', 'case_ACTIVSg2000']
GAZ = 'https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_gaz_place_{fips}.txt'
GAZ_STATES = {'tx': '48', 'il': '17', 'sc': '45'}
NESO_PKG = 'https://api.neso.energy/api/3/action/package_show?id={p}'
NESO_FILES = {'pkg.json': 'system-frequency-data', 'system-inertia.json': 'system-inertia'}
HENRY_HUB_XLS = 'https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls'


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


BLOCKED: dict[str, str] = {}


def download(url: str, dest: Path, tries: int = 3, pause: float = 1.0) -> bool:
    """Resumable-by-file download: skip if dest exists; write to .part, then rename. Records failures by host."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    host = url.split('/')[2]
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=300) as r, open(part, 'wb') as f:
                shutil.copyfileobj(r, f, 1 << 20)
            part.replace(dest)
            time.sleep(pause)
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


def run(args: list[str], env_extra: dict | None = None) -> int:
    """Run one data script with the grid root in the environment, isolated (-I), single thread."""
    env = dict(os.environ, ML_RATCHET_GRID=str(ROOT), OMP_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
    env.update(env_extra or {})
    log('  run', ' '.join(str(a) for a in args))
    return subprocess.run([PY, '-I', *map(str, args)], env=env, cwd=str(DATA)).returncode


def script(name, *argv):
    return run([SCRIPTS / name, *argv])


# ---------------------------------------------------------------- stages
def stage_unpack(zpath: Path):
    """Unpack data/proc/** and gridsim/refdata/** from the v1.0 release zip; keep files already identical."""
    man = json.loads(MANIFEST.read_text())['files']
    n_new = n_same = 0
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        top = names[0].split('/')[0] + '/' if all(n.startswith(names[0].split('/')[0] + '/') for n in names) else ''
        for n in names:
            rel = n[len(top):]
            if n.endswith('/') or rel not in man:
                continue
            dest = ROOT / rel
            if dest.exists() and dest.stat().st_size == man[rel][1] and sha256(dest) == man[rel][0]:
                n_same += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(dest, 'wb') as f:
                shutil.copyfileobj(src, f, 1 << 20)
            n_new += 1
    log(f'unpack: {n_new} files written, {n_same} already identical')


def stage_fetch():
    for t in PUDL_TABLES:
        download(PUDL.format(t=t), RAW / 'pudl' / f'{t}.parquet')
    download(RTS_GEN, RAW / 'rts_gmlc' / 'gen.csv')
    for c in MATPOWER_CASES:
        download(MATPOWER.format(c=c), RAW / 'matpower' / f'{c}.m')
    for st, fips in GAZ_STATES.items():
        download(GAZ.format(fips=fips), RAW / 'census' / f'gaz_places_{st}.txt')
    for fn, pkg in NESO_FILES.items():
        download(NESO_PKG.format(p=pkg), RAW / 'neso' / fn)
    # ERCOT monthly Operations Reports (public docx; the list is the one the shipped event table was built from)
    urls = (CODE / 'data' / 'sources' / 'ercot_ops_urls.tsv').read_text().split('\n')
    od = RAW / 'ercot_ops'
    od.mkdir(parents=True, exist_ok=True)
    (od / 'urls.tsv').write_text('\n'.join(u for u in urls if u.strip()) + '\n')
    for line in urls:
        if line.strip():
            m, u = line.split('\t')
            download(u, od / f'{m}.docx', pause=8.0)     # gently, as the original fetch
    # gridsim reference tables
    for c in REF_MATPOWER:
        src = RAW / 'matpower' / f'{c}.m'
        if src.exists():
            (REFDATA / 'matpower').mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, REFDATA / 'matpower' / f'{c}.m')
    for t in ('out_eia923__monthly_fuel_receipts_costs',):
        if (RAW / 'pudl' / f'{t}.parquet').exists():
            (REFDATA / 'fuel').mkdir(parents=True, exist_ok=True)
            shutil.copy2(RAW / 'pudl' / f'{t}.parquet', REFDATA / 'fuel' / f'{t}.parquet')
    src = RAW / 'pudl' / 'out_eia923__yearly_generation_fuel_by_generator.parquet'
    if src.exists():
        (REFDATA / 'hr').mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, REFDATA / 'hr' / src.name)
        run([CODE / 'gridsim' / 'tools' / 'build_heat_rates.py'])
    if download(HENRY_HUB_XLS, REFDATA / 'fuel' / 'RNGWHHDd.xls'):
        henry_hub_csv()


def henry_hub_csv():
    """refdata/fuel/henry_hub_daily.csv from EIA's daily Henry Hub spot price workbook (sheet 'Data 1')."""
    out = REFDATA / 'fuel' / 'henry_hub_daily.csv'
    if out.exists():
        return
    try:
        import pandas as pd
        d = pd.read_excel(REFDATA / 'fuel' / 'RNGWHHDd.xls', sheet_name='Data 1', skiprows=2)
        d.columns = ['date', 'hh_usd_per_MMBtu']
        d = d.dropna()
        d['date'] = pd.to_datetime(d['date']).dt.strftime('%Y-%m-%d')
        d.to_csv(out, index=False, lineterminator='\n')
    except Exception as e:  # reading .xls needs xlrd
        log('  henry hub csv not built:', e)


def stage_eia930():
    script('eia930_fetch.py') or script('build_eia930.py')


def stage_campd():
    script('campd_fetch.py') or script('campd_units.py')


def stage_fleet():
    script('build_fleet.py')


def stage_neso():
    for s in ('inertia', 'genmix', 'scan'):
        script('neso_fetch.py', s)


def stage_gb_events():
    script('gb_events.py', 'detect', 'windows', 'attribute', 'interconnectors', 'build')
    script('neso_months.py')


def stage_ercot():
    script('ercot_events.py')


def stage_prices():
    script('ercot_prices.py')
    script('caiso_prices.py')


def stage_weather():
    for s in ('points', 'power', 'asos', 'build'):
        script('weather.py', s)


def stage_networks():
    script('networks.py')


def stage_finalize():
    script('finalize.py')


def stage_splits():
    """make_splits.py must reproduce the frozen split; the shipped file is put back byte for byte either way."""
    sp = PROC / 'splits.json'
    frozen = CODE / 'data' / 'proc' / 'splits.json'
    keep = sp.read_bytes() if sp.exists() else frozen.read_bytes()
    try:
        script('make_splits.py')
        new = json.loads(sp.read_text())['sha256_of_content']
        log(f'splits: rebuilt content sha256 {new[:12]}... frozen {SPLITS_CONTENT_SHA[:12]}... '
            + ('MATCH' if new == SPLITS_CONTENT_SHA else 'DIFFERENT (the frozen file stays)'))
    finally:
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_bytes(keep)


def stage_baselines():
    script('baselines.py', 'demand', 'heatrate', 'battery', 'frequency')
    script('qc.py')


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
    sp = PROC / 'splits.json'
    ok_split = sp.exists() and sha256(sp) == SPLITS_FILE_SHA
    content = json.loads(sp.read_text()).get('sha256_of_content') if sp.exists() else None
    log(f'verify: {len(same)} of {len(man["files"])} shipped files identical, {len(diff)} differ, {len(missing)} missing')
    for tag, lst in (('differ', diff), ('missing', missing)):
        for r in lst[:15]:
            log(f'  {tag}: {r}')
        if len(lst) > 15:
            log(f'  ... {len(lst) - 15} more {tag}')
    log(f'splits.json: file sha256 {"OK" if ok_split else "WRONG"}; content sha256 '
        f'{"OK" if content == SPLITS_CONTENT_SHA else "WRONG"} ({(content or "none")[:12]}...)')
    rep = {'checked': time.strftime('%Y-%m-%d %H:%M'), 'root': str(ROOT), 'identical': len(same), 'differ': diff,
           'missing': missing, 'splits_ok': ok_split}
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'verify.json').write_text(json.dumps(rep, indent=1))
    if not missing:   # loader smoke test
        env = dict(os.environ, ML_RATCHET_GRID=str(ROOT))
        r = subprocess.run([PY, '-I', '-c', 'import sys; sys.path.insert(0, sys.argv[1]); import griddata as gd; '
                            'print("loader:", len(gd.events("GB", split="dev")), "GB dev events,", '
                            'len(gd.eia930("ERCO", split="dev")), "ERCO dev hours")', str(CODE)], env=env)
        if r.returncode:
            return 1
    return 0 if ok_split and not missing else 1


FUNCS = {s: globals()[f'stage_{s}'] for s in STAGES}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--only', help='comma-separated stages')
    ap.add_argument('--from', dest='start', help='start at this stage')
    ap.add_argument('--from-zip', type=Path, help='unpack the v1.0 release zip instead of downloading, then verify')
    ap.add_argument('--offline', action='store_true', help='skip every stage that downloads (fetch eia930 campd neso '
                    'gb_events prices weather)')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args(argv)
    if a.list:
        print(__doc__)
        return 0
    log(f'grid root: {ROOT}')
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
        stages = [s for s in stages if s not in ('fetch', 'eia930', 'campd', 'neso', 'gb_events', 'prices', 'weather')]
    st = load_state()
    rc = 0
    for s in stages:
        log(f'== {s}')
        t = time.time()
        r = FUNCS[s]()
        st[s] = {'done': time.strftime('%Y-%m-%d %H:%M'), 'seconds': round(time.time() - t)}
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
