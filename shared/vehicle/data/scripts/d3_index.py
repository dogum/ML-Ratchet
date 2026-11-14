"""Argonne D3: parse every downloaded test file into the shared trip contract and one index row per test.

Reads data/raw/d3/ (filled by d3_fetch.py get; the 2020 Tesla Model 3 is not downloaded) and writes
    data/proc/d3/index.parquet           one row per test (vehicle, cycle, ambient, start, totals, signals, file)
    data/proc/d3/channels.json           which raw channel fed each signal, per vehicle and era; summary-table rows
    data/proc/d3/tests/<veh>/<test>.npz  1 Hz contract arrays + *_10hz originals; <test>.json the meta
(tests/ is git-ignored: it is the raw data resampled.)

Three file eras (see data/D3.md):
    zip   "All Data" zip of <id>_Data.txt / <id> Test Data.txt, MY2009-2016: 10 Hz, tab separated, one header row
          "Name [unit]"; 10-12 channels; the fuel channel is the emissions bench's modal carbon balance [g/s] (except
          the Bi-Fuel Impala: fuel scale [cc/s] and CNG Coriolis [g/s]). Test list in a PDF ("Master Summary List").
    txt   "Comprehensive/Extended Dataset(s)/<id> Test Data.txt", MY2013-2022: 10 Hz, tab separated, one header row
          "Name__unit", 50-400 channels incl. vehicle CAN; test list in an xlsx summary.
    tdms  "Extended Datasets/<id> Analysis.tdms", MY2017-2023: NI TDMS (read by d3_tdms.py), units in channel
          properties, test info (cold/hot, cycle file, comments, dyno targets, fuel properties) in group properties.

Fuel rate [g/s]: the measured fuel flow when the file has it (direct mass flow meter [g/s], else the volumetric
meter [cc/s] x the test fuel density), otherwise the emissions bench's modal fuel by carbon balance [g/s]
(index column fuel_source says which). Both totals are kept when both exist (fuel_g_direct, fuel_g_modal).

Battery current sign: + = discharge (the contract). Each test's sign is checked against tractive power during
accelerations and flipped if needed (batt_sign_flipped).

usage:  python d3_index.py [--workers 12] [--vehicles "Camry,Bolt"] [--force]
"""
import os, re, sys, json, glob, zipfile, argparse, subprocess, io, time, traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from d3_tdms import read_tdms
from d3_fetch import VEHICLES, RAW

_VEH = os.path.abspath(os.path.join(HERE, '..', '..'))
PROC = os.path.join(_VEH, 'data', 'proc', 'd3')
TESTS = os.path.join(PROC, 'tests')
MPH = 0.44704
RHO_REF = 1.1767            # reference air density for road load from target ABC (as in checks/epa_procedures.py)
SKIP = {'2020 Tesla Model 3'}


def vehicle_dir(name):
    return re.sub(r'[^\w.-]+', '_', name)


def vehicle_id(name):
    return 'D3_' + vehicle_dir(name)


# ------------------------------------------------------------------------------------------------ channel names
UNITS = {'s', 'sec', 'mph', 'kph', 'kmh', 'km/h', 'n', 'mi', 'c', 'degc', 'deg c', 'f', 'degf', 'k', 'per', '%', 'pct',
         'inhg', 'psi', 'kpa', 'mpa', 'hpa', 'pa', 'gps', 'g/s', 'mg/s', 'ccps', 'cc/s', 'ml/s', 'v', 'a', 'kw', 'w',
         'wh', 'kwh', 'ah', 'rpm', 'nm', 'ohm', 'lpm', 'nlpm', 'deg', 'us', 'ms', 'km', 'm', 'g', 'kg', 'l', 'gal'}


def split_name(raw, unit_prop=None):
    """raw channel name -> (normalized base, unit). Handles 'Name [unit]', 'Name[unit]', 'Name__unit(__x)', TDMS."""
    s = raw.strip()
    unit = None
    m = re.match(r'^(.*?)\s*\[([^\]]*)\]\s*$', s)
    if m:
        s, unit = m.group(1), m.group(2)
    elif '__' in s:
        toks = s.split('__')
        s = toks[0]
        for t in toks[1:][::-1]:
            u = t.split('_')[0]
            if u.lower() in UNITS:
                unit = u; break
        if len(toks) > 2 and unit is not None and toks[1].split('_')[0].lower() not in UNITS:
            s = s + '_' + toks[1]                     # e.g. HVBatt_Hioki_IH1 or Volt..._analog10hz__U1__V
    if unit_prop:
        unit = unit_prop
    base = re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')
    return base, (unit or '').strip()


# canonical signal -> (ordered regexes on the normalized base name, excluded words, required unit kind or None)
SIGNALS = {
    't':            ([r'^time$', r'^timestamp$', r'^time_s_rawfacilities$'], [], None),
    'v':            ([r'^dyno_spd$', r'^dyno_speed$'], [], 'speed'),
    'v_sched':      ([r'^drive_trace_schedule$', r'^drive_schedule_speed$'], [], 'speed'),
    'dist':         ([r'^distance$'], [], None),
    'F_tractive':   ([r'^dyno_tractiveforce$', r'^dyno_tractive_force$', r'^dyno_tractive_effort$'], [], None),
    'T_cell':       ([r'^cell_temp$', r'^test_cell_temp$'], [], 'temp'),
    'RH_cell':      ([r'^cell_rh$', r'^test_cell_rh$'], [], None),
    'p_cell':       ([r'^cell_press$', r'^test_cell_baro'], [], None),
    'bag':          ([r'^exhaust_bag$', r'^phase$', r'^phase_number$'], [], None),
    'grade':        ([r'^grade_output_drivetrace$', r'^drive_trace_grade$'], [], None),
    'eng_rpm':      ([r'^engine_speed$', r'^eng_spd$', r'^eng_speed$', r'^eng_spd_can$', r'^engine_speed_can$',
                      r'^eng_spd_(pcm|ecm|ecu)$', r'^eng(ine)?_(spd|speed|rpm)', r'eng(ine)?_(spd|speed|rpm)',
                      r'engine_revolution'],
                     ['target', 'idle', 'desired', 'request', 'cmd', 'limit', 'max', 'min', 'fan', 'pump', 'command'], 'rpm'),
    'fuel_mass':    ([r'^eng_fuelflow_direct2$', r'^eng_fuelflow_direct\d*$', r'fuel_?flow_?direct', r'fuel_?flow.*(meter|scale|coriolis)',
                      r'^fuel_?flow', r'fuel_?flow'],
                     ['modal', 'dilute', 'bench', 'cng', 'gaseous', 'temp', 'press', 'target', 'injector', 'h2'], 'massflow'),
    'fuel_vol':     ([r'^eng_fuelflow_direct\d*$', r'fuel_?flow_?direct', r'fuel_?flow.*(meter|scale)', r'^fuel_?flow', r'fuel_?flow'],
                     ['modal', 'dilute', 'bench', 'cng', 'gaseous', 'temp', 'press', 'target', 'injector', 'h2'], 'volflow'),
    'fuel_vol_pfi': ([r'fuelflow.*_pfi$', r'fuel_?flow.*pfi'], [], 'volflow'),
    'fuel_vol_di':  ([r'fuelflow.*_di$', r'fuel_?flow.*_di(_|$)'], [], 'volflow'),
    'fuel_modal':   ([r'^ama_dilute_fuel$', r'fuel_flow_from_bench_modal', r'bench_modal', r'modal.*fuel', r'fuel.*modal'], [], 'massflow'),
    'fuel_modal_vol': ([r'fuel_flow_from_bench_modal', r'bench_modal', r'modal.*fuel', r'fuel.*modal'], [], 'volflow'),
    'fuel_gas':     ([r'cng_coriolis', r'gaseousfuel_high_flow', r'gaseousfuel.*flow', r'cng.*flow', r'h2.*flow'], [], 'massflow'),
    'batt_V':       ([r'^hvbatt_volt_hioki', r'^hv_battery_voltage$', r'^hvbatt_volt', r'^hv_?batt(ery)?_?volt',
                      r'hv_?batt(ery)?_?volt', r'^ess_volt', r'traction_batt.*volt', r'hvbatt_hioki_u'],
                     ['block', 'cell', 'min', 'max', 'limit', '12v', 'module', 'target', 'before', 'after', 'boost'], 'volt'),
    'batt_I':       ([r'^hvbatt_curr_hioki', r'^hv_battery_current$', r'^hvbatt_curr', r'^hv_?batt(ery)?_?curr',
                      r'hv_?batt(ery)?_?curr', r'^ess_curr', r'traction_batt.*curr'],
                     ['limit', 'max', 'min', 'integrat', 'ih', '12v', 'target', 'cell'], 'amp'),
    'batt_P':       ([r'^hvbatt_power_hioki', r'^hv_battery_power$', r'^hvbatt_power$'], ['limit', 'max', 'min', 'win', 'wout'], 'power'),
    'soc':          ([r'^hvbatt_soc', r'^hv_battery_soc', r'^hvbatt.*soc', r'^soc', r'_soc(_|$)', r'state_of_charge'],
                     ['12v', 'target', 'min', 'max', 'limit', 'display'], None),
    'T_coolant':    ([r'^engine_coolant_temp$', r'^eng_coolant_temp(_can|_pcm|_ecm)?$', r'^eng.*coolant_temp', r'engine.*coolant.*temp',
                      r'^coolant_temp'],
                     ['hvac', 'radiator', 'fc', 'motor', 'inverter', 'batt', 'target', 'outlet', 'inlet', 'heater'], 'temp'),
    'T_oil':        ([r'^engine_oil_temp$', r'^eng_oil_temp(_can|_pcm|_ecm)?$', r'^eng.*oil_temp', r'engine.*oil.*temp'],
                     ['trans', 'vvt', 'gear', 'diff', 'atf'], 'temp'),
    'T_trans_oil':  ([r'^trans.*oil_temp', r'trans.*fluid.*temp', r'^atf.*temp', r'trans.*oil.*temp'], [], 'temp'),
    'T_cabin':      ([r'^cabin_temp$', r'^cabin_air_temp', r'hvac_room_temp', r'interior_temp', r'cabin.*temp'],
                     ['vent', 'set', 'target', 'lower', 'upper', 'outlet', 'duct'], 'temp'),
    'T_batt':       ([r'^hvbatt_temp', r'hvbatt.*temp(erature)?(_avg|_mean)?', r'hv_?batt.*temp', r'battery.*temp', r'ess.*temp'],
                     ['inhal', 'inlet', 'air', '12v', 'limit', 'max', 'min', 'coolant', 'heater'], 'temp'),
    'P_heater':     ([r'heater.*power', r'ptc.*power', r'heater_power'], ['seat', 'batt', 'limit', 'request', 'water_heater_req'], 'power'),
    'P_batt_heater': ([r'batt.*heater.*power', r'ess.*heater.*power'], ['12v', 'limit', 'request'], 'power'),
    'P_ac':         ([r'^hvac_ac_power', r'a_?c_comp.*power', r'(ac|aircon|hvac|e_?comp).*comp.*power', r'compressor.*power'],
                     ['limit', 'request', 'target', 'max'], 'power'),
    'P_dcdc':       ([r'^dcdc.*power', r'^alternator.*power', r'dcdc.*power', r'alternator.*power'], ['limit'], 'power'),
    'T_solar_ind':  ([r'^solar_array'], [], 'temp'),
    'v_veh_can':    ([r'^veh_spd_can$', r'^veh_speed_can$', r'^veh_spd', r'^vehicle_speed'], [], 'speed'),
}

UNIT_KIND = {'speed': {'mph', 'kph', 'kmh', 'km/h'}, 'temp': {'c', 'degc', 'deg c', 'f', 'degf', 'k', ''},
             'massflow': {'gps', 'g/s', 'mg/s'}, 'volflow': {'ccps', 'cc/s', 'ml/s', 'mlps'}, 'volt': {'v', ''},
             'amp': {'a', ''}, 'power': {'w', 'kw'}, 'rpm': {'rpm', ''}}


def pick_channels(names, units):
    """names: raw channel names, units: {raw: unit} -> {signal: raw name}."""
    info = [(n,) + split_name(n, units.get(n)) for n in names]
    out = {}
    for sig, (pats, excl, kind) in SIGNALS.items():
        found = None
        for pat in pats:
            for raw, base, u in info:
                if not re.search(pat, base):
                    continue
                if any(x in base for x in excl):
                    continue
                if kind is not None and u.lower() not in UNIT_KIND[kind]:
                    if not (kind == 'massflow' and u == '' and 'modal' in base):
                        continue
                if sig == 'fuel_mass' and u.lower() in UNIT_KIND['volflow']:
                    continue
                found = raw; break
            if found:
                break
        if found:
            out[sig] = found
    return out


def convert(sig, x, unit, density=0.7433):
    """to SI / contract units."""
    u = (unit or '').lower()
    x = x.astype(float)
    if sig in ('v', 'v_sched', 'v_veh_can'):
        return x * (MPH if u == 'mph' else 1 / 3.6 if u in ('kph', 'kmh', 'km/h') else 1.0)
    if sig.startswith('T_'):
        if u in ('f', 'degf'): return (x - 32) / 1.8
        if u == 'k': return x - 273.15
        return x
    if sig in ('fuel_mass', 'fuel_modal', 'fuel_gas'):
        return x / 1000 if u == 'mg/s' else x
    if sig.startswith('fuel_vol') or sig == 'fuel_modal_vol':
        return x * density                       # cc/s x g/ml -> g/s
    if sig in ('P_heater', 'P_ac', 'P_dcdc', 'batt_P'):
        return x * 1000 if u == 'kw' else x
    if sig == 'p_cell':
        return x * 3386.39 if u == 'inhg' else x * 1000 if u == 'kpa' else x * 100 if u == 'hpa' else x
    if sig == 'dist':
        return x * 1609.344 if u == 'mi' else x * 1000 if u == 'km' else x
    if sig == 'grade':
        return x / 100.0                          # pct -> rise/run
    return x


# ------------------------------------------------------------------------------------------------ discovery
def discover(vname):
    """-> list of sources {num, era, kind, path, member} for one vehicle (deduplicated, extended files preferred)."""
    vd = os.path.join(RAW, vehicle_dir(vname))
    found = {}
    rank = {'tdms': 3, 'txt': 2, 'zip': 1}
    for root, _, files in os.walk(vd):
        for f in files:
            fp = os.path.join(root, f)
            if f.endswith('.part'):
                continue
            src = []
            if f.lower().endswith('.tdms'):
                m = re.match(r'(\d{8})', f)
                if m: src.append(dict(num=m.group(1), era='tdms', path=fp, member=None))
            elif f.lower().endswith('.txt'):
                m = re.match(r'(\d{8})', f)
                if m: src.append(dict(num=m.group(1), era='txt', path=fp, member=None))
            elif f.lower().endswith('.zip'):
                try:
                    Z = zipfile.ZipFile(fp)
                except zipfile.BadZipFile:
                    continue
                for i in Z.infolist():
                    b = os.path.basename(i.filename)
                    m = re.match(r'(\d{8})', b)
                    if m and b.lower().endswith('.txt'):
                        src.append(dict(num=m.group(1), era='zip', path=fp, member=i.filename))
            for s in src:
                k = s['num']
                if k not in found or rank[s['era']] > rank[found[k]['era']]:
                    found[k] = s
    return sorted(found.values(), key=lambda s: s['num'])


# ------------------------------------------------------------------------------------------------ summaries
def _num(x):
    try:
        v = float(str(x).strip().replace(',', ''))
        return v if np.isfinite(v) else None
    except Exception:
        return None


def summary_xlsx(path):
    """D3 test summary workbook -> {test num: {...}}."""
    import openpyxl
    out = {}
    try:
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    except Exception as e:
        return {'_error': str(e)}
    veh = {}
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        hdr_i = None
        for i, r in enumerate(rows[:60]):
            cells = [str(c).strip() for c in r if c is not None]
            if cells and cells[0].lower().startswith('test id'):
                hdr_i = i; break
            if len(cells) >= 2 and cells[0].lower() in ('test weight [lb]', 'test weight', 'dyno target a:', 'target a [lb]'):
                veh[cells[0].lower()] = _num(cells[1])
        if hdr_i is None:
            continue
        hdr = [str(c).strip() if c is not None else '' for c in rows[hdr_i]]
        col = lambda *keys: next((j for j, h in enumerate(hdr) if all(k in h.lower() for k in keys)), None)
        jc = {'cycle': col('cycle') if col('cycle') != col('cycle', 'distance') else None,
              'start': col('start comment'), 'end': col('end comment'), 'temp': col('test cell temp'),
              'weight': col('test weight'), 'A': col('target a'), 'B': col('target b'), 'C': col('target c'),
              'dist': col('distance'), 'fuel_g': next((j for j, h in enumerate(hdr) if 'consumed' in h.lower() and '[g]' in h.lower()), None),
              'batt_Wh': next((j for j, h in enumerate(hdr) if 'hv' in h.lower() and 'wh' in h.lower() and 'mi' not in h.lower() and 'net' in h.lower()), None),
              'time': col('time')}
        jc['cycle'] = next((j for j, h in enumerate(hdr) if h.lower() == 'cycle'), jc['cycle'])
        section = ''
        for r in rows[hdr_i + 1:]:
            cells = [c for c in r if c is not None and str(c).strip() != '']
            if not cells:
                continue
            jid = next((j for j, h in enumerate(hdr) if h.lower().startswith('test id')), 0)
            c0 = r[jid] if jid < len(r) else None
            m = re.match(r'^(\d{8})', str(int(c0) if isinstance(c0, float) else c0).strip()) if c0 is not None else None
            if not m:
                if len(cells) == 1 and isinstance(cells[0], str):
                    section = cells[0].strip()
                continue
            num = m.group(1)
            g = lambda k: (r[jc[k]] if jc.get(k) is not None and jc[k] < len(r) else None)
            cyc = str(g('cycle') or '')
            d = out.setdefault(num, dict(cycle_text=re.sub(r',?\s*Ph\s*\d.*$', '', cyc).strip(), start_comments=str(g('start') or ''),
                                         end_comments=str(g('end') or ''), section=section, temps=[], weight_lb=_num(g('weight')),
                                         A=_num(g('A')), B=_num(g('B')), C=_num(g('C')), fuel_g_phases=[], batt_Wh_phases=[], source='xlsx'))
            t = _num(g('temp'))
            if t is not None: d['temps'].append(t)
            if not re.search(r'Ph\s*\d\s*\+\s*\d', cyc):           # skip combined phase rows
                f = _num(g('fuel_g'))
                if f is not None: d['fuel_g_phases'].append(f)
                w = _num(g('batt_Wh'))
                if w is not None: d['batt_Wh_phases'].append(w)
    for d in out.values():
        d['T_set_C'] = float(np.mean(d.pop('temps'))) if d.get('temps') else None
        d['fuel_g'] = float(np.sum(d['fuel_g_phases'])) if d['fuel_g_phases'] else None
        d['batt_Wh'] = float(np.sum(d['batt_Wh_phases'])) if d['batt_Wh_phases'] else None
    return out


def pdftext(path):
    try:
        return subprocess.run(['pdftotext', '-layout', path, '-'], capture_output=True, timeout=60).stdout.decode('utf-8', 'replace')
    except Exception:
        return ''


def summary_pdf(path):
    """D3 'Master Summary List' / 'TestSummary' PDF -> {test num: {...}}, plus vehicle header (weight, ABC)."""
    txt = pdftext(path)
    out = {}
    hdr = {}
    for key, pat in (('weight_lb', r'Test weight \[lb\]\s+([\d.]+)'), ('A', r'Target A \[lb\]\s+([-\d.]+)'),
                     ('B', r'Target B \[lb/mph\]\s+([-\d.]+)'), ('C', r'Target C \[lb/mph\^?2\]\s+([-\d.]+)')):
        m = re.search(pat, txt)
        if m: hdr[key] = _num(m.group(1))
    tset = re.search(r'(-?\d+)\s*(?:°|deg)?\s*F\b', os.path.basename(path))
    for line in txt.splitlines():
        m = re.match(r'^\s*(\d{8})(?:_\d+)?\s{1,}(.+?)\s*$', line)
        if not m:
            continue
        num, rest = m.group(1), m.group(2)
        parts = re.split(r'\s{2,}', rest)
        cyc = parts[0].strip()
        start = None
        ms = re.search(r'\b(CSt|HSt|CS|HS)\b', rest)
        if ms: start = 'cold' if ms.group(1).startswith('C') else 'hot'
        d = out.setdefault(num, dict(cycle_text=cyc, start_comments='', end_comments='', section='', source='pdf',
                                     start_flag=start, pdf_line=rest[:200], **hdr))
        if start and not d.get('start_flag'): d['start_flag'] = start
        if tset and 'T_file_F' not in d: d['T_file_F'] = float(tset.group(1))
    return out


def vehicle_summaries(vname):
    vd = os.path.join(RAW, vehicle_dir(vname))
    out = {}
    for root, _, files in os.walk(vd):
        for f in sorted(files):
            fp = os.path.join(root, f)
            if f.lower().endswith(('.xlsx', '.xlsm')):
                s = summary_xlsx(fp)
            elif f.lower().endswith('.pdf') and os.path.getsize(fp) < 3e6:
                s = summary_pdf(fp)                    # summary lists; big analysis reports are skipped
            else:
                continue
            for k, v in s.items():
                if k == '_error':
                    continue
                if k not in out or (out[k]['source'] == 'pdf' and v['source'] == 'xlsx'):
                    out[k] = v
    return out


# ------------------------------------------------------------------------------------------------ reading a test
def _header_and_skip(lines):
    """-> (header fields, number of lines to skip before data)."""
    hdr, skip = None, 0
    for i, ln in enumerate(lines):
        f = ln.rstrip('\r\n').split('\t')
        nnum = sum(_num(x) is not None for x in f)
        if len(f) > 2 and nnum >= 0.8 * len(f):
            return hdr, i
        if len(f) > 2:
            hdr = f
    return hdr, len(lines)


def read_text(raw_bytes_or_path, member=None):
    if member is not None:
        with zipfile.ZipFile(raw_bytes_or_path) as Z:
            data = Z.read(member)
        fh = lambda: io.BytesIO(data)
    else:
        fh = lambda: open(raw_bytes_or_path, 'rb')
    with fh() as f:
        head = [f.readline().decode('latin1') for _ in range(8)]
    hdr, skip = _header_and_skip(head)
    names = [h.strip() for h in hdr]
    sel = pick_channels(names, {})
    use = sorted({names.index(r) for r in sel.values()})
    with fh() as f:
        df = pd.read_csv(f, sep='\t', header=None, skiprows=skip, usecols=use, encoding='latin1', engine='c',
                         low_memory=False, na_values=['NaN', 'Inf', '-Inf', 'nan', ''])
    cols = {names[j]: pd.to_numeric(df[j], errors='coerce').to_numpy(float) for j in use}
    units = {n: split_name(n)[1] for n in names}
    return names, units, cols, sel, {}


def read_tdms_test(path):
    g, pr = read_tdms(path, props=True)
    G = max(g, key=lambda k: len(g[k]))
    ch = g[G]
    names = list(ch)
    units = {c: str(pr.get((G, c), {}).get('unit_string', '') or '') for c in names}
    for c in names:                             # anonymous Hioki channels carry a wrong unit ('s')
        if re.match(r'^(U|I|P|IH|WP|PWP|MWP)\d$', c):
            units[c] = ''
    sel = pick_channels(names, units)
    gp = pr.get((G,), {})
    props = {k.replace('TestInfo_', ''): v for k, v in gp.items() if k.startswith('TestInfo_') and not k.startswith('TestInfo_J2951')}
    props.update({k: v for k, v in gp.items() if k.startswith('Phase_') and ('Fuel' in k or 'Elec_Whr_Whr' in k or 'Phase_Length' in k or 'Distance' in k)})
    return names, units, {c: np.asarray(ch[c], float) for c in sel.values()}, sel, props


# ------------------------------------------------------------------------------------------------ cycles
_REFS = None


def refs():
    """reference schedules (m/s, 1 Hz) for cycle recognition."""
    global _REFS
    if _REFS is None:
        cdir = os.path.join(_VEH, 'data', 'proc', 'cycles')       # the schedules realdata.cycles() serves
        c = {}
        for k in ('UDDS', 'LA92', 'WLTC_3b', 'HWFET', 'US06', 'SC03', 'NYCC', 'JP_10_15', 'ECE_R15', 'EUDC'):
            with np.load(os.path.join(cdir, k + '.npz')) as z:
                c[k] = z['v'].astype(float)
        R = {'LA92': c['LA92'], 'UDDS': c['UDDS'], 'WLTC': c['WLTC_3b'], 'HWFET': c['HWFET'], 'US06': c['US06'],
             'SC03': c['SC03'], 'NYCC': c['NYCC'], 'JP10_15': c['JP_10_15'],
             'NEDC': np.concatenate([c['ECE_R15'][:-1]] * 4 + [c['EUDC']]), 'UDDS_505': c['UDDS'][:506]}
        _REFS = {k: np.asarray(v, float) for k, v in R.items()}
    return _REFS


def _ssd(x, r):
    """sum of squared differences of r against every window of x (FFT)."""
    from scipy.signal import fftconvolve
    L = len(r)
    if len(x) < L:
        return np.array([])
    x2 = np.convolve(x * x, np.ones(L), 'valid')
    xr = fftconvolve(x, r[::-1], 'valid')
    return x2 - 2 * xr + np.sum(r * r)


def match_cycles(v, tol=1.0):
    """v: 1 Hz speed (m/s). Non-overlapping matches of the reference schedules with RMS error < tol m/s, longest
    schedules first (so a full UDDS wins over its first 505 s). -> list of (name, start, end, rms) sorted by start."""
    v = np.nan_to_num(np.asarray(v, float))
    taken = np.zeros(len(v), bool)
    out = []
    for name, r in sorted(refs().items(), key=lambda kv: -len(kv[1])):
        s = _ssd(v, r)
        if s.size == 0:
            continue
        rms = np.sqrt(np.maximum(s, 0) / len(r))
        idx = np.where(rms < tol)[0]
        cand = []
        for i in idx:
            lo, hi = max(0, i - 30), min(len(rms), i + 31)
            if rms[i] <= rms[lo:hi].min() + 1e-9:
                cand.append((rms[i], i))
        for e, a in sorted(cand):
            b = a + len(r)
            if taken[a:b].mean() > 0.05:
                continue
            taken[a:b] = True
            out.append((name, int(a), int(b), float(e)))
    return sorted(out, key=lambda x: x[1])


KEYWORDS = [('coast', 'coastdown'), ('wot', 'wot'), ('passing', 'passing'), ('sss', 'steady'), ('steady', 'steady'),
            ('stairs', 'steady'), ('0-80-0', 'steady'), ('mph ss', 'steady'), ('grade', 'grade'), ('idle', 'idle'),
            ('hill', 'hills'), ('sawtooth', 'ramps'), ('pedal', 'pedal_map'), ('charge', 'charge'), ('range', 'range'),
            ('nedc', 'NEDC'), ('wltc', 'WLTC'), ('la92', 'LA92'), ('la-92', 'LA92'), ('us06', 'US06'), ('sc03', 'SC03'),
            ('hwy', 'HWFET'), ('hwfet', 'HWFET'), ('highway', 'HWFET'), ('udds', 'UDDS'), ('ftp', 'UDDS'), ('nycc', 'NYCC'),
            ('jc08', 'JC08'), ('wltp', 'WLTC'), ('10-15', 'JP10_15')]


def cycle_from_text(t):
    t = (t or '').lower()
    for k, v in KEYWORDS:
        if k in t:
            return v
    return None


# ------------------------------------------------------------------------------------------------ one test
def to_1hz(t, x, n):
    """mean of 10 Hz samples in [k-0.5, k+0.5) for k = 0..n-1 (NaN where no finite sample)."""
    k = np.rint(t).astype(int)
    ok = (k >= 0) & (k < n) & np.isfinite(x)
    s = np.bincount(k[ok], weights=x[ok], minlength=n)
    c = np.bincount(k[ok], minlength=n)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def _first_valid(x, n=600):
    """first plausible reading (CAN temperatures read exactly 0 until the bus wakes up)."""
    x = x[:n]
    ok = np.isfinite(x) & (x != 0.0) & (x > -45) & (x < 150)      # drop dead sensors (e.g. 1351 C oil temperature)
    return float(np.median(x[ok][:20])) if ok.any() else None


def starts_and_share(on, dt):
    on = np.asarray(on, bool)
    if on.size == 0:
        return None, None
    # debounce: an engine stop shorter than 1 s does not count
    k = max(1, int(round(1.0 / dt)))
    x = on.copy()
    i = 0
    edges = np.flatnonzero(np.diff(x.astype(int)) != 0) + 1
    seg = np.split(np.arange(len(x)), edges)
    for s in seg:
        if not x[s[0]] and len(s) < k and s[0] > 0 and s[-1] < len(x) - 1:
            x[s] = True
    starts = int(np.sum(np.diff(x.astype(int)) == 1) + (1 if x[0] else 0))
    return starts, float(x.mean())


def process(job):
    """job: dict(vehicle meta, source, summary) -> index row (writes the npz/json)."""
    v, src, summ, force = job['veh'], job['src'], job['summary'], job.get('force', False)
    vid = vehicle_id(v['name'])
    tid = 'D3_' + src['num']
    out_npz = os.path.join(TESTS, vid, tid + '.npz')
    row = dict(test_id=tid, vehicle_id=vid, vehicle=v['name'], category=v['cat'], era=src['era'], test_num=src['num'],
               model_year=int(v['name'][:4]), file=os.path.relpath(src['path'], RAW).replace('\\', '/') + (('::' + src['member']) if src['member'] else ''),
               ok=False, note='')
    if not force and os.path.exists(out_npz) and os.path.exists(out_npz[:-4] + '.json'):
        m = json.load(open(out_npz[:-4] + '.json'))
        if m.get('file') == row['file'] and m.get('parser_version') == PARSER_VERSION:
            return m['row']
    try:
        if src['era'] == 'tdms':
            names, units, cols, sel, props = read_tdms_test(src['path'])
        else:
            names, units, cols, sel, props = read_text(src['path'], src['member'])
    except Exception as e:
        row['note'] = f'read failed: {type(e).__name__}: {e}'[:300]
        return row
    s = summ or {}
    diesel = bool(re.search(r'tdi|diesel', v['name'], re.I))
    density = _num(props.get('Fuel_Density_g_ml')) or (0.85 if diesel else 0.742 if 'impala' in v['name'].lower() else 0.7433)
    U = lambda sig: units.get(sel[sig], split_name(sel[sig])[1])
    X = {sig: convert(sig, cols[sel[sig]], U(sig), density) for sig in sel if sel[sig] in cols}
    if 't' not in X or 'v' not in X:
        row['note'] = 'no time or dyno speed channel'
        return row
    t = X['t']
    ok = np.isfinite(t)
    if ok.sum() < 50:
        row['note'] = 'too few samples'
        return row
    dt = float(np.median(np.diff(t[ok]))) if ok.sum() > 2 else 0.1
    keep = ok & (t >= -0.05)
    for k in list(X):
        X[k] = X[k][keep]
    t = X['t']
    n = int(np.floor(np.nanmax(t))) + 1
    # ---- fuel
    fuel_src = 'none'
    fuel = None
    if 'fuel_mass' in X and np.nanmax(np.abs(X['fuel_mass'])) > 0:
        fuel, fuel_src = X['fuel_mass'], 'direct_mass_flow'
    elif 'fuel_vol_pfi' in X and 'fuel_vol_di' in X:                 # dual injection, two volumetric meters
        fuel, fuel_src = np.nan_to_num(X['fuel_vol_pfi']) + np.nan_to_num(X['fuel_vol_di']), f'direct_volume_flow_PFI+DI_x_density_{density:.4f}'
    elif 'fuel_vol' in X and np.nanmax(np.abs(X['fuel_vol'])) > 0:
        fuel, fuel_src = X['fuel_vol'], f'direct_volume_flow_x_density_{density:.4f}'
    elif 'fuel_modal' in X:
        fuel, fuel_src = X['fuel_modal'], 'emissions_modal_carbon_balance'
    elif 'fuel_modal_vol' in X:                       # 2010-2013 zips: bench modal fuel in cc/s
        fuel, fuel_src = X['fuel_modal_vol'], f'emissions_modal_carbon_balance_cc_s_x_density_{density:.4f}'
        X['fuel_modal'] = fuel
    if 'fuel_gas' in X and np.nansum(np.clip(X['fuel_gas'], 0, None)) * dt > 20:      # bi-fuel / CNG / H2: add the gas
        g_ = np.clip(np.nan_to_num(X['fuel_gas']), 0, None)
        fuel = g_ if fuel is None else np.nan_to_num(fuel) + g_
        fuel_src = (fuel_src + '+' if fuel_src != 'none' else '') + 'gas_mass_flow'
    integ = lambda x: float(np.nansum(x) * dt) if x is not None else None
    row['fuel_source'] = fuel_src
    row['fuel_g'] = integ(fuel)
    row['fuel_g_direct'] = integ(fuel) if fuel_src.startswith('direct') else None
    row['fuel_g_modal'] = integ(X.get('fuel_modal'))
    row['fuel_g_summary'] = s.get('fuel_g')
    if v['cat'] == 'alternative' and 'fuel_gas' in X:
        row['gas_fuel_g'] = integ(X['fuel_gas'])
    # ---- battery
    V, I = X.get('batt_V'), X.get('batt_I')
    flipped = False
    P = None
    if V is not None and I is not None and np.nanmax(np.abs(I)) > 2 and np.nanmedian(V) > 60:
        P = V * I
    elif 'batt_P' in X and np.nanmax(np.abs(X['batt_P'])) > 500:
        P = X['batt_P']
    if P is not None:
        Ftr = X.get('F_tractive')
        if Ftr is not None:
            Pw = Ftr * X['v']
            m = np.isfinite(Pw) & np.isfinite(P) & (np.abs(Pw) > 5000)
            if m.sum() > 50 and np.corrcoef(Pw[m], P[m])[0, 1] < -0.2:
                flipped = True
        if flipped:
            P = -P
            if I is not None: I = -I
        row['batt_Wh'] = float(np.nansum(P) * dt / 3600)
        row['batt_Wh_discharge'] = float(np.nansum(np.clip(P, 0, None)) * dt / 3600)
    else:
        row['batt_Wh'] = None
    row['batt_sign_flipped'] = flipped
    row['batt_Wh_summary'] = s.get('batt_Wh')
    # ---- engine
    er = X.get('eng_rpm')
    if er is not None and v['cat'] != 'electric' and np.nanmax(er) > 300:
        on = np.nan_to_num(er) > 400
        row['engine_starts'], row['engine_on_share'] = starts_and_share(on, dt)
    else:
        row['engine_starts'], row['engine_on_share'] = (0, 0.0) if v['cat'] == 'electric' else (None, None)
    # ---- temperatures and HVAC
    Tc = X.get('T_cell')
    row['T_amb_C'] = float(np.nanmedian(Tc)) if Tc is not None and np.isfinite(Tc).any() else s.get('T_set_C')
    if row['T_amb_C'] is None and s.get('T_file_F') is not None:
        row['T_amb_C'] = (s['T_file_F'] - 32) / 1.8
    Ta = row['T_amb_C']
    row['T_class'] = None if Ta is None else 'cold' if Ta < 5 else 'hot' if Ta >= 30 else 'std' if 15 <= Ta < 30 else 'mild'
    for k, sig in (('T_coolant0_C', 'T_coolant'), ('T_oil0_C', 'T_oil'), ('T_cabin0_C', 'T_cabin'), ('T_batt0_C', 'T_batt'), ('T_trans0_C', 'T_trans_oil')):
        row[k] = _first_valid(X[sig]) if sig in X else None
    text = ' '.join(str(s.get(k, '')) for k in ('cycle_text', 'start_comments', 'end_comments', 'section', 'pdf_line'))
    text += ' ' + ' '.join(str(props.get(k, '')) for k in ('Test_CS_HS', 'Test_Cycle_File_Name', 'Test_Start_Comments', 'Test_End_Comments', 'Test_name'))
    tl = text.lower()
    start_flag = None                               # explicit cold/hot field (TDMS Test_CS_HS, PDF CSt/HSt column)
    cshs = str(props.get('Test_CS_HS', '')).strip().upper()
    if cshs in ('CS', 'CST'): start_flag = 'cold'
    elif cshs in ('HS', 'HST'): start_flag = 'hot'
    elif s.get('start_flag'): start_flag = s['start_flag']
    start_txt = None                                # free-text comments (sometimes copied from an earlier test)
    if re.search(r'cold\s*start|\bcst\b|cold soak|overnight soak', tl): start_txt = 'cold'
    elif re.search(r'(warm|hot)\s*start|\bhst\b|no soak', tl): start_txt = 'hot'
    T0s = [x for x in (row['T_coolant0_C'], row['T_oil0_C']) if x is not None]
    T0 = min(T0s) if T0s else None                  # min: some coolant channels read a placeholder (99) until start
    start_data = None
    if T0 is not None and Ta is not None and v['cat'] != 'electric':
        start_data = 'cold' if T0 - Ta < 15 else 'hot'
    start_cyc = None                                # 'UDDS CS' / 'UDDS HS' in the cycle name (zip-era PDFs)
    mc = re.search(r'\b(CS|HS)\b', str(s.get('cycle_text', '')))
    if mc: start_cyc = 'cold' if mc.group(1) == 'CS' else 'hot'
    row['start_flag'], row['start_text'], row['start_data'] = start_flag, start_cyc or start_txt, start_data
    # physical evidence first; the PDF CSt/HSt column is sometimes filled down ("CSt" on every 20 F/95 F row)
    row['start'] = start_data or start_cyc or start_flag or start_txt or 'unknown'
    row['start_source'] = ('initial coolant/oil temperature' if start_data else 'cycle name (CS/HS)' if start_cyc else
                           'test cold/hot flag' if start_flag else 'test comments' if start_txt else 'none')
    sol = None
    if re.search(r'850|solar|w/m2|w/m\^2|lamp', tl):
        sol = not re.search(r'no solar|solar off|lamps? off', tl)
    if 'T_solar_ind' in X and Tc is not None:
        d = np.nanmedian(X['T_solar_ind'] - Tc)
        if np.isfinite(d):
            sol = bool(d > 8) if sol is None else sol
            row['solar_ind_minus_cell_K'] = float(d)
    if sol is None and Ta is not None:
        sol = Ta >= 30                              # D3 95 F tests run with 850 W/m2 lamps
        row['solar_source'] = 'inferred from ambient >= 30 C'
    else:
        row['solar_source'] = 'test notes / solar-array sensor'
    row['solar_lamps'] = bool(sol)
    hv = None
    if re.search(r'hvac[- ]?off|a/?c[- ]?off|climate[- ]?off|hvac\s*off|\boff\s+(up|down|closed|open)\b', tl): hv = 'off'
    elif re.search(r'a/?c[- ]?on|hvac[- ]?on|auto\s*7[0-9]|climate[- ]?on|heater[- ]?on|defrost|\b7[0-4]\s*f\s+(up|down|closed|open)\b', tl): hv = 'on'
    for sig, key in (('P_heater', 'heater_Wh'), ('P_ac', 'ac_Wh'), ('P_batt_heater', 'batt_heater_Wh'), ('P_dcdc', 'dcdc_Wh')):
        row[key] = float(np.nansum(np.clip(X[sig], 0, None)) * dt / 3600) if sig in X else None
    row['hvac_text'] = hv
    # ---- 1 Hz arrays
    A = {}
    A['t'] = np.arange(n, dtype=float)
    vv = to_1hz(t, X['v'], n)
    A['v'] = np.clip(pd.Series(vv).interpolate(limit_direction='both').to_numpy(), 0, None)
    A['grade'] = np.nan_to_num(to_1hz(t, X['grade'], n)) if 'grade' in X else np.zeros(n)
    A['elev'] = np.zeros(n); A['lat'] = np.full(n, np.nan); A['lon'] = np.full(n, np.nan); A['heading'] = np.zeros(n)
    A['T_amb'] = pd.Series(to_1hz(t, Tc, n)).interpolate(limit_direction='both').to_numpy() if Tc is not None else np.full(n, Ta if Ta is not None else np.nan)
    A['wind_speed'] = np.zeros(n); A['wind_dir'] = np.zeros(n)
    A['rho_air'] = np.full(n, RHO_REF)
    A['surface'] = np.zeros(n)
    A['fuel_rate'] = to_1hz(t, fuel, n) if fuel is not None else np.full(n, np.nan)
    A['engine_rpm'] = to_1hz(t, er, n) if er is not None else np.full(n, np.nan)
    A['batt_current'] = to_1hz(t, I, n) if I is not None else np.full(n, np.nan)
    A['batt_voltage'] = to_1hz(t, V, n) if V is not None else np.full(n, np.nan)
    A['soc'] = to_1hz(t, X['soc'] / (100.0 if np.nanmax(X['soc']) > 1.5 else 1.0), n) if 'soc' in X else np.full(n, np.nan)
    A['aux_power'] = to_1hz(t, X['P_dcdc'], n) if 'P_dcdc' in X else np.full(n, np.nan)
    A['solar'] = np.full(n, 850.0 if row['solar_lamps'] else 0.0)
    if P is not None: A['batt_power'] = to_1hz(t, P, n)
    if 'fuel_modal' in X and fuel_src != 'emissions_modal_carbon_balance': A['fuel_rate_modal'] = to_1hz(t, X['fuel_modal'], n)
    if 'v_sched' in X: A['v_sched'] = to_1hz(t, X['v_sched'], n)
    if 'p_cell' in X and Tc is not None:
        A['rho_cell'] = to_1hz(t, X['p_cell'] / (287.05 * (Tc + 273.15)), n)
    for sig in ('T_coolant', 'T_oil', 'T_cabin', 'T_batt', 'T_trans_oil', 'P_heater', 'P_ac', 'P_batt_heater', 'bag', 'F_tractive'):
        if sig in X:
            A[sig] = to_1hz(t, X[sig], n)
    # 10 Hz originals of the dynamic signals
    A['t_10hz'] = t
    A['v_10hz'] = X['v']
    for k, x in (('fuel_rate_10hz', fuel), ('engine_rpm_10hz', er), ('batt_current_10hz', I), ('batt_voltage_10hz', V),
                 ('P_heater_10hz', X.get('P_heater')), ('P_ac_10hz', X.get('P_ac')), ('F_tractive_10hz', X.get('F_tractive'))):
        if x is not None:
            A[k] = x
    # ---- cycles
    vref = A.get('v_sched') if 'v_sched' in A and np.nanmax(A['v_sched']) > 1 else A['v']
    ph = match_cycles(np.nan_to_num(vref), tol=0.6 if vref is not A['v'] else 1.0)
    driving = A['v'] > 0.5
    cov = sum(int(driving[a:b].sum()) for _, a, b, _ in ph) / max(1, int(driving.sum()))
    seq = [p[0] for p in ph]
    row['cycle_seq'] = '+'.join(seq)
    row['n_cycles'] = len(seq)
    row['cycle_coverage'] = float(cov)
    txtc = cycle_from_text(str(props.get('Test_Cycle_File_Name', '')) + ' ' + str(props.get('Test_Start_Comments', ''))) if props.get('Test_Cycle_File_Name') else cycle_from_text(s.get('cycle_text', ''))
    if seq and cov > 0.6:
        u = list(dict.fromkeys('UDDS' if x == 'UDDS_505' else x for x in seq))
        row['cycle'] = u[0] if len(u) == 1 else '+'.join(u)
    else:
        row['cycle'] = txtc or ('steady' if driving.mean() > 0.5 and np.nanstd(np.diff(A['v'][driving])) < 0.05 else 'other')
    if props.get('Test_Cycle_File_Name'):          # TDMS: the cycle file and comments in the file beat the PDF table
        row['cycle_text'] = os.path.splitext(re.split(r'[\\/]', str(props['Test_Cycle_File_Name']))[-1])[0] + ' | ' + str(props.get('Test_Start_Comments', ''))
    else:
        row['cycle_text'] = s.get('cycle_text') or ''
    row['notes_text'] = (str(s.get('start_comments', '')) + ' | ' + str(s.get('end_comments', '')) + ' | ' + str(props.get('Test_Start_Comments', '')) + ' | ' + str(props.get('Test_End_Comments', '')))[:300]
    gr = A['grade']
    row['grade_max_pct'] = float(np.nanmax(np.abs(gr)) * 100)
    # ---- dyno settings
    row['test_weight_lb'] = _num(props.get('Dyno_Test_Weight_lb')) or s.get('weight_lb')
    for k, pk in (('target_A_lbf', 'Dyno_Target_A'), ('target_B_lbf_mph', 'Dyno_Target_B'), ('target_C_lbf_mph2', 'Dyno_Target_C')):
        row[k] = _num(props.get(pk)) if props.get(pk) is not None else s.get(k[7])
    row['fuel_density_g_ml'] = density
    row['fuel_name'] = str(props.get('Fuel_Name', '') or '')
    row['fuel_lhv_btu_lbm'] = _num(props.get('Fuel_Net_HV_BTU_lbm'))
    # ---- totals
    row['duration_s'] = float(n)
    row['distance_km'] = float(np.sum(A['v']) / 1000)
    row['signals'] = ','.join(sorted(sel))
    for sig in ('fuel_mass', 'fuel_vol', 'fuel_modal', 'eng_rpm', 'batt_V', 'batt_I', 'soc', 'T_coolant', 'T_oil', 'T_cabin', 'T_batt',
                'P_heater', 'P_ac', 'P_batt_heater', 'P_dcdc', 'v_sched', 'grade'):
        row['has_' + sig] = sig in sel
    row['dt_s'] = dt
    row['path'] = f'tests/{vid}/{tid}.npz'
    # quality: speed trace present and non-trivial
    row['ok'] = bool(n > 30 and np.isfinite(A['v']).all())
    if row['ok'] and row['distance_km'] < 0.05 and row['cycle'] not in ('idle', 'charge'):
        row['note'] = 'no distance driven'
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz, **{k: np.asarray(x, np.float32) for k, x in A.items()})
    meta = dict(row=row, file=row['file'], parser_version=PARSER_VERSION, channels={k: sel[k] for k in sel},
                units={k: units.get(sel[k], '') for k in sel}, phases=[dict(cycle=a, start=b, end=c, rms_mps=round(d, 3)) for a, b, c, d in ph],
                summary={k: (v_ if not isinstance(v_, float) or np.isfinite(v_) else None) for k, v_ in s.items()},
                test_info={k: (str(v_) if not isinstance(v_, (int, float)) else v_) for k, v_ in props.items()},
                n_channels_in_file=len(names))
    json.dump(meta, open(out_npz[:-4] + '.json', 'w'), indent=1, default=str)
    return row


PARSER_VERSION = 4


def _safe(job):
    try:
        return process(job)
    except Exception as e:
        return dict(test_id='D3_' + job['src']['num'], vehicle_id=vehicle_id(job['veh']['name']), vehicle=job['veh']['name'],
                    category=job['veh']['cat'], era=job['src']['era'], test_num=job['src']['num'], ok=False,
                    note=f'{type(e).__name__}: {e}'[:300] + ' @ ' + traceback.format_exc().splitlines()[-3][:150])


def complete_vehicles():
    """vehicles whose every file in listing.json is on disk with the listed size."""
    L = json.load(open(os.path.join(RAW, 'listing.json'), encoding='utf-8'))
    done = {}
    for v in L:
        d = vehicle_dir(v['name'])
        n_ok = 0
        for f in v['files']:
            p = os.path.join(RAW, d, *[re.sub(r'[<>:"|?*]', '_', s_) for s_ in f['path'].split('/')])
            n_ok += os.path.exists(p) and os.path.getsize(p) == f['size']
        done[v['name']] = (n_ok, len(v['files']))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--vehicles', default='')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    names = [n.strip().lower() for n in a.vehicles.split(',') if n.strip()]
    status = complete_vehicles()
    jobs, chan = [], {}
    for cat, name, slug, host, share in VEHICLES:
        if name in SKIP or (names and not any(n in name.lower() for n in names)):
            continue
        srcs = discover(name)
        summ = vehicle_summaries(name)
        chan[vehicle_id(name)] = dict(vehicle=name, category=cat, files_on_disk=status.get(name, (0, 0))[0],
                                      files_listed=status.get(name, (0, 0))[1], n_tests_found=len(srcs),
                                      n_summary_rows=len(summ))
        for s in srcs:
            jobs.append(dict(veh=dict(cat=cat, name=name), src=s, summary=summ.get(s['num']), force=a.force))
    print(f'{len(jobs)} tests from {len(chan)} vehicles', flush=True)
    jobs.sort(key=lambda j: -os.path.getsize(j['src']['path']))       # big files first
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(_safe, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f'  {i + 1}/{len(jobs)}  {time.time() - t0:.0f} s', flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(PROC, exist_ok=True)
    old = os.path.join(PROC, 'index.parquet')
    if names and os.path.exists(old):                  # partial run: merge with the existing index
        prev = pd.read_parquet(old)
        df = pd.concat([prev[~prev.vehicle_id.isin(df.vehicle_id.unique())], df], ignore_index=True)
    df = df.sort_values(['category', 'vehicle', 'test_num']).reset_index(drop=True)
    for c in df.columns:
        if df[c].dtype == object:
            vals = df[c].dropna()
            if len(vals) and vals.map(lambda x: isinstance(x, (bool, np.bool_))).all():
                df[c] = df[c].astype('boolean')
    df.to_parquet(old, index=False)
    # channel inventory per vehicle (which raw channel fed each signal) from the meta files
    inv_path = os.path.join(PROC, 'channels.json')
    inv = json.load(open(inv_path)) if (names and os.path.exists(inv_path)) else {}
    for vid, c in chan.items():
        per = {}
        for p in glob.glob(os.path.join(TESTS, vid, '*.json')):
            m = json.load(open(p))
            for sig, raw in m['channels'].items():
                per.setdefault(sig, {}).setdefault(f"{raw} [{m['units'].get(sig, '')}]", 0)
                per[sig][f"{raw} [{m['units'].get(sig, '')}]"] += 1
            c.setdefault('eras', set()).add(m['row']['era'])
            c['n_channels_in_file_max'] = max(c.get('n_channels_in_file_max', 0), m.get('n_channels_in_file', 0))
        c['eras'] = sorted(c.get('eras', []))
        c['signals'] = per
        inv[vid] = c
    json.dump(inv, open(inv_path, 'w'), indent=1)
    ok = df[df.ok == True]
    print(f'{len(df)} tests, {len(ok)} ok, {df.vehicle_id.nunique()} vehicles; {time.time() - t0:.0f} s')
    print(ok.groupby('category').agg(vehicles=('vehicle_id', 'nunique'), tests=('test_id', 'size')).to_string())
    bad = df[df.ok != True]
    if len(bad):
        print('not ok:'); print(bad[['vehicle', 'test_num', 'note']].to_string()[:4000])


if __name__ == '__main__':
    main()
