"""Engine generations by model year (physics v3 option 'engine_era').

RoadSim has one hybrid Atkinson map, the 2017 Toyota 2.5 L TNGA prototype (EPA ALPHA package; peak brake thermal
efficiency 40.75 % as published, Toyota's production claim 41 %). It stands in for every hybrid engine, also the older
generations that EPA, ORNL and ANL never published a map for (1NZ-FXE, 2ZR-FXE, 2AR-FXE: no ALPHA package exists;
the ANL Autonomie maps sit behind anl.gov's browser challenge and are not used). The v3 option scales that map's
efficiency for older engines by the ratio of published peak brake thermal efficiencies:

    eta_era(w, T) = eta_map(w, T) x (BTE_era / BTE_ref) ** exp

option value (data/calibration_v3.json), {} = off (v1, v2, v2.1):
    {'maps': [bsfc_map ids it applies to], 'ref_bte': 0.41,
     'eras': [[make regex, first MY, last MY, peak BTE], ...], 'exp': 1.0}
Only makes with a published generation history are listed (Toyota / Lexus); other makes keep the map (scale 1).

The model year comes from the spec: spec['model_year'] (EPA hybrid specs, checks/hybrid_specs.py), else a
19xx/20xx token in spec['name'] or spec['id'] ('2010 Toyota Prius', 'D3_2010_Toyota_Prius'), else
spec['generator']['donor_year'] when the option says 'generator': true. A spec without a model year is modern
(scale 1). The make is spec['make'], else the word after the year in the name or id ('D3_2010_Toyota_Prius' ->
TOYOTA); generated vehicles carry spec['make'] when the generator sets one.

v3.1 (14 Nov 2026): the option value may carry 'blocks', a list of further dicts of the same form, each optionally
limited to architectures ('archs': prefixes such as 'hev', 'phev'); the scales of all matching blocks multiply. An era entry may name a measured load table as a fifth element,
    [make regex, first MY, last MY, peak BTE, table name]
with 'tables': {name: [[BMEP bar, fuel ratio], ...]} in the option value (data/engine_era_d3.json, built by
checks/d3_engine_maps.py from D3 dev 10 Hz data). The table multiplies the warm map's fuel power at each torque by the
ratio at that BMEP (4 pi T / displacement; linear between points, the end values beyond them). With 'table_norm': true
the table is divided by its last (highest-load) ratio, so the peak scaling above still sets the high-load level and
the table adds the measured low-load excess only. thermal.engine_map() applies it at the start of every run: the
engine gets a re-weighted fuel grid and that grid's optimal operating line (the vehicle cache keeps the original and
every run restores it first).
"""
import re

_YEAR = re.compile(r'(?:^|[\s_])((?:19|20)\d\d)(?=$|[\s_])')


def model_year(spec, generator=False):
    y = spec.get('model_year')
    if y is not None:
        try:
            return int(y)
        except (TypeError, ValueError):
            return None
    for k in ('name', 'id'):
        m = _YEAR.search(str(spec.get(k) or ''))
        if m:
            return int(m.group(1))
    if generator:
        y = (spec.get('generator') or {}).get('donor_year')
        if y is not None and y == y:
            return int(y)
    return None


def make_of(spec):
    mk = spec.get('make')
    if mk:
        return str(mk).upper()
    for k in ('name', 'id'):
        m = re.search(r'(?:^|[\s_])(?:19|20)\d\d[\s_]+([A-Za-z-]+)', str(spec.get(k) or ''))
        if m:
            return m.group(1).upper()
    return ''


def _blocks(era):
    """the option value as a list of blocks: the v3 form (one dict) plus, from v3.1, its 'blocks' list of dicts of the
    same form (each may add 'archs': architecture prefixes it applies to)."""
    if not era:
        return []
    return [era] + list(era.get('blocks') or [])


def _row(spec, map_id, blk):
    if map_id not in (blk.get('maps') or ()):
        return None
    archs = blk.get('archs')
    if archs and not str(spec.get('architecture') or '').startswith(tuple(archs)):
        return None
    y = model_year(spec, bool(blk.get('generator', False)))
    if y is None:
        return None
    mk = make_of(spec)
    for row in blk.get('eras') or ():
        mre, a, b = row[0], row[1], row[2]
        if a <= y <= b and re.search(mre, mk):
            return row
    return None


def scale(spec, map_id, era):
    """efficiency multiplier (<= 1 for older engines) for this spec's engine under option value `era`: the product over
    the blocks of (BTE_era / ref_bte) ** exp, first matching row of each block."""
    out = 1.0
    for blk in _blocks(era):
        row = _row(spec, map_id, blk)
        if row is not None:
            out *= (float(row[3]) / float(blk.get('ref_bte', 0.41))) ** float(blk.get('exp', 1.0))
    return out


def table(spec, map_id, era):
    """[[BMEP bar, fuel ratio], ...] for this spec's engine under option value `era`, or None (no load table): the first
    block whose matching row names a table."""
    for blk in _blocks(era):
        row = _row(spec, map_id, blk)
        if row is None or len(row) < 5 or not row[4]:
            continue
        tab = [[float(x), float(r)] for x, r in (blk.get('tables') or {})[row[4]]]
        if blk.get('table_norm'):
            top = tab[-1][1]
            tab = [[x, r / top] for x, r in tab]
        return tab
    return None


def p2_type(spec):
    """P2 hardware type: spec['p2_type'] (EPA hybrid specs: 'tmed', 'converter', 'clutch', 'mild'), else Hyundai / Kia
    P2s without a torque converter are 'tmed' (engine started by the belt hybrid starter-generator, HSG), other P2s
    'converter' when the transmission keeps one, else 'clutch'."""
    t = spec.get('p2_type')
    if t:
        return t
    tr = spec.get('transmission') or {}
    has_tc = tr.get('type') == 'AT' and tr.get('torque_converter', True)
    if re.search('HYUNDAI|KIA', make_of(spec)) and not has_tc:
        return 'tmed'
    return 'converter' if has_tc else 'clutch'


def phev_type(spec):
    """plug-in heating hardware type (v3.1 option phev_heat_by_type): spec['phev_type'] when the twin builder set it
    (VED twins: checks/check_ved_twin.py), else from the name / id and model year:
      volt_gen1    Chevrolet Volt 2011-2015, Cadillac ELR 2014: engine runs for heat below 25 F (-3.9 C) (GM PIC5414C)
      volt_gen2    Chevrolet Volt 2016-2019: below 35 F (+1.7 C) or 15 F (-9.4 C) by the 'Engine Assisted Heating'
                   setting (GM PIC5414G / PIC6231C)
      ford_energi  Ford C-Max / Fusion Energi: electric heater; the engine starts in EV mode when it is cold enough
                   that the cabin needs engine heat, to heat the cabin and to limit emissions (owner's manual)
      toyota_phev  Prius Plug-in / Prius Prime
    None for every other plug-in (the global phev_heat_* options apply)."""
    t = spec.get('phev_type')
    if t:
        return t
    a = spec.get('architecture') or ''
    if not a.startswith('phev'):
        return None
    txt = (str(spec.get('name') or '') + ' ' + str(spec.get('id') or '')).upper()
    y = model_year(spec)
    if re.search(r'VOLT|ELR', txt):
        return 'volt_gen1' if (y is not None and y <= 2015) else 'volt_gen2'
    if 'ENERGI' in txt:
        return 'ford_energi'
    if re.search('TOYOTA', txt) and re.search('PRIUS', txt):
        return 'toyota_phev'
    return None
