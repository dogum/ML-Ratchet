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


def scale(spec, map_id, era):
    """efficiency multiplier (<= 1 for older engines) for this spec's engine under option value `era`."""
    if not era or map_id not in (era.get('maps') or ()):
        return 1.0
    y = model_year(spec, bool(era.get('generator', False)))
    if y is None:
        return 1.0
    mk = make_of(spec)
    for mre, a, b, bte in era.get('eras') or ():
        if a <= y <= b and re.search(mre, mk):
            return (float(bte) / float(era.get('ref_bte', 0.41))) ** float(era.get('exp', 1.0))
    return 1.0


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
