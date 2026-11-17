"""Run under buildingsim/.refenv (CoolProp, pythermalcomfort). Reads a JSON request, writes JSON answers."""
import json
import sys

req = json.load(open(sys.argv[1]))
out = {}
if 'psychro' in req:
    from CoolProp.HumidAirProp import HAPropsSI
    res = []
    for T, RH, P in req['psychro']:
        Tk = T + 273.15
        r = dict(W=HAPropsSI('W', 'T', Tk, 'P', P, 'R', RH / 100.0), Twb=HAPropsSI('B', 'T', Tk, 'P', P, 'R', RH / 100.0) - 273.15,
                 Tdp=HAPropsSI('D', 'T', Tk, 'P', P, 'R', RH / 100.0) - 273.15,
                 H=HAPropsSI('H', 'T', Tk, 'P', P, 'R', RH / 100.0),
                 V=HAPropsSI('V', 'T', Tk, 'P', P, 'R', RH / 100.0))
        res.append(r)
    out['psychro'] = res
if 'pmv' in req:
    from pythermalcomfort.models import pmv_ppd_iso
    res = []
    for ta, tr, v, rh, met, clo in req['pmv']:
        r = pmv_ppd_iso(tdb=ta, tr=tr, vr=v, rh=rh, met=met, clo=clo, limit_inputs=False, round_output=False)
        res.append(dict(pmv=float(r.pmv), ppd=float(r.ppd)))
    out['pmv'] = res
json.dump(out, open(sys.argv[2], 'w'))
