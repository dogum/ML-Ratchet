"""Page-lab reference: a small GridSim system built from four sliders (system, MW lost, inertia, renewable share,
fast response MW). The page runs a JS port of gridsim.frequency_event on the same state; check_lab_js.py compares."""
from common import *
import gridsim as gs
from gridsim.frequency import _compile

MIX = {"GB": [("gas_cc", 0.55), ("nuclear", 0.25), ("biomass", 0.12), ("hydro_res", 0.03), ("gas_ct", 0.05)],
       "ERCO": [("gas_cc", 0.45), ("coal_steam", 0.18), ("nuclear", 0.12), ("gas_ct", 0.15), ("gas_steam", 0.10)]}
DEMAND = {"GB": 30000.0, "ERCO": 50000.0}


def state(system="GB", loss=1000.0, inertia_GWs=200.0, ibr=0.4, fast_MW=1000.0, demand=None):
    f0 = 50.0 if system == "GB" else 60.0
    dem = demand or DEMAND[system]
    sync = dem * (1 - ibr)
    units = []
    for tech, sh in MIX[system]:
        P = sync * sh
        units.append(dict(tech=tech, MW=0.85 * P, pmax_MW=P, pmin_MW=0.4 * P, mva=P / 0.9, H_s=1.0,
                          governor=dict(droop=0.05, headroom_frac=0.06 if tech != "nuclear" else 0.0,
                                        enabled=tech != "nuclear",
                                        deadband_Hz=0.015 if system == "GB" else (0.034 if tech in ("coal_steam", "gas_steam") else 0.0166))))
    E0 = sum(u["mva"] for u in units)
    for u in units:
        u["H_s"] = inertia_GWs * 1e3 / E0
    units.append(dict(tech="wind", MW=dem * ibr, pmax_MW=dem * ibr, H_s=0.0, mva=dem * ibr))
    ffr = []
    if system == "GB" and fast_MW > 0:          # Dynamic Containment knee (see gpsim p1)
        for MW, db, full in ((0.05 * fast_MW, 0.015, 0.2), (0.95 * fast_MW, 0.2, 0.5)):
            units.append(dict(tech="battery", MW=0.0, pmax_MW=MW, pmin_MW=-MW, H_s=0.0, mva=MW,
                              governor=dict(enabled=True, droop=(full - db) / f0, deadband_Hz=db, up_MW=MW, dn_MW=MW, T_s=0.5)))
    if system == "ERCO":
        if fast_MW > 0:
            ffr.append(dict(MW=fast_MW, trigger_Hz=59.85, delay_s=0.25, ramp_s=0.1))
        ffr.append(dict(MW=1000.0, trigger_Hz=59.7, delay_s=0.4, ramp_s=0.05))    # load resources (RRS)
    st = gs.make_system_state(units, load_MW=dem, region=system, D=0.75 if system == "GB" else 1.0, ffr=ffr,
                              secondary=dict(MW=0.15 * loss, T_s=60.0, deadband_Hz=0.015))
    return st


def compiled(st):
    """Flat arrays the JS port needs."""
    f0 = st["f0"]; P = _compile(st, f0)
    E = gs.frequency.inertia_MWs(st)
    return dict(f0=f0, E=E, D=st["D"], load=st["load_MW"], K=P["K"].tolist(), db=P["db"].tolist(), T1=P["T1"].tolist(),
                rate=P["rate"].tolist(), up=P["up"].tolist(), dn=P["dn"].tolist(), An=P["An"].tolist(), Ad=P["Ad"].tolist(),
                Bn=P["Bn"].tolist(), Bd=P["Bd"].tolist(), ffr=st["ffr"], ufls=st["ufls"], secondary=st["secondary"])


def simulate(st, loss, t_end=68.0):
    ev = gs.frequency_event(st, loss, t_end=t_end, dt=0.1, h=0.02, record=False)
    return ev["t"], ev["f"]
