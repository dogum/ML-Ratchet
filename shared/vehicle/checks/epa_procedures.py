"""Emulate EPA chassis-dynamometer procedures with roadsim.

FTP-75: cold-start UDDS (bags 1+2) after a soak at 24 C, 10-minute hot soak, hot-start bag 3 (first 505 s of UDDS).
        FE = 1 / (0.43 (g1+g2)/(d1+d2) + 0.57 (g3+g2)/(d3+d2))  [40 CFR 600.113]
HWFET, US06: hot start (preceded by a preconditioning run).
Hybrids: test starts at the SOC reached after preconditioning runs; residual net battery energy converted to fuel with a
          fixed 1/(0.33 x 0.90) equivalence (SAE J1711-style correction).
BEV: DC battery energy over a warm cycle, divided by an assumed 0.88 wall-to-battery charging efficiency to compare
     with EPA's AC recharge energy (SAE J1634).  HVAC off for FTP/HWFET/US06 (as on the test).

Procedure corrections of 13 Nov 2026 (applied to every physics version; logged in roadsim/CALIBRATION.md):
- 20 F cold FTP (40 CFR 1066.710(c)): heat on, A/C off, outside air ((c)(1)(i)(D)), full automatic climate control
  at 72 F ((c)(2); the D3 cold tests are logged "HVAC ON-AUTO-72F"). Before: HVAC off. The vehicle is soaked 12-36 h at
  -7 C ((e)), so the tyres start at the soak temperature (before: warm tyres).
- Every soak (FTP 10-minute hot soak, SC03 soak) cools every thermal state RoadSim has: coolant (and oil, coupled),
  transmission, axle oil, tyres, PTC loop, v1 tyre warm-up state (before: coolant and transmission only).
- SC03 (40 CFR 1066.835): 35 C, 850 W/m2, preconditioning drive over the first 505 s of the UDDS ((c)(5)), 9-11 min
  soak with the engine off ((c)(6)), A/C "maximum": recirculated air, automatic 72 F ((c)(4)). Cabin air and interior
  at 45 C at the start of SC03 (D3 dev SC03 tests: 44-46 C). Before: no preconditioning drive, cabin 50 C, hot engine.
- Options: every simulate() call takes the check chain's physics options (checks/_opts.py; None = physics v1).

Since 14 Nov 2026 the code lives in the roadsim package (roadsim/roadsim/epa.py, explicit options= argument); this module
keeps the old import path and behaviour for the check chain and EnergyNet's training (models/energynet/training/
relabel.py): every procedure runs with the check chain's current options (checks/_opts.py; None = physics v1).
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'roadsim'))
sys.path.insert(0, HERE)
import _opts
from roadsim import epa as _E
from roadsim.epa import (T_TEST, T_COLD, T_SET_72F, COLD_HVAC, SC03_HVAC, SC03_CABIN_SOAK_C, ETA_CHARGE, ETA_BATT_EQ,  # noqa: F401
                         RHO_REF, test_spec, trip_for, _gal, _bag, _batt_out_J, _fuel_equiv_g)


def _sim(spec, trip, extra=None, **kw):
    return _E._sim(spec, trip, extra, _opts.get(), **kw)


def soak(spec, fs, T, secs, extra=None, options=None):
    return _E.soak(spec, fs, T, secs, extra, _opts.get() if options is None else options)


def cabin_soak(spec, fs, T, secs, solar, extra=None):
    return _E.cabin_soak(spec, fs, T, secs, solar, extra, options=_opts.get())


def precondition(spec, name, strategy=None, n=2, init=None, T=T_TEST, extra=None, solar=0.0, soc0=None):
    return _E.precondition(spec, name, strategy, n, init, T, extra, solar, soc0, options=_opts.get())


def ftp(spec, strategy=None, T=T_TEST, cold=True, soc0=None):
    return _E.ftp(spec, strategy, T, cold, soc0, options=_opts.get())


def hot_cycle(spec, name, strategy=None, T=T_TEST, soc0=None):
    return _E.hot_cycle(spec, name, strategy, T, soc0, options=_opts.get())


def sc03(spec, strategy=None, return_run=False, soc0=None):
    return _E.sc03(spec, strategy, return_run, soc0, options=_opts.get())
