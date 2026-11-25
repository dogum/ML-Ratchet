"""Conformal wrapper (agent C, Vehicle Bake-off): enet_v21 + Mondrian split-conformal bands.

The wrapped arm's predict() runs unchanged; its 5-95 % and 25-75 % band endpoints are multiplied by exp(shift) per
stratum (powertrain x road / dyno / heavy duty, from the batch's arch, cls and dynamometer flag), with shifts fitted
on enet_v21's DEV residuals only (calib.json; method in ../lib/vcf_mondrian.py). pred and q50 are untouched.
numpy only (plus whatever the wrapped arm needs).
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = json.load(open(os.path.join(HERE, 'calib.json')))
_spec = importlib.util.spec_from_file_location('vcf_mondrian_' + os.path.basename(HERE),
                                               os.path.normpath(os.path.join(HERE, '..', 'lib', 'vcf_mondrian.py')))
CF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CF)
_HARNESS = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'harness'))
if _HARNESS not in sys.path:
    sys.path.insert(0, _HARNESS)
import armlib  # noqa: E402

BASE = os.path.normpath(os.path.join(HERE, CAL['base_rel']))
_base_predict, _base_mod = armlib.load_predict(BASE)


def _wrap(b, o):
    return CF.apply(o, CF.strata_of(b), CAL['tracks'][b['track']]['shifts'])


def predict(batch):
    return _wrap(batch, _base_predict(batch))


def predict_many(batches):
    if hasattr(_base_mod, 'predict_many'):
        outs = _base_mod.predict_many(batches)
    else:
        outs = [_base_predict(b) for b in batches]
    return [_wrap(b, o) for b, o in zip(batches, outs)]
