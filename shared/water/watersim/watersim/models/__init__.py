"""Runoff-generation models. Each module exposes NAME, PARAMS (name, default, lo, hi, unit, description), STATES,
init_state(p, init) and a numba `step` with a common signature (see engine.py)."""
from . import hbv, gr4j, sacsma, vic_lite

MODULES = {'hbv': hbv, 'gr4j': gr4j, 'sacsma': sacsma, 'vic_lite': vic_lite}
MODEL_IDS = {'hbv': 0, 'gr4j': 1, 'sacsma': 2, 'vic_lite': 3, 'none': -1}

CAPACITY_LABEL = {   # an effective soil water capacity (mm) per model, for labels
    'hbv': lambda p: p['FC'],
    'gr4j': lambda p: p['X1'],
    'sacsma': lambda p: p['UZTWM'] + p['LZTWM'],
    'vic_lite': lambda p: p['W1M'],
}
