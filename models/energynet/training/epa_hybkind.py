"""Rung r8 input: the hybrid system of each EPA test-car hybrid from its model line (RoadSim's hybrid_specs.lookup, a
table of public spec sheets written from model names, shared/vehicle/checks; read-only import). RoadData's EPA table
says only 'hev'; v1 told the network every EPA hybrid was a power-split. Adds 'hyb_arch' and 'hyb_trans' to
data/real_epa_<split>_meta.parquet (P2 -> hev_parallel_p2 with the EPA transmission, series -> hev_series, power-split
-> hev_powersplit; excluded or unknown designs stay power-split). python epa_hybkind.py"""
import sys, os
import _paths as _P
from common import *
import pandas as pd
import realdata as rd
sys.path.insert(0, os.path.join(_P.VEHICLE, 'checks'))
import hybrid_specs as H
from realdata import _TRANS

KIND = {'p2': 'hev_parallel_p2', 'series': 'hev_series', 'ps': 'hev_powersplit', 'fps': 'hev_powersplit', 'ops': 'hev_powersplit'}
e = rd.epa_testcars()
for split in ('dev', 'exam'):
    p = f'{EN}/data/real_epa_{split}_meta.parquet'
    M = pd.read_parquet(p)
    ha, ht = [], []
    for vid, arch in zip(M.vehicle_id, M.arch):
        if arch != 'hev_powersplit':
            ha.append(arch); ht.append(None); continue
        r = e.loc[vid]
        k = H.lookup(r.make, r.model, int(r.model_year), r.displacement_L)[0]
        a = KIND.get(k, 'hev_powersplit')
        ha.append(a)
        ht.append('eCVT' if a == 'hev_powersplit' else ('single' if a == 'hev_series' else (_TRANS.get(str(r.trans_code)) or 'AT')))
    M['hyb_arch'] = ha; M['hyb_trans'] = ht
    M.to_parquet(p)
    print(split, M[M.arch == 'hev_powersplit'].drop_duplicates('vehicle_id').hyb_arch.value_counts().to_dict())
