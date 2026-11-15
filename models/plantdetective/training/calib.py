"""Versioned simulator / emulator settings (one entry per bank version; later versions hold dev-calibrated values).
Dev calibration only ever uses the 10 dev plants (40 units) and dev-year inputs."""
import copy
from emulate import EMU

BASE = dict(
    emu=copy.deepcopy(EMU),
    start_mult=dict(coal_steam=(1.0, 0.8), gas_cc=(1.0, 0.8), gas_ct=(0.5, 0.8), gas_steam=(1.0, 0.8)),
    fc_noise=(0.03, 0.3),
    p_gs_background=0.3,
    p_cogen_given_chp=0.6,
    include_gs_units=True,
    kappa=dict(coal_steam=(0.0, 0.0), gas_cc=(0.0, 0.0), gas_ct=(0.0, 0.0), gas_steam=(0.0, 0.0)),
    cogen_trips=(1.0, 10.0),
)

VERSIONS = {
    'v0': dict(p_gs_background=1.0),       # GridSim unit-level MILP units only (bank_dp called with 0 DP units)
    'v1': dict(),                          # + self-committed / cogeneration units on GridSim and ERCOT backgrounds
    'v2pre': dict(cogen_trips=(0.5, 4.0)),  # dev-measured start/stop trajectories + short spells (code), before grid
    # v2: dev twin calibration (out/calib_twin_v2pre.json grid + dev start/stop measurements + dev EIA-923 ratio)
    'v2': dict(cogen_trips=(0.5, 4.0), cogen_level='wide', include_gs_units='separate',
               start_mult=dict(coal_steam=(2.0, 0.5), gas_cc=(1.5, 0.5), gas_ct=(1.0, 0.5), gas_steam=(0.5, 0.5)),
               kappa=dict(coal_steam=(3.0, 12.0), gas_cc=(5.0, 20.0), gas_ct=(5.0, 15.0), gas_steam=(20.0, 60.0)),
               steam_credit=(0.0, 15.0),
               emu=dict(hr923_basis='gross', hr923_sigma=0.05, chp_alloc=(0.4, 0.9),
                        q_trip=dict(gas_ct=0.02, gas_cc=0.05, coal_steam=0.15, gas_steam=0.05),
                        q_abort=dict(gas_ct=0.02, gas_cc=0.05, coal_steam=0.15, gas_steam=0.05),
                        start=dict(gas_ct=dict(op1=(0.3, 1.0), a1=(0.3, 0.8), r_wu=(0.5, 1.0)),
                                   gas_cc=dict(op1=(0.30, 1.0), a1=(0.08, 0.30), r_wu=(0.25, 0.6)),
                                   coal_steam=dict(op1=(0.50, 1.0), a1=(0.0, 0.05), r_wu=(0.02, 0.08)),
                                   gas_steam=dict(op1=(0.60, 1.0), a1=(0.0, 0.05), r_wu=(0.06, 0.2))))),
}


def get(version):
    c = copy.deepcopy(BASE)
    for k, v in VERSIONS.get(version, {}).items():
        if k == 'emu':
            c['emu'].update(v)
        else:
            c[k] = v
    return c
