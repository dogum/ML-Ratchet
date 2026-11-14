"""Aggregated swing model vs a full multi-machine simulation (ANDES 2.0, Kundur two-area system).

ANDES: 4 GENROU machines (6th-order), EXDC2 exciters, TGOV1 governors, 15 lines, constant-power loads (p2p=q2q=1);
a 200 MW constant-power load switched on at bus 7 at t = 1 s.  Centre-of-inertia frequency of the 4 machines.
gridsim: the same four machines as TGOV1-type units (T1 0.49 s, reheat 2.1/7 s, R 0.05 on 900 MW), no load damping,
200 MW imbalance.
Run with the reference venv:  <refenv python> check_andes.py   (or run_all.py --ref <refenv python>)
Writes grid/checks/frequency_andes.json and out/andes_kundur.png
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import json, sys
sys.path.insert(0, f"{GRID_CODE}/gridsim")
import numpy as np
import andes
from gridsim.frequency import frequency_event, make_system_state



def main():
    """Under a __main__ guard: ANDES starts worker processes, and Windows spawns them by re-importing this file."""
    andes.config_logger(stream_level=40)
    case = andes.get_case("kundur/kundur_full.xlsx")
    ss = andes.load(case, setup=False, no_output=True, default_config=True)
    ss.add("PQ", dict(idx="PQ_step", bus=7, p0=2.0, q0=0.0, u=0))   # 200 MW load, switched on at t = 1 s
    ss.add("Toggle", dict(model="PQ", dev="PQ_step", t=1.0))
    ss.setup()
    ss.Toggle.u.v[0] = 0   # disable the case's built-in line trip; keep only the load step
    ss.PQ.config.p2p = 1.0; ss.PQ.config.q2q = 1.0; ss.PQ.config.p2z = 0.0; ss.PQ.config.q2z = 0.0
    ss.PFlow.run()
    ss.TDS.config.tf = 31.0
    ss.TDS.config.tstep = 0.005
    ss.TDS.config.no_tqdm = 1
    ss.TDS.run()
    t = np.array(ss.dae.ts.t)
    omega = ss.dae.ts.x[:, ss.GENROU.omega.a]
    M = ss.GENROU.M.v   # system base (relative weights only)
    alive = np.array([True, True, True, True])
    f0 = ss.config.freq
    fcoi = f0 * (omega[:, alive] * M[alive]).sum(1) / M[alive].sum()
    Pg0 = ss.GENROU.Pe.v if hasattr(ss.GENROU, "Pe") else None

    H = ss.GENROU.M.v * ss.config.mva / ss.GENROU.Sn.v / 2   # ANDES stores M on the system base after setup
    units = [dict(tech="coal_steam", MW=700.0, pmax_MW=900.0, H_s=float(H[i]), mva=900.0,
                  governor=dict(enabled=True, droop=0.05, deadband_Hz=0.0, T_s=0.49, LL_A=(2.1, 7.0), LL_B=(0, 0),
                                up_MW=1e9, dn_MW=1e9)) for i in range(4)]
    st = make_system_state(units, load_MW=2734.0, region="ERCO", D=0.0, ufls=[], db=0.0)
    import gridsim.frequency as F
    F.TECH_DYN["coal_steam"]["rate"] = 1e6
    r = frequency_event(st, 200.0, t_end=30.0, dt=0.005, h=0.005)
    ts = t - 1.0
    m = ts >= 0
    fs = np.interp(ts[m], r["t"], r["f"])
    fa = fcoi[m]
    out = {"andes_version": andes.__version__, "disturbance": "200 MW load step at bus 7 (constant-power loads)",
           "andes": {"nadir_Hz": float(fa.min()), "t_nadir_s": float(ts[m][fa.argmin()]), "f_30s_Hz": float(fa[-1]),
                     "rocof_0_0p5s": float(np.polyfit(ts[m][ts[m] <= 0.5], fa[ts[m] <= 0.5], 1)[0])},
           "gridsim": {"nadir_Hz": float(r["f"].min()), "t_nadir_s": float(r["t"][r["f"].argmin()]),
                       "f_30s_Hz": float(r["f"][-1]), "rocof_0_0p5s": float(r["rocof_Hz_s"])},
           "rms_diff_Hz": float(np.sqrt(np.mean((fs - fa) ** 2))),
           "note": "differences come from electrical losses changing with flows, voltage dynamics and the 6th-order machines"}
    json.dump(out, open(f"{GRID_CODE}/checks/frequency_andes.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4)); plt.plot(ts[m], fa, "k", label="ANDES 4-machine COI"); plt.plot(r["t"], r["f"], "C0--", label="gridsim aggregate")
        plt.xlabel("s after trip"); plt.ylabel("Hz"); plt.legend(); plt.grid(alpha=.3); plt.title("Kundur two-area: 200 MW load step")
        plt.tight_layout(); plt.savefig(f"{GRID_ROOT}/checks/out/andes_kundur.png", dpi=110)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
