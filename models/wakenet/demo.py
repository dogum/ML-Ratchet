"""Run WakeNet on 6 real exam snapshots (Kelmarsh 2020-21 + unseen 14-turbine Penmanshiel 2019-20).

Examples were drawn once by a fixed seed (20261116) within input-defined wind strata — never selected by error.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from wakenet import WakeNet  # noqa: E402


def main():
    t0 = time.perf_counter()
    net = WakeNet()
    files = sorted(glob.glob(os.path.join(HERE, "examples", "*.npz")))
    print(f"WakeNet v{net.meta.get('version', '1.0')} — {len(files)} real wind-farm exam snapshots (0 real wake labels in training)\n")

    farm_errs = []
    waked_errs = []
    inside_68 = []

    for fp in files:
        z = np.load(fp)
        m = json.loads(str(z["meta"]))
        r = net.read(
            east_m=z["east_m"],
            north_m=z["north_m"],
            hub_h_m=z["hub_h_m"],
            turbine=m["turbine_model"],
            u_inf=m["u_inf_ms"],
            wind_dir_deg=m["wind_dir_deg"],
            ti_inf=m["ti_inf"],
            sigma_wd_deg=m["sigma_wd_deg"],
            yaw_deg=z["yaw_deg"],
            online=z["online"],
            elev_m=z["elev_m"],
            air_density=m["air_density"],
            p_front_kw=m["p_front_kw"],
        )

        on = z["online"].astype(bool)
        free = r["is_free"]
        waked = on & (~free)
        p_true = z["power_true_kw"].astype(np.float64)
        p_front = float(m["p_front_kw"])
        eta_true = np.where(on, p_true / p_front, 0.0)
        w_farm_true = 1.0 - float(np.sum(eta_true[on])) / float(np.sum(on))

        f_pred_pct = 100.0 * r["farm_wake_loss"]
        f_sig_pct = 100.0 * r["farm_wake_sig"]
        f_true_pct = 100.0 * w_farm_true
        farm_errs.append(abs(f_pred_pct - f_true_pct))

        print(f"[{m['id']}] {m['stratum']}  ({m['timestamp_utc']} UTC)")
        print(f"  inflow: U={m['u_inf_ms']:.1f} m/s from {m['wind_dir_deg']:.0f} deg  TI={100*m['ti_inf']:.1f}%  sigma_wd={m['sigma_wd_deg']:.1f} deg  P_front={p_front:.0f} kW")
        print(f"  farm wake loss: WakeNet {f_pred_pct:+5.1f}% +- {f_sig_pct:4.1f}%  |  SCADA truth {f_true_pct:+5.1f}%")

        # The summary scores every waked downstream turbine, as the exam does; until 27 Nov 2026 it scored only the six
        # shown per snapshot (24 of 31 readings), which reversed WakeNet's ranking against tuned physics (audit).
        all_w = np.where(waked)[0]
        for idx in all_w:
            wl_p_a = 100.0 * (1.0 - float(r["eta"][idx])); wl_t_a = 100.0 * (1.0 - float(eta_true[idx]))
            waked_errs.append(abs(wl_p_a - wl_t_a))
            inside_68.append(abs(wl_p_a - wl_t_a) <= 100.0 * float(r["eta_sig"][idx]))
        # Show up to 6 waked downstream turbines (sorted by deepest predicted wake loss)
        w_idx = all_w[np.argsort(r["eta"][all_w])][:6]
        for idx in w_idx:
            code = m["codes"][idx]
            wl_p = 100.0 * (1.0 - float(r["eta"][idx]))
            wl_s = 100.0 * float(r["eta_sig"][idx])
            wl_t = 100.0 * (1.0 - float(eta_true[idx]))
            pk_p = float(r["power_kw"][idx])
            pk_s = float(r["power_sig_kw"][idx])
            pk_t = float(p_true[idx])
            yaw_i = float(z["yaw_deg"][idx])
            in_bar = abs(wl_p - wl_t) <= wl_s
            mark = "in 68%" if in_bar else "out"
            print(f"    {code:<5s}  loss {wl_p:+5.1f}% +- {wl_s:4.1f}% (truth {wl_t:+5.1f}%)  |  P {pk_p:6.0f} +- {pk_s:4.0f} kW (truth {pk_t:6.0f} kW)  yaw={yaw_i:+4.0f} deg [{mark}]")
        print()

    dt = time.perf_counter() - t0
    cov = 100.0 * float(np.mean(inside_68)) if inside_68 else 0.0
    print(f"Summary across {len(files)} exam snapshots ({len(waked_errs)} waked readings): waked-rotor MAE {np.mean(waked_errs):.1f}%  |  farm wake MAE {np.mean(farm_errs):.1f}%  |  68% bar coverage {cov:.0f}%  ({dt:.2f} s)")


if __name__ == "__main__":
    main()
