"""Check independent analytic limits and a solve_ivp reference on a pressure jump and recovery."""
import numpy as np
from scipy.integrate import solve_ivp

from noise import HERE, write_json
from simulator import simulate
from physics import PUBLISHED, TWO_STATE


def check():
    # Fixed southward 8 nT at 500 km/s => E=4 mV/m; doubling pressure changes only the observation.
    wind = np.tile([-8., 500., 2.], (1, 96, 1))
    wind[:, 32:, 0] = 3.
    wind[:, 64:, 2] = 8.
    out = {}
    for name, par in (("published", PUBLISHED), ("two_state", TWO_STATE)):
        gain, tf, ts, share, b, c = par
        got = simulate(wind, par, np.random.default_rng(0), scatter=False)[0]
        history = []
        state = np.zeros(2)
        for bz, speed, pressure in wind[0]:
            electric = max(0., -bz * speed * 0.001)
            tau = np.array([tf, ts]) * 2.4 * np.exp(9.74 / (4.69 + electric))
            source = -gain * max(electric - 0.49, 0) * np.array([share, 1 - share])
            solved = solve_ivp(lambda t, y: source - y / tau, [0., 1.], state,
                               method="DOP853", rtol=1e-11, atol=1e-12)
            state = solved.y[:, -1]
            history.append(state.sum() + b * np.sqrt(pressure) - c)
        tau = np.array([tf, ts]) * 2.4 * np.exp(9.74 / 8.69)
        expected = np.sum(-gain * 3.51 * np.array([share, 1 - share]) * tau *
                          (1 - np.exp(-32 / tau))) + b * np.sqrt(2.) - c
        out[name] = dict(ivp_max_nT=float(np.max(np.abs(got - history))),
                         analytic_32h_step_nT=float(abs(got[31] - expected)))
        assert max(out[name].values()) < 1e-8
    quiet = np.tile([4., 450., 2.], (1, 2, 1))
    quiet[:, 1, 2] = 8.
    y = simulate(quiet, PUBLISHED, np.random.default_rng(0), scatter=False)[0]
    out["quiet_pressure_jump_nT"] = float(y[1] - y[0])
    assert np.isclose(y[1] - y[0], 7.26 * (np.sqrt(8.) - np.sqrt(2.)))
    return out


if __name__ == "__main__":
    result = check()
    write_json(HERE / "physics_check.json", result)
    print(result)
