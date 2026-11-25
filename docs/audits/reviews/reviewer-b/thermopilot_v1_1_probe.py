"""Probe ThermoPilot's frozen controller and the separate exam-time card construction."""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
MODEL = ROOT / "models/thermopilot"
sys.path[:0] = [
    str(MODEL),
    str(MODEL / "training"),
    str(ROOT / "shared/building/buildingsim"),
]
import thermopilot_core as tp
from tp.features import full_card
from tp.mpc import Model
from buildingsim import Simulator
from buildingsim.weather import SIGMA, clear_cache


def fixture():
    # Fixed nominal house and known tariff. Only actual wind at hours >=48 varies.
    idx = pd.date_range("2019-01-14", periods=24 * 12 + 1, freq="h", tz="UTC")
    h = np.arange(len(idx))
    T = -3 + 7 * np.sin(2 * np.pi * h / 24)
    G = 250 * np.maximum(0, np.sin(np.pi * (h % 24 - 6) / 12))
    df = pd.DataFrame(
        dict(
            T_out=T,
            RH=70,
            GHI=G,
            DNI=G,
            DHI=G * 0.3,
            wind_speed=2.0,
            wind_dir=180.0,
            pressure=101325.0,
            IR_horiz=SIGMA * (T + 273.15) ** 4 * 0.8,
            cloud=0.5,
        ),
        index=idx,
    )
    meta = dict(lat=41.9, lon=-87.6, elev=200.0, tz="UTC")
    spec = dict(
        primary_use="residential_single",
        floor_area_m2=150,
        year_built=2000,
        climate_zone="5A",
        system="heat_pump_air",
        ventilation=dict(natural_ventilation=False),
    )
    ctx = tp.make_ctx(
        idx[:-1],
        np.where((h[:-1] % 24 >= 15) & (h[:-1] % 24 < 20), 0.48, 0.10),
        np.full(len(h) - 1, 20.0),
        np.full(len(h) - 1, 25.0),
        20.0,
        25.0,
        np.zeros(len(h) - 1),
    )
    models = []
    clear_cache()
    for wind, fresh in ((2.0, False), (8.0, False), (8.0, True)):
        if fresh:
            clear_cache()
        weather = df.copy()
        weather.loc[idx[48] :, "wind_speed"] = wind
        sim = Simulator(
            spec,
            (weather, meta),
            idx[0],
            idx[-1],
            dt="1h",
            warmup_days=0,
            record_trend=False,
        )
        b = Model(sim)
        models.append((b, full_card(b, ctx, 25.0)))
    base, original = models[0]
    print(
        "mean squared wind term after hour 48, low / cached high / fresh high",
        *[b.CwU2[48:].mean() for b, _ in models],
    )
    np.testing.assert_array_equal(base.CwU2, models[1][0].CwU2)
    assert not np.array_equal(base.CwU2[48:], models[2][0].CwU2[48:])
    changed_map = models[2][1]["map"]
    # Isolate the card-map path. Both controllers see identical forecasts and
    # initial states. Freeze all other card fields, including sizing and base load.
    cards = [original, dict(original, map=changed_map)]
    print("At coefficient, base / future-wind card", *[c["map"]["At"] for c in cards])
    changed = 0
    largest = 0.0
    example = None
    for hour in range(24):
        for tz in (19.75, 20.5, 21.25):
            actions = []
            for card in cards:
                p = tp.pilot(card, ctx, 3.0, base.base_W)
                actions.append(
                    p.step(hour, tz, 0.0, 0.0, T[hour : hour + 24], G[hour : hour + 24])
                )
            shift = max(abs(actions[1][k] - actions[0][k]) for k in actions[0])
            if shift:
                changed += 1
                if shift > largest:
                    largest = shift
                    example = (hour, tz, actions)
    print(
        "matched fresh starts with changed setpoints / 72, largest K, example",
        changed,
        largest,
        example,
    )


def packaged():
    z = np.load(MODEL / "examples/bdg2/demo_episodes.npz")
    W = tp.load(MODEL / "thermopilot_v1.npz")
    assert all(np.isfinite(a).all() for a in W.values())
    print("v1 feature scales", W["sd"].min(), W["sd"].max())
    for i, m in enumerate(json.loads(str(z["meta"]))):
        p = f"ep{i}_"

        def subset(prefix):
            return {
                k.removeprefix(p + prefix): z[k] for k in z if k.startswith(p + prefix)
            }

        ctx = dict(subset("ctx_"), pref_lo=m["pref_lo"], pref_hi=m["pref_hi"])
        card = dict(m["card"], **subset("card_"))
        r = subset("rec_")
        base = z[p + "base_W"]
        pilot = tp.pilot(card, ctx, m["lam"], base, m["gas_price"])
        feats = tp.Featurizer(card, ctx, m["lam"], base)
        vectors = []
        same = 0
        for k, t in enumerate(r["T"]):
            args = (k, t, r["q"][k], r["e"][k], r["fT"][k], r["fG"][k])
            sp = pilot.step(*args)
            same += np.allclose(
                [sp["heat_sp"], sp["cool_sp"]],
                [r["h"][k], r["c"][k]],
                atol=1e-6,
                rtol=0,
            )
            vectors.append(feats.step(*args))
        X = np.stack(vectors)
        # v1.0 optional network. Match the training ReLU MLP on its actual inputs.
        standardized = (X - W["mu"]) / W["sd"]
        print("max absolute standardized network input", np.abs(standardized).max())
        h = torch.tensor(standardized, dtype=torch.float32)
        for j in range(3):
            h = torch.nn.functional.linear(
                h,
                torch.tensor(W[f"W{j}"].T, dtype=torch.float32),
                torch.tensor(W[f"b{j}"], dtype=torch.float32),
            )
            if j < 2:
                h = h.relu()
        print(
            m["id"],
            "v1.1 replay",
            same,
            len(X),
            "v1.0 parity",
            np.abs(h.numpy() - tp.mlp_forward(W, X)).max(),
        )
        print(
            "float32 vs release argmax disagreements",
            int((h.argmax(1).numpy() != tp.mlp_forward(W, X).argmax(1)).sum()),
        )
        assert same == len(X)

        # Reprice the recorded trajectory. Both policies get the same price
        # at k and identical past observations. Only leads 1..23 differ.
        prices = pd.read_csv(
            ROOT / "models/batterypilot/examples/ERCOT_HB_HOUSTON_2023.csv"
        )
        prices = prices[["rt", "da"]].iloc[: len(ctx["price"])].to_numpy().T
        rt, da = np.clip(prices / 1000.0, -0.05, 3.0) + 0.10
        tariff = dict(ctx, price=rt, drate=np.zeros_like(rt), frate=np.zeros_like(rt))
        issued = dict(tariff, price=da.copy())
        oracle, day_ahead = [
            tp.pilot(card, c, m["lam"], base, m["gas_price"]) for c in (tariff, issued)
        ]
        changed = 0
        for k, t in enumerate(r["T"]):
            issued["price"][k] = rt[k]
            args = (k, t, r["q"][k], r["e"][k], r["fT"][k], r["fG"][k])
            changed += oracle.step(*args) != day_ahead.step(*args)
        print("Houston repricing, RT vs DA future setpoint changes", changed, len(X))

        def first(c, b, q, e):
            agent = tp.pilot(card, c, m["lam"], b, m["gas_price"])
            sp = agent.step(0, r["T"][0], q, e, r["fT"][0], r["fG"][0])
            return sp, agent.Q

        c = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in ctx.items()}
        c["price"][24:] += 1
        b = base.copy()
        b[24:] *= 2
        s0, q0 = first(ctx, base, 0.0, 0.0)
        for s1, q1 in [first(c, b, 0.0, 0.0), first(ctx, base, 5000.0, 15.0)]:
            assert s1 == s0 and np.array_equal(q1, q0)


def ladder():
    for f in ("ratchet.json", "ratchet_r8.json"):
        rows = json.loads((MODEL / "training" / f).read_text())["rungs"]
        iv = idv = 0.0
        anchor = next((r["val"]["skill"] for r in rows if r["tag"] == "card4"), None)
        for r in rows:
            v, d = r["val"]["skill"], r["dev"]
            if f == "ratchet_r8.json" and r["tag"] in ("r7", "mpc_c", "card0"):
                keep = r["tag"] == "r7"
            else:
                keep = (
                    v > iv + 0.01
                    and d["skill"] > 0
                    and d["skill"] >= idv - 0.03
                    and d["DH_pol"] <= d["DH_fixed"] + 1
                )
                if r["tag"].startswith("r8_s"):
                    keep = keep and v > anchor + 0.01
            assert keep == r["kept"], r["tag"]
            if keep:
                iv, idv = v, d["skill"]
        print(f, len(rows), "rows replayed")


if __name__ == "__main__":
    torch.set_num_threads(1)
    fixture()
    packaged()
    ladder()
