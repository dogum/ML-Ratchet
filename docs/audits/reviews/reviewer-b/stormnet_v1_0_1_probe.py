"""Forecast availability, interval bins and latency sensitivity on the six archived storm replays."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

MODEL = Path(__file__).resolve().parents[4] / "models/stormnet"
sys.path[:0] = [str(MODEL), str(MODEL / "training")]
import split
from noise import fill_wind
from stormnet import features, StormNet


def ladder():
    current = yaml.safe_load((MODEL / "ratchet.yaml").read_text())["ladders"][0]
    original = json.loads((MODEL / "training/ladder_history.json").read_text())["original"]
    for log in (original, current):
        best = None
        for row in log["rungs"]:
            keep = best is None or (row["sim"] > best["sim"] + .01 and
                                    all(v <= 1.05 * best["dev"][k] for k, v in row["dev"].items()))
            assert keep == (row["decision"] in {"released", "kept"}), row["id"]
            if keep:
                best = row
        print("ladder", log["scored_on"], len(log["rungs"]), best["id"])


def cases(time, wind, dst):
    with patch.object(split, "read_year", return_value=(time, wind, dst, "fixture")):
        return split.period([2024])[0]


def main():
    ladder()
    model = StormNet()
    w = model.weights
    assert all(np.isfinite(a).all() for a in w.values()) and (w["scale"] > 0).all()
    assert (np.linalg.norm(w["kernel"], axis=0) > 1e-4).all()
    examples = MODEL / "examples"
    meta = json.loads((examples / "storms.json").read_text())
    results = []
    scaled_features = []
    with np.load(examples / "storms.npz") as archive:
        for i, m in enumerate(meta):
            wind, dst = (archive[f"{name}{i}"].astype(float) for name in ("wind", "dst"))
            time = np.datetime64(m["start_utc"][:-1], "h") + np.arange(len(wind))
            # Gallery wind has already been forward-filled upstream. Preserve its archived gaps.
            idx = np.arange(36, len(dst) - 3)
            x = np.stack([wind[k-35:k+1] for k in idx])
            fut = np.stack([dst[k+1:k+4] for k in idx])
            ok = np.isfinite(x).all(axis=(1, 2)) & np.isfinite(fut).all(axis=1) & np.isfinite(dst[idx-1]) & np.isfinite(dst[idx])
            idx, x, truth = idx[ok], x[ok], fut[ok].min(axis=1)
            now = dst[idx]
            raw = model.predict(x, now, calibrated=False)
            pred = model.predict(x, now)
            delayed = model.predict(x, dst[idx-1])
            # Independent reconstruction of the random-feature ridge readout from its exported coefficients.
            f, anchor = features(x, now, model.meta["mode"], model.meta["physics"], model.meta["wind_law"])
            z = np.clip((f - w["mean"]) / w["scale"], -8, 8)
            scaled_features.append(z)
            h = np.tanh(z @ w["kernel"] + w["bias"])
            loc = w["location"]
            rebuilt = anchor + loc[0] + z @ loc[1:1+z.shape[1]] + h @ loc[1+z.shape[1]:]
            assert np.allclose(rebuilt, pred["min_dst"], rtol=0, atol=1e-10)
            spread = w["spread"]
            sigma = np.exp(np.clip(spread[0] + z @ spread[1:1+z.shape[1]] + h @ spread[1+z.shape[1]:], -1, 5))
            np.testing.assert_allclose(raw["sigma"], sigma, rtol=1e-10)
            factors = np.array(model.meta["widen_by_dst"]["factors"])
            edges = model.meta["widen_by_dst"]["edges"]
            doc = raw["sigma"] * factors[np.searchsorted(edges, now, side="left")]
            results.append(np.column_stack((truth, now, pred["min_dst"], pred["sigma"], delayed["min_dst"], doc)))
            # Exercise the actual exam slicer with a known one-channel gap. A gap of length 3 blocks issue.
            k = 42
            g = wind.copy()
            g[k-1:k+1, 0] = np.nan
            assert fill_wind(g)[k, 0] == g[k-2, 0]
            g[k-2, 0] = np.nan
            assert np.isnan(fill_wind(g)[k, 0])
            assert k not in cases(time, g, dst)["row"]
            g = wind.copy()
            g[k-1:k+1, 0] = np.nan
            other = g.copy()
            other[k+1:, 0] += 20
            d2 = dst.copy()
            d2[k+1:] += 30
            rows = [cases(time, a, b) for a, b in ((g, dst), (other, d2))]
            original, changed = (model.predict(c["wind"][c["row"] == k], c["dst"][c["row"] == k]) for c in rows)
            assert original["min_dst"].size == 1
            for key in original:
                np.testing.assert_array_equal(original[key], changed[key])
            assert rows[0]["time"][rows[0]["row"] == k][0] == time[k] + np.timedelta64(1, "h")
            assert rows[1]["truth"][rows[1]["row"] == k][0] == np.min(dst[k+1:k+4]) + 30
            print(m["label"], len(idx), "hours")
    truth, now, point, sigma, lag, documented = np.concatenate(results).T
    z = np.concatenate(scaled_features)
    print("normalized feature std range / clipped values", z.std(0).min(), z.std(0).max(), np.sum(abs(z) == 8))
    for label, take in (("all", np.ones(len(now), bool)), ("storm targets", truth <= -50), ("extreme targets", truth <= -150)):
        print(label, take.sum(), "MAE nT original / stale Dst, coverage original / prose bins",
              np.mean(abs(truth[take] - point[take])), np.mean(abs(truth[take] - lag[take])),
              np.mean(abs(truth[take] - point[take]) <= sigma[take]), np.mean(abs(truth[take] - point[take]) <= documented[take]))
    print("edge counts -50 / -30", (now == -50).sum(), (now == -30).sum(), "max change to width nT",
          max(abs(sigma - documented)), "max change to median with stale Dst nT", max(abs(point-lag)))
    print("changed coverage outcomes", np.sum((abs(truth-point) <= sigma) != (abs(truth-point) <= documented)))


if __name__ == "__main__":
    main()
