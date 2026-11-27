"""QuakeNet v1.2 public reader, source event split and archived rung decisions."""
import base64
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4] / "models"
MODEL = ROOT / "quakenet"
sys.path.insert(0, str(MODEL))
from quakenet import QuakeNet


def check_split():
    split = json.loads((ROOT / "quakenet-v2/splits.json").read_text())
    events = split["events"]
    dev = set(split["dev"])
    exam = set(split["exam"])
    assert set(events) == dev | exam and dev.isdisjoint(exam)
    drawn = np.random.default_rng(2026).choice(sorted(events), 40, replace=False)
    assert set(drawn) == dev
    assert len(dev) == 40 and len(exam) == 1734
    for ids, label in ((dev, "dev"), (exam, "exam")):
        assert all(events[k]["split"] == label for k in ids)
        print(label, len(ids), "events,", sum(len(events[k]["stations"]) for k in ids), "records")
    return exam


def check_ladder(ladder):
    # The archived web release rounds every value to three decimals.
    inc = None
    for row in ladder:
        if inc is None:
            keep = True
        else:
            keys = ("mw_mae", "dist_mae", "depth_med_km", "baz_med_deg")
            a = np.array([row["dev"][k] for k in keys])
            b = np.array([inc["dev"][k] for k in keys])
            half = 0.0005
            rel = np.mean(a / b)
            gain = row["objective"] - inc["objective"]
            keep = gain > 0.01 and rel <= 1.05
            lower = ((a - half) / (b + half)).mean()
            upper = ((a + half) / (b - half)).mean()
            print(row["ckpt"], "gain, dev ratio, dev rounding bounds", np.round([gain, rel, lower, upper], 5))
        assert keep == row["accepted"]
        if keep:
            inc = row


def main():
    net = QuakeNet()
    exam = check_split()
    page = (MODEL / "lab/index.html").read_text()
    data = json.JSONDecoder().raw_decode(page.split('<script id="data" type="application/json">', 1)[1])[0]
    check_ladder(data["ladder"])
    for i, path in enumerate(sorted((MODEL / "examples").glob("*.npz"))):
        with np.load(path) as z:
            x = z["zne"].astype(float)
            guess = json.loads(str(z["meta"]))["p_guess"]
        assert path.stem.split("_")[0] in exam
        base = net.read(x, guess)
        rot = x.copy()
        rot[1:] *= -1
        turned = net.read(rot, guess)
        residual = (turned["baz"] - base["baz"]) % 360 - 180
        broken = x.copy()
        broken[0, base["p_pick"]] = np.nan
        nan_out = net.read(broken, guess)
        assert all(np.isnan(nan_out[k]) for k in ("mw", "dist", "depth", "baz"))
        assert np.isnan(nan_out["p_strength"]) and turned["p_pick"] == base["p_pick"]
        rng = np.random.default_rng(740 + i)
        white = net.read(rng.normal(0, 1e-6, x.shape), guess)
        # A specimen of partial acquisition passed shortly after the expected P.
        partial = net.read(x[:, :guess + 61], guess)
        print(path.stem, "180-degree residual, original Mw, noise Mw/sigma, 60s prefix Mw/sigma, pick shift",
              np.round([residual, base["mw"], white["mw"], white["mw_sig"],
                        partial["mw"], partial["mw_sig"], partial["p_pick"] - base["p_pick"]], 3))
        assert partial["p_pick"] == base["p_pick"]
        assert abs(residual) < 3 and 5.5 < white["mw"] < 7 and partial["mw"] < base["mw"]
    for i, sample in enumerate(data["noise"]):
        # webdata.py stores six real quiet-time windows in velocity units.
        noise = np.frombuffer(base64.b64decode(sample["x"]), dtype="<i2").reshape(3, 2460)
        noise = noise.astype(float) * sample["q"] / 32767
        out = net.read(noise, 60)
        print("quiet station window", i, "Mw/sigma, dist, strength, pick",
              np.round([out[k] for k in ("mw", "mw_sig", "dist", "p_strength", "p_pick")], 3))
        assert 5 < out["mw"] < 6 and 0 < out["mw_sig"] < 0.2


if __name__ == "__main__":
    main()
