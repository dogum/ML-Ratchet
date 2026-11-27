"""Blind-wind controls on public raw fires and a census of the archived exam IDs."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MODEL = ROOT / "models/firedetective"
sys.path.insert(0, str(MODEL))
from demo import load_fire
from firedetective import FireDetective, load
import pipeline


def main():
    model = load()
    split = json.loads((ROOT / "shared/fire/data/proc/splits.json").read_text())["fires"]
    exam = json.loads((MODEL / "training/results/exam.json").read_text())
    scored = {row["fid"]: row for row in exam["fires"]}
    assert len(scored) == len(exam["fires"]) == exam["n_records"] == 944
    assert not set(split["dev"]).intersection(split["exam"]) and set(scored) <= set(split["exam"])
    print("unique dev, exam-pool, scored", len(set(split["dev"])), len(set(split["exam"])), len(scored))

    # An exact change of units in the first affine layer tests whether the
    # scale alone determines predictions from the released weights.
    weights = {k: v.copy() for k, v in model.W.items()}
    scale = weights["seq_sd"][24]
    for key in weights:
        if key.endswith(".emb.weight"):
            weights[key][:, 24] /= scale
    weights["seq_sd"][24] = 1
    matched = FireDetective(weights, model.cal)
    for file in sorted((MODEL / "examples").glob("*.npz")):
        fire, meta = load_fire(file)
        seq, hid, static = [x[None].astype(float) for x in pipeline.features(**fire)]
        public = model.predict(seq, hid, static)
        assert abs(public["wind"].item() - scored[file.stem]["wind_pred"]) < 0.006
        for key, val in matched.predict(seq, hid, static).items():
            np.testing.assert_allclose(val, public[key], atol=1e-12, rtol=1e-12)
        # Remove direct wind fields at the source, while keeping all other
        # measurements, derived fire geometry and weather fields unchanged.
        changed = dict(fire, wx=[dict(w, vs=0.0, th=0.0) for w in fire["wx"]])
        visible2, hidden2, static2 = pipeline.features(**changed)
        assert not np.array_equal(seq[0], visible2)
        np.testing.assert_array_equal(hid[0], hidden2)
        np.testing.assert_array_equal(static[0], static2)

        def wind(hidden):
            return model.predict(seq, hidden, static)["wind"].item()

        observed = public["wind"].item()
        flag_small = hid.copy()
        flag_small[..., 24] *= scale
        no_flag = hid.copy()
        no_flag[..., 24] = 0
        # Remove one valid step at a time, retaining the encoded static context.
        omission = []
        for step in np.flatnonzero(hid[0, :, 0]):
            one = hid.copy()
            one[:, step] = 0
            omission.append(wind(one) - observed)
        print(meta["name"], "truth / original / flag z=1 / no flag / shown",
              [round(x, 4) for x in (meta["truth"]["wind"], observed, wind(flag_small), wind(no_flag), wind(seq))],
              "range of one-step omissions", np.round([min(omission), max(omission)], 4))
        assert 1.7 < wind(flag_small) < 2 and observed > wind(flag_small)
        assert min(omission) != max(omission)
        # No observations, but keep valid static attributes for this site.
        with np.errstate(over="ignore", invalid="ignore"):
            empty = model.predict(np.zeros_like(seq), np.zeros_like(hid), static)
        assert np.isposinf(empty["wind"].item())
        without_steps = dict(fire, masks=fire["masks"][:0], day=[], wx=[], dets=[])
        with np.errstate(over="ignore", invalid="ignore"):
            raw_empty = pipeline.run([without_steps], model)
        assert np.isposinf(raw_empty["wind"].item())
        print("no steps, retained static / rebuilt static LFMC",
              empty["lfmc"].item(), raw_empty["lfmc"].item())


if __name__ == "__main__":
    main()
