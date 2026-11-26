"""Reference annotations vs the detector in the same public classifier, on scored ECG interiors."""
import ast
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4] / "models/beatnet"
sys.path.insert(0, str(ROOT))
from beatnet import BeatNet


def matches(ref, detected, tol):
    # One-to-one nearest pairs, chronological indices as deterministic tie breakers.
    allowed = np.argwhere(np.abs(ref[:, None] - detected) <= tol)
    pairs = sorted((abs(ref[i] - detected[j]), i, j) for i, j in allowed)
    used_ref, used_det, result = set(), set(), []
    for _, i, j in pairs:
        if i not in used_ref and j not in used_det:
            used_ref.add(i)
            used_det.add(j)
            result.append((i, j))
    return result


def main():
    api = BeatNet()
    dev = set(json.loads((ROOT / "training/calib_v1.json").read_text())["dev_records"])
    common = ast.parse((ROOT / "training/common.py").read_text())
    exam = set(next(ast.literal_eval(n.value) for n in common.body if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "DS2"))
    # PhysioNet's MIT-BIH directory identifies 201 and 202 as one subject.
    subject = lambda rec: 201 if rec == 202 else rec
    overlap = {subject(r) for r in dev} & {subject(r) for r in exam}
    print("record overlap", dev & exam, "subject overlap (201/202 grouped)", overlap)
    assert not dev & exam and overlap == {201}
    total = np.zeros(6, dtype=int)
    for file in sorted((ROOT / "examples").glob("*.npz")):
        with np.load(file) as z:
            ecg, beats, truth, show = (z[k] for k in ("ecg", "beats", "ref", "show"))
            fs = json.loads(str(z["meta"]))["fs"]
        direct = api.classify(ecg, fs, beats)
        auto = api.classify(ecg, fs)
        inside = lambda pos: (pos >= show[0]) & (pos < show[1])
        r, det = inside(beats), inside(auto["beats"])
        ref, predicted = truth[r], auto["label"][det]
        pair = matches(beats[r], auto["beats"][det], .15 * fs)
        direct_hits = int(np.sum(direct["label"][r] == ref))
        auto_hits = sum(ref[i] == predicted[j] for i, j in pair)
        counts = [len(ref), len(predicted), len(pair), direct_hits, auto_hits, len(ref) - len(pair)]
        total += counts
        print(file.stem, "reference/detected/matched/label-correct(ref,auto)/missed", counts)
        missing = ecg.astype(float).copy()
        missing[len(missing) // 2] = np.nan
        damaged = api.classify(missing, fs, beats)
        bad = (~np.isfinite(damaged["probs"]).all(axis=1)) & r
        assert bad.sum() == r.sum() and np.all(damaged["label"][r] == "N")
        repaired = missing.copy()
        mid = len(missing) // 2
        repaired[mid] = (ecg[mid - 1] + ecg[mid + 1]) / 2
        fixed = api.classify(repaired, fs, beats)
        print("corrupted rows with S/V/F reference", int(np.sum(bad & np.isin(truth, list("SVF")))),
              "after interpolating just this sample, label agreement with original",
              np.sum(fixed["label"][r] == direct["label"][r]), "/", r.sum())
        print("one NaN, nonfinite probability rows", int(bad.sum()), "of", int(r.sum()),
              "returned classes", np.unique(damaged["label"][bad]).tolist())
    print("total", total.tolist())
    try:
        api.classify(np.zeros(3600), 360)
    except (IndexError, ValueError) as exc:
        print("flat trace, automatic mode", type(exc).__name__, str(exc))


if __name__ == "__main__":
    main()
