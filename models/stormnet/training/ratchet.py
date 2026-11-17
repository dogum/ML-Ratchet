"""Judge the original ladder, replay on changed physics, then freeze the released readout."""
import json
from pathlib import Path

import numpy as np
import yaml
from mlratchet import Ratchet, gates
from mlratchet.calibrate import widen_factor

from _paths import MODEL, work
from noise import HERE, PROTOCOL, write_json
from simbank import load_bank
from split import dev_cases
from train import RUNS
from stormnet import StormNet


def error_ratio(data, prediction):
    truth = data["truth"]
    active = truth <= -50
    err, base = abs(prediction - truth), abs(data["dst"] - truth)
    if not active.any() or min(base.mean(), base[active].mean()) <= 0:
        raise ValueError("Both gate strata require a finite nonzero persistence error")
    return float(0.5 * (err.mean() / base.mean() + err[active].mean() / base[active].mean()))


def main():
    dev = dev_cases()
    val = [load_bank(name) for name in ("val1", "val2")]
    r = Ratchet("stormnet", frozen_set="val1", margin=PROTOCOL["margin"],
                dev_gate=gates.no_worse_than(PROTOCOL["dev_max_ratio"], keys=list(dev)),
                objective="Mean MAE skill against persistence on all issues and active targets (Dst ≤ −50 nT)",
                rule="Gain > 0.01 on frozen skill and no dev year's mean error ratio more than 5% worse.")
    scores, dev_scores = {}, {}
    for rid, _, _, change in RUNS:
        model = StormNet(work("ckpt", rid + ".npz"))

        def read(d):
            return model.predict(d["wind"], d["dst"])["min_dst"]

        scores[rid] = [1 - error_ratio(d, read(d)) for d in val]
        dev_scores[rid] = {yr: error_ratio(d, read(d)) for yr, d in dev.items()}
        if rid in ("r1", "r2"):
            r.judge(rid, change, scores[rid][0], dev_scores[rid])
    initial = r.to_dict()
    flips = r.rescore({rid: s[1] for rid, s in scores.items()}, "val2")
    for rid, _, _, change in RUNS[2:]:
        r.judge(rid, change, scores[rid][1], dev_scores[rid])
    r.release()
    chosen = r.incumbent.id
    ladder = r.to_dict()
    ladder.update(frozen_sets=["val1", "val2"], dev_gate="2008, 2011 and 2014 gated separately.",
                  columns=list(dev), columns_source="Dev error ratios to persistence (lower is better)")
    for entry in ladder["rungs"]:
        entry["metrics"] = [round(entry["dev"][yr], 4) for yr in dev]
    (Path(MODEL) / "ratchet.yaml").write_text(yaml.safe_dump({"model": "stormnet", "ladders": [ladder]},
                                                             sort_keys=False, allow_unicode=True))
    model = StormNet(work("ckpt", chosen + ".npz"))
    residuals, sig = [], []
    for d in dev.values():
        pred = model.predict(d["wind"], d["dst"])
        residuals.extend(pred["min_dst"] - d["truth"])
        sig.extend(pred["sigma"])
    model.meta["widen_factor"] = widen_factor(residuals, sig)
    model.meta["physics_version"] = "ring-2" if chosen in ("r3", "r4") else "ring-1"
    weights = {k: v.astype("float32") for k, v in model.weights.items()}
    release = Path(MODEL) / "stormnet_weights.npz"
    np.savez_compressed(release, **weights, meta=json.dumps(model.meta))
    exported = StormNet(release)
    checks = {}
    for d in dev.values():
        a, b = model.predict(d["wind"], d["dst"]), exported.predict(d["wind"], d["dst"])
        for key in a:
            checks[key] = max(checks.get(key, 0), float(np.max(abs(a[key] - b[key]))))
    if max(checks.values()) > 1e-4:
        raise ValueError(f"Float32 export drift: {checks}")
    write_json(HERE / "ladder_history.json", dict(original=initial, replay_flips=flips, scores=scores,
               dev=dev_scores, released=chosen, widen_factor=model.meta["widen_factor"],
               parameters=sum(v.size for v in weights.values()), export_max_abs_nT=checks))
    print(ladder, model.meta["widen_factor"], checks)


if __name__ == "__main__":
    main()
