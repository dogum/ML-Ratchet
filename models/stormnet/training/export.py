"""Draw the public replay by stratum and independent block before looking at forecast errors."""
from pathlib import Path
import numpy as np
from noise import PROTOCOL, write_json


def examples(cases, meta, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(PROTOCOL["example_seed"])
    truth = cases["truth"]
    bands = [("Extreme", -np.inf, -150), ("Intense", -150, -100), ("Storm", -100, -50),
             ("Quiet", -50, np.inf)]
    selected, excluded = [], set()
    for label, lo, hi in bands:
        eligible = (truth > lo) & (truth <= hi)
        options = sorted(set(cases["group"][eligible]) - excluded)
        if not options:
            continue
        group = int(rng.choice(options))
        i = int(rng.choice(np.flatnonzero(eligible & (cases["group"] == group))))
        excluded.add(group)
        j = i
        while (i - j < 6 and j > 0 and cases["group"][j - 1] == group and
               cases["row"][j] - cases["row"][j - 1] == 1):
            j -= 1
        frames = slice(j, i + 1)
        filename = label.lower() + ".npz"
        np.savez_compressed(dest / filename, wind=cases["wind"][frames].astype("float32"),
                            last_dst=cases["dst"][frames], past_dst=cases["past_dst"][frames],
                            future_dst=cases["future"][frames])
        selected.append(dict(label=label, file=filename, block=meta[group]["id"],
                             issues=[str(t) + ":00Z" for t in cases["time"][frames]]))
    write_json(dest / "records.json", selected)
