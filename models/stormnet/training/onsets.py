"""Retrospective dev-only warnings before the first -50 nT crossing in each selected storm."""
from statistics import NormalDist

import numpy as np

from baselines import NAMES, forecast
from noise import RESULTS, read_result, read_year, write_json
from split import dev_cases
from stormnet import StormNet

HOUR = np.timedelta64(1, "h")


def wilson(hits, total):
    if not total:
        return None
    z = NormalDist().inv_cdf(.975)
    p = hits / total
    denom = 1 + z**2 / total
    midpoint = (p + z**2 / (2 * total)) / denom
    spread = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return [max(0., midpoint - spread), min(1., midpoint + spread)]


def main():
    selected = read_result("split.json")["dev_blocks"]
    cal = read_result("calibration.json")
    model = StormNet()
    events, excluded = [], []
    for year, cases in dev_cases().items():
        clock, _, dst, _ = read_year(int(year), index=True)
        forecasts = {"StormNet": model.predict(cases["wind"], cases["dst"])["min_dst"], **forecast(cases, cal)}
        times = cases["time"]
        for block in selected[year]:
            if block["kind"] != "storm":
                continue
            within = (clock >= np.datetime64(block["start"])) & (clock < np.datetime64(block["end_exclusive"]))
            crossing = np.flatnonzero(within & (dst <= -50))[0]
            if crossing < 1 or not np.isfinite(dst[crossing - 1]) or dst[crossing - 1] <= -50:
                excluded.append({"block": block["id"], "reason": "No confirmed first crossing"})
                continue
            end = clock[crossing] + HOUR
            issue_times = end - np.array([3, 2, 1]) * HOUR
            indices = np.searchsorted(times, issue_times)
            if (indices >= len(times)).any() or not np.array_equal(times[indices], issue_times):
                excluded.append({"block": block["id"], "reason": "Incomplete three-issue history"})
                continue
            if np.any(cases["dst"][indices] <= -50):
                excluded.append({"block": block["id"], "reason": "A forecast already saw storm-strength Dst"})
                continue
            leads = (end - issue_times) // HOUR
            alerts = {name: int(np.max(leads[p[indices] <= -50], initial=0)) for name, p in forecasts.items()}
            events.append({"block": block["id"], "crossing_hour_end_utc": str(end), "warning_lead_hours": alerts})

    summary = {}
    for name in ("StormNet", *NAMES):
        lead = [e["warning_lead_hours"][name] for e in events if e["warning_lead_hours"][name] > 0]
        summary[name] = dict(hits=len(lead), storms=len(events), hit_fraction_ci95=wilson(len(lead), len(events)),
                             median_lead_h_on_hits=float(np.median(lead)) if lead else None)
    write_json(RESULTS / "onsets_dev.json", dict(summary=summary, excluded=excluded, events=events))


if __name__ == "__main__":
    main()
