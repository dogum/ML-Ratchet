"""One point per frozen dev block from the four matched twin draws (optional Matplotlib)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _paths import work
from noise import read_result
from split import dev_cases
from twin import paired_draws
from stormnet import StormNet


if __name__ == "__main__":
    model = StormNet()
    center = read_result("calibration.json")["double"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), layout="constrained")
    fig.patch.set_facecolor("#f2eee4")
    for (year, d), color in zip(dev_cases().items(), ("#127967", "#a55720", "#67539b")):
        mask = np.isfinite(d["future_wind"]).all(axis=(1, 2))
        wind, truth, gids = d["wind"][mask], d["truth"][mask], d["group"][mask]
        full = np.concatenate((wind, d["future_wind"][mask]), axis=1)
        err, level = paired_draws(model, full, wind, center, year)
        real = model.predict(wind, d["dst"][mask])["min_dst"] - truth
        values = [[], []]
        for g in np.unique(gids):
            take = gids == g
            values[0].append([np.median(truth[take]), np.median(level[:, take])])
            values[1].append([np.abs(real[take]).mean(), np.abs(err[:, take]).mean()])
        for ax, points in zip(axes, values):
            a, b = np.array(points).T
            ax.scatter(a, b, label=year, color=color, s=25, alpha=.8, edgecolors="white", linewidth=.4)
    for ax, title, limits in zip(axes, ("Typical three-hour minimum", "Mean absolute forecast error"),
                                 ((-90, 15), (0, 16))):
        ax.set_facecolor("#fbf9f4")
        # Include the full observed range if a future source revision is deliberately validated.
        ranges = [*ax.get_xlim(), *ax.get_ylim(), *limits]
        limits = (min(ranges), max(ranges))
        ax.plot(limits, limits, "--", color="#938b79", lw=.9, zorder=0)
        ax.set(xlim=limits, ylim=limits, aspect="equal", xlabel="Real recording (nT)", ylabel="Simulated twin (nT)")
        ax.set_title(title, loc="left", fontsize=11, weight="semibold")
        ax.grid(alpha=.13)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(work("results", "dev-twins.png", mkdir=True), dpi=160, metadata={"Software": "StormNet"})
