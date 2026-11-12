"""The ratchet: keep a new rung only if it beats the incumbent on a frozen simulated set by more
than a stated margin AND passes a gate on a small real dev set. Log every rung, kept or not.

This is the rule every model in the collection was built with (docs/method.md). A minimal use:

    from mlratchet import Ratchet, gates

    r = Ratchet("quakenet", margin=0.01, dev_gate=gates.no_worse_than(1.05),
                objective="mean skill over Mw, distance, log depth, back-azimuth", frozen_set="val8")
    r.judge("r1", "textbook ak135", sim=0.556, dev={"mw": 0.20, "dist": 6.1})
    r.judge("r2", "slower surface waves", sim=0.690, dev={"mw": 0.16, "dist": 5.0})
    r.save("ratchet.yaml")

When the physics changes, build a new frozen set, score every rung on it, and replay the
decisions in order with :meth:`Ratchet.rescore` — a rung rejected before may be kept now, and
the other way round.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import yaml

DevMetrics = Mapping[str, float]
Gate = Callable[[DevMetrics, DevMetrics], "tuple[bool, str]"]


@dataclass
class Rung:
    id: str
    change: str
    sim: float
    dev: dict | None = None
    decision: str = "pending"  # kept | rejected | released
    why: str = ""
    metrics: list | None = None  # optional columns reported alongside (e.g. the real exam, never used)

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if v not in (None, "", [])}
        return d


class gates:
    """Ready-made dev gates. Each returns ``gate(candidate_dev, incumbent_dev) -> (ok, reason)``."""

    @staticmethod
    def no_worse_than(max_ratio: float = 1.05, keys: list[str] | None = None, lower_is_better: bool = True) -> Gate:
        """Every dev error may grow by at most ``max_ratio`` (1.05 = 5% worse).

        Used by QuakeNet and SwellCast.
        """

        def gate(cand: DevMetrics, inc: DevMetrics):
            ks = keys or [k for k in cand if k in inc]
            bad = []
            for k in ks:
                c, i = float(cand[k]), float(inc[k])
                worse = c > i * max_ratio if lower_is_better else c < i / max_ratio
                if worse:
                    bad.append(f"{k} {c:.4g} vs {i:.4g}")
            return (not bad, "dev gate passed" if not bad else "dev gate: " + "; ".join(bad))

        return gate

    @staticmethod
    def mean_ratio_at_most(max_mean_ratio: float = 1.10, keys: list[str] | None = None) -> Gate:
        """The mean of candidate/incumbent dev-error ratios may be at most ``max_mean_ratio``.

        Used by StarQuake (four dev median errors, ≤ 1.10).
        """

        def gate(cand: DevMetrics, inc: DevMetrics):
            ks = keys or [k for k in cand if k in inc]
            ratios = [float(cand[k]) / float(inc[k]) for k in ks if float(inc[k]) != 0]
            m = sum(ratios) / len(ratios) if ratios else 1.0
            return (m <= max_mean_ratio, f"dev error ratio {m:.3f} (limit {max_mean_ratio})")

        return gate

    @staticmethod
    def max_drop(key: str, max_drop: float) -> Gate:
        """A higher-is-better dev score may drop by at most ``max_drop``.

        Used by BeatNet (dev S/V F1 may drop ≤ 0.02).
        """

        def gate(cand: DevMetrics, inc: DevMetrics):
            d = float(cand[key]) - float(inc[key])
            return (d >= -max_drop, f"dev {key} {float(cand[key]):.4g} vs {float(inc[key]):.4g} ({d:+.4g})")

        return gate

    @staticmethod
    def all_of(*gs: Gate) -> Gate:
        def gate(cand: DevMetrics, inc: DevMetrics):
            reasons, ok = [], True
            for g in gs:
                o, r = g(cand, inc)
                ok &= o
                reasons.append(r)
            return ok, "; ".join(reasons)

        return gate


@dataclass
class Ratchet:
    """A ladder of rungs judged in order against the current incumbent."""

    model: str
    margin: float = 0.01
    dev_gate: Gate | None = None
    higher_is_better: bool = True
    objective: str = ""
    rule: str = ""
    frozen_set: str = ""
    name: str = ""
    rungs: list[Rung] = field(default_factory=list)

    @property
    def incumbent(self) -> Rung | None:
        for r in reversed(self.rungs):
            if r.decision in ("kept", "released"):
                return r
        return None

    def judge(self, id: str, change: str, sim: float, dev: DevMetrics | None = None,
              metrics: list | None = None) -> Rung:
        """Judge a candidate rung against the incumbent, record it, and return it."""
        inc = self.incumbent
        rung = Rung(id=id, change=change, sim=float(sim), dev=dict(dev) if dev else None, metrics=metrics)
        if inc is None:
            rung.decision, rung.why = "kept", "First rung."
        else:
            gain = (sim - inc.sim) if self.higher_is_better else (inc.sim - sim)
            if gain <= self.margin + 1e-9:  # "more than the margin", robust to float rounding
                rung.decision = "rejected"
                rung.why = f"{gain:+.3f} on {self.frozen_set or 'the frozen set'}, needs more than +{self.margin}."
            elif self.dev_gate is not None and dev is not None and inc.dev is not None:
                ok, reason = self.dev_gate(dev, inc.dev)
                rung.decision = "kept" if ok else "rejected"
                rung.why = f"{gain:+.3f} on the frozen set; {reason}."
            else:
                rung.decision, rung.why = "kept", f"{gain:+.3f} on the frozen set."
        self.rungs.append(rung)
        return rung

    def release(self, id: str | None = None) -> Rung:
        """Mark the incumbent (or a given kept rung) as the released one."""
        target = next((r for r in self.rungs if r.id == id), None) if id else self.incumbent
        if target is None or target.decision not in ("kept", "released"):
            raise ValueError("Only a kept rung can be released.")
        for r in self.rungs:
            if r.decision == "released":
                r.decision = "kept"
        target.decision = "released"
        return target

    def rescore(self, sim_scores: Mapping[str, float], frozen_set: str) -> list[tuple[str, str, str]]:
        """Replay every decision, in order, with scores from a NEW frozen set.

        Returns ``(rung id, old decision, new decision)`` for every rung whose fate changed.
        Dev metrics are reused; only the simulated scores change.
        """
        old = [(r.id, r.decision) for r in self.rungs]
        released = next((r.id for r in self.rungs if r.decision == "released"), None)
        replay = Ratchet(self.model, self.margin, self.dev_gate, self.higher_is_better,
                         self.objective, self.rule, frozen_set, self.name)
        for r in self.rungs:
            replay.judge(r.id, r.change, sim_scores[r.id], r.dev, r.metrics)
        if released and any(r.id == released and r.decision == "kept" for r in replay.rungs):
            replay.release(released)
        norm = {"released": "kept"}
        changed = [(i, a, b.decision) for (i, a), b in zip(old, replay.rungs)
                   if norm.get(a, a) != norm.get(b.decision, b.decision)]
        self.rungs, self.frozen_set = replay.rungs, frozen_set
        return changed

    # ---- persistence: the same ratchet.yaml format every model folder uses -----------------
    def to_dict(self) -> dict:
        return {
            "name": self.name or self.model,
            "objective": self.objective,
            "rule": self.rule or f"Keep if the frozen-set objective beats the incumbent by more than {self.margin}"
                                 + (" AND the dev gate passes." if self.dev_gate else "."),
            "margin": self.margin,
            "scored_on": self.frozen_set,
            "rungs": [r.to_dict() for r in self.rungs],
        }

    def save(self, path: str | Path) -> None:
        doc = {"model": self.model, "ladders": [self.to_dict()]}
        Path(path).write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
