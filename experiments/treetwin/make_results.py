"""Aggregate per-platform TreeTwin + ConformalRatchet exam scores into results.json and RESULTS.md."""
from __future__ import annotations

import json

import tt_paths as P

PLATFORMS = ("buildingdetective", "airfoilnet", "energynet")


def build_results() -> dict:
    scores = {}
    for p in PLATFORMS:
        fp = P.HERE / "scores" / f"{p}.json"
        if not fp.exists():
            raise FileNotFoundError(f"missing exam score file: {fp}")
        scores[p] = json.loads(fp.read_text(encoding="utf-8"))

    summary = {
        "experiment": "TreeTwin + ConformalRatchet (M2)",
        "date": "2026-11-24",
        "prereg_log": json.loads((P.HERE / "prereg_log.json").read_text(encoding="utf-8")),
        "platforms": scores,
    }
    (P.HERE / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")

    lines = [
        "# TreeTwin + ConformalRatchet — Results",
        "",
        "Pre-registered cross-platform method experiment (`PREREG.md`, frozen SHA-256 "
        f"`{summary['prereg_log']['sha256'][:12]}…`). Zero real training labels entered any GBDT; "
        "every real exam split was opened once.",
        "",
        "## 1. Point Accuracy: Physics-Trained GBDT vs Released Neural Incumbent",
        "",
        "| Platform | Exam units (rows) | Metric (lower better) | Incumbent MLP / DeepONet | GBDT-Plain | GBDT-Mono | Mono − Inc (95% CI) | Parity vs LGB |",
        "|---|---:|---|---:|---:|---:|---|---:|",
    ]
    for p in PLATFORMS:
        s = scores[p]
        b = s["paired_bootstrap"]["gbdt_mono_vs_incumbent"]
        bp = s["paired_bootstrap"]["gbdt_plain_vs_incumbent"]
        verdict = "win" if b["significant_a_better"] else ("loss" if b["significant_b_better"] else "tie")
        ci = f"[{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] ({verdict})"
        lines.append(
            f"| **{s['title']}** | {s['n_exam_units']} ({s['n_exam']:,}) | {s['target']} | "
            f"{b['b']:.3f} | {bp['a']:.3f} | **{b['a']:.3f}** | {b['diff_a_minus_b']:+.3f} {ci} | "
            f"`{s['sim_val']['mono_parity']:.1e}` |"
        )

    lines += [
        "",
        "## 2. Uncertainty Calibration: Pooled Widen-Only vs Mondrian Conformal (Nominal 68.3%)",
        "",
        "Fitted strictly on real Dev residuals (widen-only `>= 1.0`), graded once across held-out exam strata.",
        "",
        "| Platform & Learner | Raw cov (worst gap) | Pooled widen cov (worst gap) | Mondrian raw cov (worst gap) | Mondrian shrunk cov (worst gap) | Mean width: Pooled -> Shrunk |",
        "|---|---|---|---|---|---|",
    ]
    for p in PLATFORMS:
        s = scores[p]
        for arm_label, arm_key in (("Incumbent", "incumbent"), ("GBDT-Mono", "gbdt_mono")):
            cc = s["conformal_coverage"][arm_key]
            def _fmt(k):
                c = cc[k]
                return f"{100*c['pooled_coverage']:.1f}% (`-{100*c['worst_under_gap']:.1f} pp`)"
            w_p = cc["pooled_widen"]["mean_halfwidth"]
            w_s = cc["mondrian_shrunk"]["mean_halfwidth"]
            lines.append(
                f"| **{s['title']}** ({arm_label}) | {_fmt('raw')} | {_fmt('pooled_widen')} | "
                f"{_fmt('mondrian_raw')} | **{_fmt('mondrian_shrunk')}** | {w_p:.4f} -> {w_s:.4f} |"
            )

    lines += [
        "",
        "## 3. Exact TreeSHAP Twin-Test Gap Rankings (Real Dev vs Simulator)",
        "",
    ]
    rebuild = P.HERE / "dev_rebuild_2026-11-27.json"
    rb = json.loads(rebuild.read_text(encoding="utf-8"))["platforms"] if rebuild.exists() else None
    if rb:
        lines += [
            "Rebuilt 27 Nov 2026 (`dev_rebuild_2026-11-27.json`, dev only) after the PR #19 review: the first version",
            "grouped EnergyNet features by the wrong indices, filed AirfoilNet's tmax under trailing-edge thickness, and",
            "ran BuildingSim with constant 3 m/s wind. The tables below come from forests rebuilt on Windows with the same",
            "scripts and seeds (close to, not identical with, the forests the exam scored). The score files keep the",
            "original tables as they were run.",
            "",
        ]
    for p in PLATFORMS:
        s = scores[p]
        ts = rb[p]["treeshap_twin_dev"] if rb else s["treeshap_twin_dev"]
        lines.append(f"### {s['title']} (`{ts['mode']}`, units: {ts['unit']})")
        lines.append("")
        lines.append("| Rank | Physical feature group | Mean twin | Mean real | Mean real − twin | Mean |real − twin| | Gap share |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for r_i, row in enumerate(ts["ranked"][:6], 1):
            lines.append(
                f"| {r_i} | {row['feature']} | {row['mean_twin']:+.2f} | {row['mean_real']:+.2f} | "
                f"{row['mean_diff']:+.2f} | {row['mean_abs_diff']:.2f} | {100*row['share_of_gap']:.1f}% |"
            )
        lines.append("")

    lines += [
        "## 4. Cross-Family Disagreement as a Label-Free Gap Detector (`|GBDT-Mono − Incumbent|`)",
        "",
        "| Platform | Spearman r vs Incumbent |err| | p-value | Top-10% disagreement error enrichment | Spearman r vs GBDT-Mono |err| |",
        "|---|---:|---:|---:|---:|",
    ]
    for p in PLATFORMS:
        s = scores[p]
        di = s["disagreement_exam"]["vs_incumbent_error"]["overall"]
        dm = s["disagreement_exam"]["vs_gbdt_mono_error"]["overall"]
        lines.append(
            f"| **{s['title']}** | {di['spearman_r']:+.3f} | {di['spearman_p']:.2e} | "
            f"{di['top10pct_enrichment']:.2f}× | {dm['spearman_r']:+.3f} |"
        )

    lines.append("")
    (P.HERE / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Wrote {P.HERE / 'results.json'} and {P.HERE / 'RESULTS.md'}")
    return summary


if __name__ == "__main__":
    build_results()
