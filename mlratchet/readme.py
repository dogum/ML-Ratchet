"""Render each model's README.md from its model.yaml and ratchet.yaml.

The YAML files are the single source of truth; READMEs, the CLI and the website all read them.
Regenerate after editing a card:   mlratchet readme --write
CI checks they are in sync:        mlratchet readme --check
"""
from __future__ import annotations

from .registry import Model

SITE = "https://dogum.github.io/ML-Ratchet"
REPO = "https://github.com/dogum/ML-Ratchet"

VERDICT = {"win": "✅ wins", "loss": "❌ loses", "tie": "➖ tie", "mixed": "↔️ mixed", "info": "ℹ️"}
DECISION = {"released": "◆ released", "kept": "● kept", "rejected": "○ rejected"}
KIND = {"classical": "classical", "dumb": "dumb baseline", "published": "published"}


def _v(v, unit: str | None = "") -> str:
    s = f"{v:g}" if isinstance(v, float) else str(v)
    if not unit:
        return s
    return s + unit if unit == "%" else f"{s} {unit}"


def _cell(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render(m: Model) -> str:
    c, L = m.card, []
    w = L.append
    w(f"<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->")
    w(f"# {c['name']} <sub>v{c['version']}</sub>\n")
    w(f"> **{c['tagline']}**\n")
    if c.get("disclaimer"):
        w(f"> [!WARNING]\n> {c['disclaimer']}\n")
    if c["status"] == "partial":
        w("> [!NOTE]\n> Released, with part of the exam still waiting on data — see [Help wanted](#help-wanted).\n")
    w(f"*{c['question']}*\n")
    w(" ".join(c["summary"].split()) + "\n")
    lab = f"[Interactive lab]({SITE}/models/{m.id}/lab/) · " if (m.path / c["files"].get("lab", "lab/index.html")).exists() else ""
    w(f"{lab}[Project page]({SITE}/models/{m.id}/) · "
      f"[Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method]({REPO}/blob/main/docs/method.md)\n")

    w("| | |\n|---|---|")
    w(f"| **Reads** | {_cell(c['reads'])} |")
    w(f"| **Returns** | {'<br>'.join(_cell(r) for r in c['returns'])} |")
    w(f"| **Learned from** | {_cell(' '.join(c['physics']['summary'].split()))} |")
    if c.get("noise"):
        w(f"| **Noise** | {_cell(c['noise'])} |")
    if c.get("network"):
        w(f"| **Network** | {_cell(c['network'])} |")
    w(f"| **Dev set** | {_cell(c['split']['dev'])} |")
    w(f"| **Exam** | {_cell(c['split']['exam'])} |\n")

    w("## Run it\n")
    w("```bash")
    w(f"cd models/{m.id}")
    w(c["usage"]["install"])
    w(c["usage"]["demo"] + (f"        # about {c['demo_seconds']:g} s" if c.get("demo_seconds") else ""))
    w("```\n")
    w(f"or from anywhere in the repo: `mlratchet demo {m.id}`.\n")
    if c["usage"].get("python"):
        w("```python")
        w(c["usage"]["python"].rstrip())
        w("```\n")

    w("## The exam\n")
    w(f"{c['exam']['summary']} Every number here is on real data that never chose anything.\n")
    w(f"| Metric | {c['name']} | Baselines | |\n|---|---|---|---|")
    for r in c["exam"]["results"]:
        u = r.get("unit", "")
        base = "<br>".join(f"{_cell(b['name'])} ({KIND[b['kind']]}): {_v(b['value'], u)}" for b in r.get("baselines", [])) or "—"
        note = f"<br><sub>{_cell(r['note'])}</sub>" if r.get("note") else ""
        w(f"| {_cell(r['metric'])}{note} | **{_v(r['model'], u)}** | {base} | {VERDICT[r['verdict']]} |")
    if c["exam"].get("coverage"):
        w("\nCoverage of the 68% error bars on the exam: "
          + ", ".join(f"{x['output']} **{x['actual']:g}%**" for x in c["exam"]["coverage"]) + ".")
    w("")

    if c.get("wins") or c.get("losses"):
        w("### Where it wins\n")
        w("\n".join(f"- {x}" for x in c.get("wins", [])) + "\n")
        w("### Where it loses\n")
        w("\n".join(f"- {x}" for x in c.get("losses", [])) + "\n")

    w("## Limits\n")
    w("\n".join(f"- {x}" for x in c["limits"]) + "\n")

    w("## The ratchet\n")
    for lad in m.ladder.get("ladders", []):
        if len(m.ladder["ladders"]) > 1:
            w(f"### {lad['name']}\n")
        w(f"**Objective:** {lad['objective']}  ")
        w(f"**Rule:** {' '.join(lad['rule'].split())}\n")
        cols = lad.get("columns") or []
        head = "| Rung | Change | Frozen-set score | Decision | Why |" + "".join(f" {x} |" for x in cols)
        w(head)
        w("|---" * (5 + len(cols)) + "|")
        for r in lad["rungs"]:
            sim = f"{r['sim']:.3f}" if "sim" in r else "—"
            met = "".join(f" {_cell(x)} |" for x in (r.get("metrics") or [""] * len(cols)))
            dec = DECISION[r["decision"]]
            dec = f"**{dec}**" if r["decision"] == "released" else dec
            w(f"| {r['id']} | {_cell(r['change'])} | {sim} | {dec} | {_cell(r['why'])} |{met}")
        if lad.get("columns_source"):
            w(f"\n<sub>Columns after *Why*: {lad['columns_source']}.</sub>")
        if lad.get("note"):
            w(f"\n{lad['note']}")
        w("")
    w("Raw per-rung logs are in [`training/`](training/).\n")

    w("## Physics\n")
    w(" ".join(c["physics"]["summary"].split()) + "\n")
    w("**Simulator**\n")
    w("\n".join(f"- {x}" for x in c["physics"]["simulator"]) + "\n")
    w("**Checked against independent references**\n")
    w("\n".join(f"- {x}" for x in c["physics"]["checks"]) + "\n")

    if c.get("help_wanted"):
        w("## Help wanted\n")
        w("\n".join(f"- {x}" for x in c["help_wanted"]) + "\n")
    if c.get("next"):
        w("## Next rungs\n")
        w("\n".join(f"- {x}" for x in c["next"]) + "\n")

    f = c["files"]
    w("## Files\n")
    w("| Path | What |\n|---|---|")
    w(f"| {', '.join(f'`{x}`' for x in f['inference'])} | Inference — numpy/scipy only |")
    w(f"| {', '.join(f'`{x}`' for x in f['weights'])} | Weights |")
    w(f"| `{f['demo']}`, `{f.get('examples', 'examples/')}` | Demo on real exam examples (fixed random draw, not picked by result) |")
    w(f"| `{f['card']}` | Full model card: every number, every caveat |")
    w("| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |")
    w("| `training/` | Every script that built it, as run; paths via `training/_paths.py` |")
    if f.get("lab"):
        w(f"| `{f['lab']}` | The original interactive page, self-contained (the network runs in the browser) |")
    if f.get("notebook"):
        w(f"| `{f['notebook']}` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |")
    w("")

    w("## Data and credits\n")
    w("| Source | Used for | Licence / terms |\n|---|---|---|")
    for d in c["data"]:
        w(f"| [{_cell(d['name'])}]({d['url']}) | {_cell(d['role'])} | {_cell(d['licence'])} |")
    w("\nPlease cite the data providers when you use this model. Data stays under its providers' terms; "
      "see [DATA_LICENSES.md](../../DATA_LICENSES.md).\n")
    w("---\n")
    w(f"Part of **[ML Ratchet]({REPO})** — physics-first models that are trained only on simulations, "
      "kept only when they ratchet up, and graded on real data they never saw.")
    return "\n".join(L) + "\n"


def render_backlog(root) -> str:
    """Render docs/backlog.md from docs/backlog.yaml."""
    import yaml

    doc = yaml.safe_load((root / "docs" / "backlog.yaml").read_text(encoding="utf-8"))
    L = ["<!-- Generated from docs/backlog.yaml by `mlratchet readme --write`. Edit that, not this file. -->",
         "# The idea backlog\n",
         "Candidates for the next models. Each has physics to simulate, public real data, and a real exam lined up.",
         "Data reachability was checked from a cloud sandbox in November 2026 ([data-sources.md](data-sources.md)).",
         "Want to build one? Read [adding-a-model.md](adding-a-model.md) and open an issue first.\n",
         "| Idea | The question | Physics | Real data | Exam | Status |",
         "|---|---|---|---|---|---|"]
    for i in doc["ideas"]:
        note = f"<br><sub>{_cell(i['note'])}</sub>" if i.get("note") else ""
        L.append(f"| **{i['name']}**<br><sub>{i['domain']}</sub> | {_cell(i['question'])} | {_cell(i['physics'])} | "
                 f"[{_cell(i['data'])}]({i['data_url']}) | {_cell(i['exam'])} | {i['status']}{note} |")
    L.append("\nModel upgrades (the next rungs for existing models) live in each model's `model.yaml` under `next`, "
             "and blocked exams under `help_wanted`. Collection-level plans: [roadmap.md](roadmap.md).")
    return "\n".join(L) + "\n"
