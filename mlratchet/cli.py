"""``mlratchet`` — the collection from the command line.

    mlratchet list                   every model, its question and its exam record
    mlratchet info quakenet          the card: what it reads, returns, wins, loses, limits
    mlratchet ladder quakenet        every rung tried, kept or not, and why
    mlratchet demo quakenet          run the demo on real exam examples
    mlratchet scoreboard             wins and losses against baselines, across the collection
    mlratchet validate [id ...]      check folders against the model contract
    mlratchet new gridpulse          scaffold a new model from the template
    mlratchet readme --write         regenerate model READMEs from model.yaml + ratchet.yaml
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import textwrap
from pathlib import Path

from . import __version__
from .registry import get, list_models, repo_root

GLYPH = {"win": "▲", "loss": "▼", "tie": "=", "mixed": "≈", "info": "·"}
DECISION = {"kept": "●", "released": "◆", "rejected": "○"}


def _out(s: str = "") -> None:
    try:
        print(s)
    except UnicodeEncodeError:  # old Windows consoles
        print(s.encode("ascii", "replace").decode())


def _wrap(s: str, indent: int = 2, width: int = 96) -> str:
    return textwrap.fill(" ".join(str(s).split()), width, initial_indent=" " * indent, subsequent_indent=" " * indent)


def _value(v, unit=""):
    s = f"{v:g}" if isinstance(v, float) else str(v)
    if not unit:
        return s
    return s + unit if unit == "%" else f"{s} {unit}"


def cmd_list(args) -> int:
    models = list_models()
    _out(f"{'model':<12} {'ver':<5} {'domain':<14} {'exam W/L':<9} question")
    _out("-" * 100)
    for m in models:
        v = m.verdicts()
        wl = f"{v.get('win', 0)}/{v.get('loss', 0)}"
        q = m.card.get("question", "")
        _out(f"{m.id:<12} {m.version:<5} {m.card.get('domain', ''):<14} {wl:<9} {q[:58] + ('…' if len(q) > 58 else '')}")
    _out(f"\n{len(models)} models. `mlratchet info <id>` for one card.")
    return 0


def cmd_info(args) -> int:
    m = get(args.id)
    c = m.card
    _out(f"{c['name']} v{m.version}  ·  {c['domain']}  ·  {m.status}")
    _out(f"{c['tagline']}\n")
    if c.get("disclaimer"):
        _out(f"  !! {c['disclaimer']}\n")
    _out(_wrap(c["summary"]))
    _out("\nReads\n" + _wrap(c["reads"]))
    _out("\nReturns")
    for r in c["returns"]:
        _out(f"  - {r}")
    _out("\nExam: " + c["exam"]["summary"])
    for row in c["exam"]["results"]:
        base = "; ".join(f"{b['name']} {_value(b['value'], row.get('unit', ''))}" for b in row.get("baselines", []))
        _out(f"  {GLYPH.get(row['verdict'], '·')} {row['metric']}: {_value(row['model'], row.get('unit', ''))}"
             + (f"   vs {base}" if base else ""))
    if c["exam"].get("coverage"):
        _out("  68% error bars cover: " + ", ".join(f"{x['output']} {x['actual']:g}%" for x in c["exam"]["coverage"]))
    for title, key in (("Wins", "wins"), ("Loses", "losses"), ("Limits", "limits"), ("Help wanted", "help_wanted")):
        if c.get(key):
            _out(f"\n{title}")
            for x in c[key]:
                _out(_wrap(f"- {x}", 2).replace("  - ", "  - ", 1))
    _out(f"\nTry it\n  cd {m.path.relative_to(repo_root())}\n  {c['usage']['install']}\n  {c['usage']['demo']}")
    return 0


def cmd_ladder(args) -> int:
    m = get(args.id)
    for lad in m.ladder.get("ladders", []):
        _out(f"{m.name} — {lad['name']}")
        _out(_wrap(f"Objective: {lad['objective']}"))
        _out(_wrap(f"Rule: {lad['rule']}"))
        cols = lad.get("columns")
        _out("")
        for r in lad["rungs"]:
            sim = f"{r['sim']:7.3f}" if "sim" in r else "      –"
            mark = DECISION.get(r["decision"], "?")
            _out(f"  {mark} {r['id']:<4} {sim}  {r['decision']:<9} {r['change']}")
            _out(_wrap(f"why: {r['why']}", 22))
            if cols and r.get("metrics"):
                _out(_wrap("  ".join(f"{c}: {v}" for c, v in zip(cols, r["metrics"])), 22))
        if lad.get("note"):
            _out("\n" + _wrap(f"Note: {lad['note']}"))
        _out("")
    _out("◆ released  ● kept  ○ rejected")
    return 0


def cmd_demo(args) -> int:
    m = get(args.id)
    _out(f"Running {m.name} demo on real exam examples (about {m.card.get('demo_seconds', '?')} s)…\n")
    return m.demo(*args.extra).returncode


def cmd_scoreboard(args) -> int:
    _out("Each row is one exam metric: ▲ beats every baseline listed, ▼ loses to at least one, "
         "= tie, ≈ mixed, · context only.\n")
    tot: dict[str, int] = {}
    for m in list_models():
        v = m.verdicts()
        for k, n in v.items():
            tot[k] = tot.get(k, 0) + n
        marks = "".join(GLYPH.get(r["verdict"], "·") for r in m.card["exam"]["results"])
        _out(f"{m.name:<12} {marks:<10} {v.get('win', 0)} wins, {v.get('loss', 0)} losses")
    _out(f"\nCollection: {tot.get('win', 0)} wins, {tot.get('loss', 0)} losses, {tot.get('tie', 0)} ties, "
         f"{tot.get('mixed', 0)} mixed. The losses are published on purpose.")
    return 0


def cmd_validate(args) -> int:
    from .validate import check

    ids = args.ids or [m.id for m in list_models()]
    bad = 0
    for i in ids:
        problems = check(get(i))
        if problems:
            bad += 1
            _out(f"✗ {i}")
            for p in problems:
                _out(f"    {p}")
        else:
            _out(f"✓ {i}")
    return 1 if bad else 0


def cmd_new(args) -> int:
    mid = args.id.lower()
    if not re.fullmatch(r"[a-z][a-z0-9-]*", mid):
        _out("Model ids are lower-case letters, digits and hyphens, starting with a letter.")
        return 2
    dest = repo_root() / "models" / mid
    if dest.exists():
        _out(f"{dest} already exists.")
        return 2
    name = args.name or mid.capitalize()
    tpl = Path(__file__).parent / "templates" / "model"
    shutil.copytree(tpl, dest)
    for f in dest.rglob("*"):
        if f.is_file():
            if "__ID__" in f.name:
                f = f.rename(f.with_name(f.name.replace("__ID__", mid)))
            try:
                t = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            f.write_text(t.replace("__ID__", mid).replace("__NAME__", name), encoding="utf-8", newline="\n")
    _out(f"Created models/{mid}/ from the template. Next: read models/{mid}/README.md and docs/adding-a-model.md.")
    return 0


def cmd_readme(args) -> int:
    from .readme import render

    ids = args.ids or [m.id for m in list_models()]
    stale = 0
    for i in ids:
        m = get(i)
        text, path = render(m), m.path / "README.md"
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if args.write:
            if current != text:
                path.write_text(text, encoding="utf-8", newline="\n")
                _out(f"wrote {path.relative_to(repo_root())}")
        elif args.check:
            if current != text:
                stale += 1
                _out(f"✗ models/{i}/README.md is out of date — run `mlratchet readme --write`")
        else:
            _out(text)
    if not args.ids:  # the backlog page too
        from .readme import render_backlog

        path = repo_root() / "docs" / "backlog.md"
        text = render_backlog(repo_root())
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if args.write and current != text:
            path.write_text(text, encoding="utf-8", newline="\n")
            _out("wrote docs/backlog.md")
        elif args.check and current != text:
            stale += 1
            _out("✗ docs/backlog.md is out of date — run `mlratchet readme --write`")
    return 1 if stale else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mlratchet", description="Physics-first models that only ratchet up.")
    p.add_argument("--version", action="version", version=f"mlratchet {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="every model").set_defaults(fn=cmd_list)
    s = sub.add_parser("info", help="one model's card"); s.add_argument("id"); s.set_defaults(fn=cmd_info)
    s = sub.add_parser("ladder", help="one model's ratchet ladder"); s.add_argument("id"); s.set_defaults(fn=cmd_ladder)
    s = sub.add_parser("demo", help="run a model's demo"); s.add_argument("id"); s.add_argument("extra", nargs="*"); s.set_defaults(fn=cmd_demo)
    sub.add_parser("scoreboard", help="wins and losses against baselines").set_defaults(fn=cmd_scoreboard)
    s = sub.add_parser("validate", help="check model folders against the contract"); s.add_argument("ids", nargs="*"); s.set_defaults(fn=cmd_validate)
    s = sub.add_parser("new", help="scaffold a new model"); s.add_argument("id"); s.add_argument("--name"); s.set_defaults(fn=cmd_new)
    s = sub.add_parser("readme", help="render model READMEs from model.yaml + ratchet.yaml")
    s.add_argument("ids", nargs="*"); g = s.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true"); g.add_argument("--check", action="store_true")
    s.set_defaults(fn=cmd_readme)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
