"""The Ratchet Observatory's dataset (site/src/data/observatory.json), built from the repository.

Every number on the Observatory page (site/src/pages/observatory.astro) comes from here: models/*/model.yaml and
ratchet.yaml, docs/audits (tracker, resolution table, each report's Resolution section), docs/backlog.yaml,
docs/experiments/prior-wars.md and git history. StormNet's live network is not copied in: the page reads the files
its instrument already ships (site/public/instruments/stormnet/).

`mlratchet readme --write` refreshes the committed copy, which local builds use. The site deploy regenerates it from
the full history before building (.github/workflows/pages.yml), so the published page is always current; CI does not
check the committed copy, because every model change on main would make open pull requests' copies stale.

Derived fields, approximate and labelled so on the page: `margin` (log10 of the model against the best numeric
baseline, positive = model better), `beat` (the model beats every numeric baseline of a kind), `rc` on rejected rungs
(a keyword reading of the logged reason) and the audit class (from the tracker's verdict wording).

Run directly to write the file: `python -m mlratchet.observatory`.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import math
import re
import subprocess
from pathlib import Path

import yaml

OUT = Path("site") / "src" / "data" / "observatory.json"
VOLATILE = ("head", "series", "days")      # from git history; not part of the staleness check


def _y(root: Path, rel: str):
    with open(root / rel, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True,
                          encoding="utf-8").stdout


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _beats(model, base, better):
    """True if the model is strictly better than this baseline, False if strictly worse, "tie" if equal."""
    if not (_num(model) and _num(base)) or better not in ("lower", "higher"):
        return None
    if model == base:
        return "tie"
    return model < base if better == "lower" else model > base


def _margin(row) -> float | None:
    m, better = row["model"], row["better"]
    bs = [b["value"] for b in row["baselines"] if _num(b["value"])]
    if not _num(m) or not bs or better not in ("lower", "higher") or m <= 0 or any(b <= 0 for b in bs):
        return None
    best = min(bs) if better == "lower" else max(bs)
    return round(math.log10(best / m) if better == "lower" else math.log10(m / best), 4)


def _cells(line: str) -> list[str]:
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def _audits(root: Path) -> dict:
    tracker = (root / "docs/audits/README.md").read_text(encoding="utf-8")
    audit = {}
    for line in tracker.splitlines():
        if not line.startswith("| **"):
            continue
        cells = _cells(line)
        if len(cells) < 7:
            continue
        links = re.findall(r"\(([^)]+\.md)\)", line)
        audit[cells[0].strip("* ")] = {"label": cells[4].strip("* "), "version": cells[1].replace("`", ""),
                                       "finding": cells[5], "report": links[-1] if links else None}
    grab = False
    for line in tracker.splitlines():
        if line.startswith("| Model | Fixed now"):
            grab = True
            continue
        if grab:
            if not line.startswith("|"):
                break
            if line.startswith("|---"):
                continue
            c = _cells(line)
            if c[0] in audit and len(c) >= 4:
                audit[c[0]].update(fixedNow=c[1], examEffect=c[2], queued=c[3])
    return audit


def _audit_class(label: str) -> str:
    l = label.lower()
    if "critical" in l:
        return "critical"
    if "passed" in l or l == "clean":
        return "clean"
    if ("limit" in l or "capacity" in l) and "defect" not in l:
        return "limit"
    return "defect"


def _resolutions(root: Path, report: str | None) -> list[dict]:
    if not report or not (root / "docs/audits" / report).exists():
        return []
    sec = (root / "docs/audits" / report).read_text(encoding="utf-8").split("## Resolution", 1)
    if len(sec) < 2:
        return []
    out = []
    for line in sec[1].splitlines():
        if line.startswith("## "):
            break
        if not line.startswith("| ") or line.startswith("| Finding") or line.startswith("|---"):
            continue
        cells = _cells(line)
        if len(cells) < 3:
            continue
        act = cells[2]
        b = re.match(r"\*\*([^*]+)\*\*", act)
        word = (b.group(1) if b else act.split(" ")[0]).strip(":. ").lower()
        if "fix" in word:
            kind = "fixed"
        elif word.startswith("warn"):
            kind = "warned"
        elif word.startswith("document"):
            kind = "queued" if "queued" in word else "documented"
        elif word.startswith("queue"):
            kind = "queued"
        elif act.lower().startswith("not changed"):
            kind = "kept"
        else:
            kind = "other"
        out.append({"finding": re.sub(r"[`*]", "", cells[0]), "real": re.sub(r"[`*]", "", cells[1]),
                    "action": kind, "text": re.sub(r"\*\*", "", act)})
    return out


def _reject_reason(w: str) -> str:
    """Keyword reading of a rejected rung's logged reason. Approximate; the page shows the text itself."""
    l = w.lower()
    if re.search(r"stopped|not scored|abandon|failed to learn|did not converge|diverged|not trained|never finished|crash", l):
        return "stopped"
    gate = re.search(r"\bbut\b[^.]*\b(dev|gate|real)\b|dev[^.;]*(worse|rose|grew|fell|dropped|breaks|broke|fails|failed|higher|lower than)"
                     r"|gate[^.;]*(fail|no\b|not)|dev (error|errors|mae|score)[^.;]*[+−-]\d|dev [^.;]*\+\d+(\.\d+)? ?%"
                     r"|fails? the (dev )?gate|dev gate: ", l)
    frozen = re.search(r"does not beat|short of|margin|needs? (more than )?\+?0\.0|needed \+|below the|score fell|fell\b|no gain"
                       r"|worse on (every|the frozen)|^[−-]\d|^[-−]0|vs [^.;]*\([−-]|only, needed|not more than"
                       r"|lower frozen|frozen[^.;]*(fell|lower|worse|below)|^sim [−-]|^sim \+0\.00", l)
    if gate and not re.search(r"does not beat|short of|below the 0\.0|needed \+", l):
        return "gate"
    if frozen:
        return "frozen"
    return "gate" if gate else "other"


def _params(net: str) -> int | None:
    tot = 0.0
    for v, s in re.findall(r"([\d.,]+)\s*([kKM])?\s*(?:weights|parameters|params)", net):
        try:
            tot += float(v.replace(",", "")) * {"k": 1e3, "K": 1e3, "M": 1e6}.get(s, 1)
        except ValueError:
            continue
    return int(tot) if tot else None


def _models(root: Path, audit: dict) -> list[dict]:
    models = []
    for f in sorted((root / "models").glob("*/model.yaml")):
        d, mid = f.parent, f.parent.name
        m = _y(root, f"models/{mid}/model.yaml")
        if m.get("status") in ("in-progress", "idea"):
            continue
        lad = _y(root, f"models/{mid}/ratchet.yaml") if (d / "ratchet.yaml").exists() else {"ladders": []}
        rows = []
        for i, r in enumerate(m["exam"]["results"]):
            bmap: dict = {}
            for b in r["baselines"]:
                ok = _beats(r["model"], b["value"], r["better"])
                if ok is not None:
                    bmap.setdefault(b["kind"], []).append(ok)
            rows.append({"i": i, "metric": r["metric"], "better": r["better"], "unit": r.get("unit"), "model": r["model"],
                         "baselines": [{"name": b["name"], "kind": b["kind"], "value": b["value"]} for b in r["baselines"]],
                         "verdict": r["verdict"], "note": r.get("note"), "margin": _margin(r),
                         # per baseline kind: True beats every one, False some baseline is better, "tie" otherwise
                         "beat": {k: (False if False in v else True if all(x is True for x in v) else "tie")
                                  for k, v in bmap.items()}})
        ladders = [{
            "name": re.sub(r"\bPhase [A-Z],\s*|\s*\(Phase [A-Z]\)|,\s*Phase [A-Z]\b", "", L["name"]),
            "objective": L.get("objective", ""), "rule": L.get("rule", ""), "margin": L.get("margin"),
            "scored_on": L.get("scored_on"), "higher": L.get("higher_is_better", True),
            "rungs": [{"id": str(r["id"]), "change": r["change"], "sim": r.get("sim"), "decision": r["decision"],
                       "why": r.get("why", ""),
                       **({"rc": _reject_reason(r.get("why", ""))} if r["decision"] == "rejected" else {})}
                      for r in L["rungs"]],
        } for L in lad["ladders"]]
        net = m.get("network") or ""
        a = audit.get(m["name"])
        models.append({
            "id": mid, "name": m["name"], "version": str(m.get("version", "")), "status": m["status"],
            "released": str(m.get("released") or ""), "order": m.get("order", 99), "domain": m["domain"],
            "platform": m.get("platform") or "standalone", "accent": m["accent"], "tagline": m["tagline"],
            "question": m.get("question_short") or m["question"], "summary": m["summary"], "disclaimer": m.get("disclaimer"),
            "reads": m["reads"], "returns": m["returns"], "network": net, "noise": m.get("noise"),
            "approach": m.get("approach") or {}, "data": m["data"], "split": m["split"],
            "exam": {"short": m["exam"].get("short", ""), "summary": m["exam"]["summary"],
                     "coverage": m["exam"].get("coverage") or []},
            "rows": rows, "wins": m.get("wins") or [], "losses": m.get("losses") or [], "limits": m.get("limits") or [],
            "help": m.get("help_wanted") or [], "next": m.get("next") or [],
            "physics": {"summary": m["physics"]["summary"], "checks": m["physics"]["checks"]},
            "ladders": ladders,
            "weightBytes": sum((d / w).stat().st_size for w in m["files"]["weights"] if (d / w).exists()),
            "params": _params(net),
            "instrument": (root / "site/public/instruments" / mid / "reference.json").exists(),
            "audit": ({**a, "cls": _audit_class(a["label"]), "resolutions": _resolutions(root, a["report"])} if a else None),
        })
    models.sort(key=lambda m: m["order"])
    return models


def _history(root: Path) -> dict:
    """The collection after each commit that changed a model.yaml or ratchet.yaml, commits per day, and HEAD."""
    log = _git(root, "log", "--reverse", "--format=@@%H|%aI|%s", "--name-only", "--", "models")
    commits, cur = [], None
    for line in log.splitlines():
        if line.startswith("@@"):
            h, t, s = line[2:].split("|", 2)
            cur = {"h": h, "t": t, "s": s, "files": []}
            commits.append(cur)
        elif line.strip() and cur is not None:
            cur["files"].append(line.strip())
    state: dict = {}
    series = []
    for c in commits:
        touched = {(mm.group(1), mm.group(2)) for p in c["files"]
                   if (mm := re.match(r"models/([^/]+)/(model|ratchet)\.yaml$", p))}
        if not touched:
            continue
        for mid, kind in touched:
            s = state.setdefault(mid, {"status": None, "w": 0, "l": 0, "t": 0, "rows": 0, "rungs": 0, "rej": 0})
            try:
                doc = yaml.safe_load(_git(root, "show", f"{c['h']}:models/{mid}/{kind}.yaml"))
            except Exception:
                doc = None
            if kind == "model":
                if not doc:
                    s["status"] = None
                    continue
                s["status"] = doc.get("status")
                res = (doc.get("exam") or {}).get("results") or []
                v = collections.Counter(r.get("verdict") for r in res)
                s.update(w=v["win"], l=v["loss"], t=v["tie"] + v["mixed"], rows=len(res))
            else:
                rs = [r for L in (doc or {}).get("ladders", []) for r in L.get("rungs", [])]
                s.update(rungs=len(rs), rej=sum(r.get("decision") == "rejected" for r in rs))
        live = [s for s in state.values() if s["status"] in ("released", "partial")]
        series.append({"t": c["t"], "h": c["h"][:7], "s": c["s"][:110], "models": len(live),
                       "rows": sum(s["rows"] for s in live), "w": sum(s["w"] for s in live),
                       "l": sum(s["l"] for s in live), "ties": sum(s["t"] for s in live),
                       "rungs": sum(s["rungs"] for s in live), "rej": sum(s["rej"] for s in live)})
    days = collections.Counter(line[:10] for line in _git(root, "log", "--format=%aI").splitlines())
    h, t, s = _git(root, "log", "-1", "--format=%h|%aI|%s").strip().split("|", 2)
    return {"head": {"h": h, "t": t, "s": s}, "series": series, "days": sorted(days.items())}


def build(root: Path, history: bool = True) -> dict:
    audit = _audits(root)
    backlog = [{"id": i["id"], "name": i["name"], "kind": i.get("kind", "model"), "domain": i.get("domain"),
                "status": i.get("status"), "question": i.get("question")} for i in _y(root, "docs/backlog.yaml")["ideas"]]
    prior, grab = [], False
    for line in (root / "docs/experiments/prior-wars.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| Model | Best arm"):
            grab = True
            continue
        if grab:
            if not line.startswith("|"):
                break
            if line.startswith("|---"):
                continue
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            prior.append({"model": c[0], "headline": c[1], "zero": c[2], "k": c[3], "items": c[4]})
    out = {"models": _models(root, audit), "backlog": backlog, "prior": prior, "auditTotal": len(audit)}
    if history:
        out.update(_history(root))
    return out


def _dumps(d: dict) -> str:
    def default(o):
        if isinstance(o, (dt.date, dt.datetime)):
            return o.isoformat()
        raise TypeError(type(o))
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"), default=default) + "\n"


def _stable(d: dict) -> str:
    return _dumps({k: v for k, v in d.items() if k not in VOLATILE})


def is_current(root: Path) -> bool:
    """True when the committed file matches the repository, git-history fields aside."""
    p = root / OUT
    if not p.exists():
        return False
    return _stable(json.loads(p.read_text(encoding="utf-8"))) == _stable(build(root, history=False))


def write(root: Path) -> bool:
    """Rewrite the file when anything but the git-history fields changed. Returns True if it was written."""
    if is_current(root):
        return False
    p = root / OUT
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_dumps(build(root)), encoding="utf-8", newline="\n")
    return True


if __name__ == "__main__":
    from .registry import repo_root

    root = repo_root()
    p = root / OUT
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_dumps(build(root)), encoding="utf-8", newline="\n")
    print(f"wrote {OUT.as_posix()}")
