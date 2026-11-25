# Independent Reviewer Passes (`docs/audits/reviews/`)

This directory provides separate lanes so independent reviewers (human or AI) can run unbiased first-pass audits against the same base commit before comparing notes and updating the canonical model report in `docs/audits/<model>-v<ver>.md`.

## Recommended Convention
- Place independent first-pass notes under `docs/audits/reviews/<reviewer>/<model>-v<ver>.md` (e.g., `reviews/reviewer-a/quakenet-v2.md`, `reviews/reviewer-b/quakenet-v2.md`).
- Follow the 10-point checklist and mandatory 3-question finding format (*What failed? / What changed when tested? / What consequence remains unmeasured?*) in [`../SKILL.md`](../SKILL.md).
- Once two passes have been cross-checked, reconcile verified numbers and any disagreements into `docs/audits/<model>-v<ver>.md` and update the master table in [`../README.md`](../README.md).
