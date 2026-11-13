> **Lab notebook.** hybridpilot was built in a claude.ai cloud session (12–13 Nov 2026) on the shared RoadSim + RoadData
> platform, now in `shared/vehicle/`. The full build notes (`hybridpilot-status.md`) live in the claude.ai project and
> will be added here; until then `MODEL_CARD.md` holds every number, and `shared/vehicle/HANDOFF.md` the plan.

## 13 Nov 2026 — integrated into ML Ratchet

- Moved from `hybridpilot-v1.0.zip` into `models/hybridpilot/`. Sandbox paths now go through `training/_paths.py`
  (work folder `models/hybridpilot/work/`, the vehicle platform at `shared/vehicle/` or `$ML_RATCHET_VEHICLE`).
- The demo reproduces the release's output exactly from the repository (checked against the zip's own run).
- Caveat carried from the handoff: built before RoadData's 13 Nov fixes (EPA city/highway labels were swapped; VED
  mass double-counted the 136 kg driver). Re-score after the fixes are upstreamed into RoadSim.
