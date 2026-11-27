# History

The documents the collection was planned and built from, kept as written (November 2026). They predate this
repository: paths, page links and "status" refer to the cloud sessions where the models were built.

| File | What |
|---|---|
| `agent-brief-2026-11.md` | The brief four parallel agents built BearingNet, SwellCast, BeatNet and StarQuake from. Its repo-ready successor is [../agent-brief.md](../agent-brief.md). |
| `repo-notes-2026-11.md` | The running plan for this repository: names, layout ideas, to-dos before going public, open questions. |
| `ideas-2026-11.md` | The idea pool and lessons as they stood (7 Nov). Successors: [../backlog.md](../backlog.md), [../lessons.md](../lessons.md). |
| `ideas-2026-11-13.md` | The same, updated 13 Nov 2026: GridSim chosen as the next project, other big-system candidates checked, vehicle lessons. |
| `repo-notes-2026-11-13.md` | The repository notes as updated 13 Nov 2026, after publication. |
| `ideas-2026-11-14.md` | The idea pool updated 14 Nov 2026: the power-grid project done (GridSim and four models), next big-system candidates (BuildingSim, Watershed, WindFarm), grid lessons. |
| `repo-notes-2026-11-14.md` | The repository notes as updated 14 Nov 2026: the grid project waiting to come in as a shared module next to RoadSim. |
| `ideas-2026-11-22.md` | The idea pool updated 22 Nov 2026: the B batch done (QuakeNet v2, TsunamiNet, LawFinder, AirfoilNet), the wind-tunnel scope, the Rust/WebAssembly idea (spec: [../wasm-pilot.md](../wasm-pilot.md)), lessons to 22 Nov. |
| `repo-notes-2026-11-22.md` | The repository notes as updated 22 Nov 2026: what was still to come into the repository, the B batch included. |
| `quakenet-v2-status-2026-11-22.md`, `tsunaminet-status-2026-11-22.md`, `lawfinder-status-2026-11-22.md`, `airfoilnet-status-2026-11-22.md` | The B batch's four status notes as handed over with the release zips (22 Nov 2026). Each model's current record is its own folder. |
| `ideas-2026-11-24.md` | Post-B-batch assessment (24 Nov 2026): unused learner families, verified cloud-container endpoint reachability (including Daymet v4, NGL IGS20, OpenSLR-28, Zenodo bridge SHM), BridgeSense selected, TreeTwin added to backlog, Rust/WASM deferred. |

## Edits since

- **14 Nov 2026.** Wording neutralised in these files, the model notebooks and the vehicle briefs. Personal names and
  references to chat or agent sessions became project terms ("the owner", "the second v2 calibration pass"). Facts,
  numbers, dates and paths are unchanged. The original page links stay in the notebooks and the files above as records.

- **14 Nov 2026.** The grid project's two notes files added (`*-2026-11-14.md`) with the same neutral wording; their
  claude.ai page links were replaced by each model's `lab/` folder.

Each model's own build notes are in `models/<id>/NOTEBOOK.md`. The vehicle platform's status as of 13 Nov 2026 is
[shared/vehicle/STATUS.md](../../shared/vehicle/STATUS.md); the grid platform's, as of 14 Nov 2026,
[shared/grid/STATUS.md](../../shared/grid/STATUS.md).

- **22 Nov 2026.** The B batch's notes added (`*-2026-11-22.md`) with the same neutral wording and page links replaced
  by each model's `lab/` folder. The repo notes reuse the 14 Nov text below their header, which was otherwise unchanged.

- **24 Nov 2026.** Added `ideas-2026-11-24.md` capturing the post-B-batch assessment, verified data reachability, BridgeSense selection, and TreeTwin mental model.
