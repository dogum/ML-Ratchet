# The ML Ratchet website

Astro + TypeScript, static, deployed to GitHub Pages at `https://dogum.github.io/ML-Ratchet/` by
`.github/workflows/pages.yml`.

Nothing on the site is written twice. At build time it reads:

| Source | Becomes |
|---|---|
| `models/*/model.yaml`, `ratchet.yaml` | the home page, `/models/`, `/models/<id>/`, `/scoreboard/` (`src/lib/data.ts`) |
| `models/*/lab/index.html` | `/models/<id>/lab/` — copied by `scripts/copy-labs.mjs` |
| `docs/*.md` | `/method/`, `/contract/`, `/contribute/`, `/agent-brief/`, `/lessons/`, `/data/`, `/roadmap/` (content collection; links rewritten by `src/lib/rehype-doc-links.mjs`) |
| `docs/backlog.yaml` | `/backlog/` |

```bash
npm install
npm run dev        # http://localhost:4321/ML-Ratchet/
npm run check      # type-check
npm run build      # -> dist/
```

## Design

A precision instrument's logbook: warm paper by day, oscilloscope black by night (`src/styles/global.css`).
Instrument Serif for ideas, Schibsted Grotesk for prose, Martian Mono for every number. One signal colour; each
model keeps its own `accent` from its `model.yaml`. The signature graphic is the ratchet ladder
(`src/lib/ladder.ts`): the incumbent's score as a staircase that only steps up, the margin each rung had to clear,
rejected rungs as ghosts, and the release as a diamond.

To deploy somewhere else, set `SITE_URL` and `SITE_BASE` (see `astro.config.mjs`).
