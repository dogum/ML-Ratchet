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
| `site/public/instruments/<id>/` (from `models/<id>/web/export_instrument.py`) | the live instruments — see `src/instruments/README.md` |

```bash
npm install
npm run dev        # http://localhost:4321/ML-Ratchet/
npm run check      # type-check
npm test           # every live instrument matches its Python model
npm run build      # -> dist/
```

## Design

A precision instrument's logbook: warm paper by day, oscilloscope black by night (`src/styles/global.css`).
Instrument Serif for ideas, Schibsted Grotesk for prose, Martian Mono for every number. One signal colour; each
model keeps its own `accent` from its `model.yaml`. The signature graphic is the ratchet ladder
(`src/lib/ladder.ts`): the incumbent's score as a staircase that only steps up, the margin each rung had to clear,
rejected rungs as ghosts, and the release as a diamond. The home page turns it into a machine
(`src/components/RatchetMachine.astro`): a ratchet wheel and pawl that play through every rung decision in the
collection, painting one tooth per kept rung in the model's colour.

Every model also has a **live instrument** (`src/instruments/`): its real network ported to TypeScript, running in
the visitor's browser on real exam data, in one shared "scope" language (`src/instruments/kit.ts`,
`src/styles/instrument.css`). `/story/` tells the method through QuakeNet as a pinned, scroll-driven story.

To deploy somewhere else, set `SITE_URL` and `SITE_BASE` (see `astro.config.mjs`).
