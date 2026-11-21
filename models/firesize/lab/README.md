# FireSize original page

`index.html` is the page built in the cloud sandbox on 19 Nov 2026 (`firesize_page.html` in the release zip, 1.27 MB,
unchanged; `training/src/build_release.py` wrote its data, `training/src/build_page.py` its texts, into a template
that was not shipped). Open it in a browser. The network runs in the page on the eight exam ignitions of
`examples/`, with a weather and record lab that recomputes the Rothermel inputs. It also has a containment-race toy
on four simulated dev twins, a map of 1,602 exam ignitions (a seeded draw, escapes over-sampled ×10) over a LANDFIRE
fuel-class background, the exam tables, a reliability diagram, the regional AUCs, the ratchet and the limits. Its
weights are rounded to 5 significant figures and its ROS table to 4; the status doc reports P(escape) equal to
numpy to 4 decimals. It loads its fonts from Google Fonts; everything else is inline.

The published copy of this page carries a host wrapper (a 531-byte head and a closing `</body></html>`) around these
very bytes; the zip's file is the cleaner of the two and is the one kept here.

The site's live instrument (`site/src/instruments/firesize/`) is a new port on import: the same network with
float16 weights, the full ROS table rows and the exam's own input values, written by
`python models/firesize/web/export_instrument.py` and checked against Python by `cd site && npm test`.

Data credits: `examples/README.md` (FPA-FOD, LANDFIRE and gridMET: U.S. public data).
