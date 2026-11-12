# Page sources

`page.html` (layout and styles, with placeholders), `app.js` (all interactivity), `chirpnet.js` (the CNN and two-stream
forward passes in plain JavaScript, matching numpy to about 1e-6). `training/build_page.py` inlines these, the page data and
the weights into one self-contained HTML file. The built page is `docs/index.html`.
