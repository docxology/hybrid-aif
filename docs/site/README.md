# The launch page

A static page that replays a precomputed run of the two-tier rover showcase. GitHub Pages
serves `docs/`; `docs/index.html` redirects here.

Files:

- `index.html`, `style.css`, `app.js`: the page and its plain-JS viewer (no build step, no
  dependencies).
- `bundle.js`: the replay data, one JavaScript file defining `window.HYBRID_AIF_BUNDLE`.
- `static/`: rendered views for readers without JavaScript.

To regenerate the data:

```bash
python examples/showcase_rover.py --out OUT_DIR            # the whole grid, one part file per setting
python examples/showcase_rover.py --out OUT_DIR --assemble # OUT_DIR/bundle.json
python examples/showcase_trim.py OUT_DIR/bundle.json docs/site
```

The first call trains and caches the camera's decoder and encoder in `OUT_DIR` on its first
run. Settings can be run in parallel with `--settings id,id`.

Testing: open `index.html` from disk or serve the folder; the URL hash
`#t=30&tier=mission&view=terms&setting=<id>&theme=dark` sets the initial view, and
`window.__app.set({...})` drives it from a script.
