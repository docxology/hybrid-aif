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

Bundle schema notes for the camera and the settings:

- `meta.image`: `side` (pixels per side), `layout` (row 0 is the largest y offset from the
  rover, columns run along x), `half_width` (the patch spans plus or minus this many field units
  around the rover), `camera_report` (the held-out numbers of the decoder and the recogniser).
- Per tick, `img` and `recon` are `side` x `side` uint8 rows of that patch (observed, and the
  decoder's mean at the posterior mean and the most likely terrain). `encoder` is the recogniser's
  probability vector over the three terrain levels, applied in the encoder setting and run for
  comparison in the others. `camera` holds the log evidence of the image (inverted settings) or
  the recogniser's `normaliser` and the `covered_term` (encoder setting).
- Per setting, `infogain` lists the modalities whose information gain the rover planner scores
  (`fix`, `img`; empty when the epistemic terms are off). The page shows it under the setting
  selector.
- The rover tier has two modalities, `fix` and `img`. The recognition entry in `graph.recognition`
  reads `rover.img` and targets the `rover.terrain` state factor.

Testing: open `index.html` from disk or serve the folder; the URL hash
`#t=30&tier=mission&view=terms&setting=<id>&theme=dark` sets the initial view, and
`window.__app.set({...})` drives it from a script.

Images: after `showcase_trim.py`, each tick's `img` and `recon` is one base64 string of the pixel bytes in the layout `meta.image.layout` describes (`meta.image.encoding` says so); before trimming they are rows of 0..255 integers. The viewer accepts both forms.
