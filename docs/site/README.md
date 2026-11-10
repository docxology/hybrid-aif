# The launch page

A static page that replays a precomputed run of the survey rover showcase: a mapping rover with
solar charging, a fast rover tier and a slow survey scheduler that read one learned terrain
table. GitHub Pages serves `docs/`; `docs/index.html` redirects here.

Files:

- `index.html`, `style.css`, `app.js`: the page and its plain-JS viewer (no build step, no
  dependencies).
- `bundle.js`: the replay data, one JavaScript file defining `window.HYBRID_AIF_BUNDLE`.
- `static/`: rendered views for readers without JavaScript.

## Editing the text

The page's prose lives in `page-prose.md`; `index.html` is generated from it. Edit the markdown,
then run, from the repository root (or from `docs/site` as `python build_page.py`):

```bash
.venv/bin/python docs/site/build_page.py          # regenerate index.html
.venv/bin/python docs/site/build_page.py --check  # exit 1 if index.html is out of date
```

The conventions (section ids, lead paragraphs, captions, placeholders) are listed in a comment at
the top of `page-prose.md`. What is not in the markdown lives in `page-blocks.html` as named
blocks, kept verbatim, which the markdown refers to with `<!-- block: NAME -->` placeholders: the
page head (title and description come from the markdown's front matter), the header links, the
viewer panels and their controls and figures, the code listings, the description-length table, the
noscript fallback with its image alt texts, and the script tag. The version number in the byline
is the `version` block; the citation's version is inside the `citation` block. Text the viewer
writes at run time (status line, decision and objective notes and captions, the table footnote)
is in `app.js`. The build needs only the standard library and is deterministic; with the sources
as committed it reproduces `index.html` byte for byte.

To regenerate the data:

```bash
python examples/survey_run.py --level 2 --ticks 400 --out OUT_DIR            # both settings, one part file each
python examples/survey_run.py --level 2 --ticks 400 --out OUT_DIR --assemble # OUT_DIR/bundle.json
python examples/survey_trim.py OUT_DIR/bundle.json docs/site
```

The first call trains and caches the camera's decoder in `OUT_DIR` on its first run. Settings can
be run in parallel with `--settings two-level,one-level`.

Bundle schema notes. Top level: `meta`, `graph`, `settings`, `runs` (`two-level`, `one-level`),
`mdl`, `description`.

- `meta.grid`: the rover's cell grid (`shape` 16 x 16, `low`, `high`); `meta.field` is the field
  extent `[[x0, x1], [y0, y1]]`; `meta.start` the start position.
- `meta.world`: 61 x 61 maps over the field (`xs`, `ys`), each indexed `[ix][iy]`, x first:
  `height`, `light` (illumination), `openness`, `sd_fix` (the satellite fix's noise sd, larger
  where terrain blocks the sky), `slope`, `terrain` (class index per `meta.names.terrain`).
- `meta.names`: `terrain` (firm, rough, ice) and `energy` (empty, low, medium, full).
- `meta.tiers`: `slow_ticks` (12), `options` (survey sector 1 to 4, charge), `zones`,
  `zone_split` (the field's quadrants are split at these x and y), `sector_centres`,
  `pinned_option` (the option of the one-level run).
- `meta.image`: `side` (48), `layout` (row 0 is the top of the picture, the top quarter is sky;
  the camera looks along +y), `camera_report`. `meta.map_layout` describes `map` and `counts`.
- `meta.dynamics`: `dt`, `gain` and `keep` per terrain class, `terrain_sd`, `terrain_names` and
  `rule`, a sentence giving the integration rule. The page rolls each plan forward with it.
- `meta.sign_convention`: every leaf is added to G with its sign applied, so an epistemic leaf
  appears negative.
- `graph`: `tiers` (factors, modalities, controls per tier), `edges` (`from`, `to`, `kind`:
  transition, likelihood, target, parameter) and `parameters` (`[{name: "rover.field", readers:
  [...]}]`). The viewer places the nodes by hand (the model is fixed) and draws the parameter
  variable once, with an edge to each reader.
- Per tick of a run: `t`, `v`, `true_x` (position and velocity), `true_terrain`,
  `true_energy` (`band`, `level`), `stalled`, `fix`, `gauge`, `img` and `recon` (48 x 48 uint8,
  observed and the decoder's prediction at the belief), `belief` (`components` with `w`, `mu`,
  `cov`; `terrain`; `energy`; `velocity`; `mean`), `policies` (the rover tier's four warm
  starts: `id`, `start`, `v_plan`, `terms`, `G`, `q_pi`), `chosen` (the index of the acted plan), `approximate`, `camera`,
  `free_energy`, `map` (the field's posterior mean, three uint8 per cell, cells row-major with x
  first), `counts` (observations absorbed per cell), `coverage` (fraction of cells whose top
  probability exceeds 0.6), `seen` (fraction with at least one count), `option` (the option in
  force).
- Per tick of the two-level run, `slow`: `decision` (true every twelfth tick), and on a decision
  `option`, `cell` (the scheduler's cell reading: the cell holding most of the rover's position belief), `efe_by_option` (G of the best sequence per first option), `q_by_option`,
  `policies` (the best option sequences, trimmed to keep at least the best one per first option: `options`, `terms` with the `infogain`
  `transition:mission.map` leaf and the `risk` `mission.energy` leaf, `G`, `q_pi`), `chosen`,
  `belief` (`energy`, `visit`) and `seconds`. The one-level run has no `slow`.
- `mdl.components`: one entry per priced component (`name`, `kind`, `nats`, `exact`, `entries`),
  including `parameter:rover.field`, priced once; the two readers' `transition` entries carry
  structure only (`entries` 0). The page groups them by tier from the name's prefix.

Testing: open `index.html` from a local server (or from disk); the URL hash
`#t=150&setting=one-level&tier=rover&view=terms&layer=fix&theme=dark` sets the initial view, and
`window.__app.set({...})` drives it from a script.

Images: after `survey_trim.py`, each tick's `img`, `recon` and `map` is one base64 string of the
bytes in the layout `meta.image.layout` and `meta.map_layout` describe (`meta.image.encoding` and
`meta.map_encoding` say so); before trimming they are lists. The viewer accepts both forms.
`static/*.png` are screenshots of the page at tick 150 of the two-level run.
