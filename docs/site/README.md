# The launch page

A static page that replays a precomputed run of the survey rover showcase: a mapping rover with
solar charging, a fast rover tier and a slow survey scheduler that read one learned terrain
table. GitHub Pages serves `docs/`; `docs/index.html` redirects here.

Files:

- `index.html`, `style.css`, `app.js`: the page and its plain-JS viewer (no build step, no
  dependencies); `nav.js` switches the page's panes.
- `bundle.js`: the replay data, one JavaScript file defining `window.HYBRID_AIF_BUNDLE`.
- `static/`: rendered views for readers without JavaScript.

## Layout

A fixed sidebar (a top bar with a menu button below 900 px wide) holds a logo (the
`sidebar-logo` block), the table of contents and the small print (version, licence, links); beside
it the page shows one pane at a time. The Intro pane is everything in `page-prose.md` before the
first `##`, headed by the logo (the `title-image` block); each `## Heading {#id}` section is a
pane with that id, and its `###` subheadings are listed under it in the contents. A section
written `## Heading {#id .sub}` is a child pane, listed under the nearest section before it
without `.sub`; while a child is open its parent's entry is marked open. The demo (the section that
places the `hero-viewer` block) and its children share a bar that sticks to the top of the column:
the demo's heading and the playback controls (the `demo-bar` block). The URL hash selects the pane: `#id` opens a pane, the id of a subheading opens its pane
and scrolls to it, and an empty or unknown hash opens `#intro`; a link in the prose to `#id`
therefore switches panes. The last pane, Documentation (`#documentation`), is added by the build
from the `documentation` block in `page-blocks.html` until the markdown has a section with that id.
`nav.js` shows the pane and fires `hybrid-aif:pane`, on which `app.js` redraws the viewer at the
pane's size.

## Editing the text

The page's prose lives in `page-prose.md`; `index.html` is generated from it. Edit the markdown,
then run, from the repository root (or from `docs/site` as `python build_page.py`):

```bash
.venv/bin/python docs/site/build_page.py          # regenerate index.html
.venv/bin/python docs/site/build_page.py --check  # exit 1 if index.html is out of date
.venv/bin/python docs/site/build_page.py --prose COPY.md --out OUT.html  # build a draft copy
```

The conventions (section ids, lead paragraphs, captions, placeholders) are listed in a comment at
the top of `page-prose.md`; the build's docstring has the full list. Sections are the panes and
their headings the contents entries, so adding, renaming or reordering a `##` section changes the
sidebar too; `{#id .sub}` (id and class in either order) nests a section under the one before
it, and any other class is a build error. A `###` subheading gets an id from its words, or the one
written as `### Title {#id}`.
The header's `links` block and the `version` block are shown in the sidebar's small print, and the
TLDR is plain prose. What is not in the markdown lives in `page-blocks.html` as named
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
  where terrain blocks the sky), `slope`, `terrain` (class index per `meta.names.terrain`) and,
  in bundles with a solar battery, `sun` (the solar yield, where parking charges; the viewer's
  illumination layer shows it when present).
- `meta.names`: `terrain` (firm, rough, ice) and `energy` (empty, low, medium, full).
- `meta.tiers`: `slow_ticks` (24), `options` (survey sector 1 to 4, charge), `zones`,
  `zone_split` (the field's quadrants are split at these x and y), `sector_centres`,
  `charge_site` (the charging place the charge option drives to and parks at; the viewer marks
  it), `survey_readings` (the cells a survey's pooled reading counts), `one_level` (a sentence: the
  one-level run is the rover alone, with no scheduler and no option). Bundles whose one-level run
  pinned an option carry `pinned_option` instead; the viewer reads either.
- `meta.image`: `side` (32), `layout` (row 0 is the top of the picture), `view` (what the camera
  renders: the scene ahead of the rover along its heading; a comparison camera fixed along +y
  carries `camera_heading_degrees`), `camera_mode`, `heading_rule` (the rover's heading,
  `true_heading` per tick, is part of its state and turns only by the turn command),
  `field_of_view_degrees`, `tempering` (`alpha` and a `note`: the image's evidence is tempered for
  the decoder's own error, its noise sd the calibrated sd over sqrt(alpha)) and `camera_report`
  (the decoder's noise calibration and the tempering's measurement). `meta.map_layout` describes
  `map` and `counts`.
- `meta.wall_preference`: `offset` and `note`: the rover's wall preference leaf (`utility` on the
  fix) and each rover plan's `G` are recorded less `offset`, the preference's constant value over
  a plan inside the field. `meta.position_risk`: `note`: the rover's risk leaf on its position
  and each plan's `G` are also recorded less a constant per tick, the smallest value of that leaf
  over the tick's plans (each tick's `offsets.position_risk`; `offsets.wall` is the wall's).
- `meta.dynamics`: `kind` ("vehicle"), `dt`, `gain`, `keep`, `turn_max_degrees`, `command_low`
  and `command_high` (the command's box: drive from 0 to 1, a target speed as a share of the top
  speed, and turn from -1 to 1), `state_sd`, `state`, `top_speed`, `cruise_speed` (the measured
  typical speed the energy tables are built at), `terrain_names` and `rule`, a sentence giving the
  vehicle's integration rule (position along the heading at the speed, a turn of at most
  `turn_max_degrees`, in place too, the speed moving toward the target). Bundles with a solar
  battery also carry `stall` (`level`, `width`: the drive delivers the share 1 / (1 + exp(-(level
  - stall.level) / stall.width)) of the command) and `energy` (the drain and charge constants and
  their `rule`); the energy level is the state's last coordinate. The page rolls each plan of
  (drive, turn) forward with it from the believed heading and speed (and, with a stall, the drive
  share at the believed level, held over the plan), and while playing moves the rover between
  ticks (unless the reader asks for reduced motion).
- `meta.sign_convention`: every leaf is added to G with its sign applied, so an epistemic leaf
  appears negative.
- `graph`: `tiers` (factors, modalities, controls per tier), `edges` (`from`, `to`, `kind`:
  transition, likelihood, target, parameter) and `parameters` (`[{name: "rover.field", readers:
  [...]}]`). The viewer places the nodes by hand (the model is fixed) and draws the parameter
  variable once, with an edge to each reader.
- Per tick of a run: `t`, `v`, `true_x` (position and velocity), `true_heading`, `true_terrain`,
  `recogniser` (the frozen recogniser's class probabilities for the image), `look` (the class it
  reported, when the rover observes it: `--camera look`; such a run has no `recon`),
  `true_energy` (`band`, `level`), `stalled` (with a solar battery: the level below
  `meta.dynamics.stall.level`; else the empty band), `drive_share` (the share of the command the
  drive delivers), `fix`, `gauge` (the gauge's reading: of the level, a number, with a solar
  battery; of the band, an index, before), `img` and `recon` (32 x 32 uint8,
  observed and the decoder's prediction at the belief), `belief` (`components` with `w`, `mu`,
  `cov`; `terrain`; `energy` (band probabilities; with a solar battery from the Gaussian belief
  over the level, whose mean and sd are `energy_level` and `energy_sd`); `velocity`; `mean`; `heading` and `heading_sd`, the believed heading
  and its sd; `speed` and `speed_sd`), `true_speed`, `camera` (`log_evidence`, `scale_mean`, the
  decoder's noise at the belief), `policies` (the rover tier's four warm
  starts: `id`, `start`, `v_plan`, `terms`, `G`, `q_pi`), `chosen` (the index of the acted plan), `approximate`, `camera`,
  `free_energy`, `seconds` and `seconds_parts` (`infer`, the belief update, and `plan`, the
  plan's descent), `offsets` (`wall`, `position_risk`: the constants the plans' records are
  shown less), `map` (the field's posterior mean, three uint8 per cell, cells row-major with x
  first), `counts` (observations absorbed per cell), `coverage` (fraction of cells whose top
  probability exceeds 0.6), `seen` (fraction with at least one count), `option` (the option in
  force; absent in the rover-alone run).
- Per tick of the two-level run, `slow`: `decision` (true every 24th tick), and on a decision
  `option`, `cell` (the scheduler's cell reading: the cell holding most of the rover's position belief), `efe_by_option` (G of the best sequence per first option), `q_by_option`,
  `policies` (the best option sequences, trimmed to keep at least the best one per first option: `options`, `terms` with the `infogain`
  `transition:mission.map` leaf and the `risk` `mission.energy` leaf, `G`, `q_pi`), `chosen`,
  `belief` (`energy`, `visit`) and `seconds`. The one-level run has no `slow`.
- `mdl.components`: one entry per priced component (`name`, `kind`, `nats`, `exact`, `entries`),
  including `parameter:rover.field`, priced once; the two readers' `transition` entries carry
  structure only (`entries` 0). The page groups them by tier from the name's prefix.

Testing: open `index.html` from a local server (or from disk); the URL hash
`#hero&t=150&setting=one-level&tier=rover&view=terms&layer=fix&theme=dark` opens the viewer's pane
and sets the initial view (without a pane id, settings open the viewer's pane), and
`window.__app.set({...})` drives it from a script.

Images: after `survey_trim.py`, each tick's `img`, `recon` and `map` is one base64 string of the
bytes in the layout `meta.image.layout` and `meta.map_layout` describe (`meta.image.encoding` and
`meta.map_encoding` say so); before trimming they are lists. The viewer accepts both forms.
`static/*.png` are screenshots of the page at tick 150 of the two-level run.
