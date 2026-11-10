---
title: hybrid-aif: a survey rover, one table, two timescales
description: hybrid-aif is a JAX library for active inference over models that mix discrete and continuous state. This page replays a precomputed run of a mapping rover with a slow survey scheduler, two tiers that read one learned terrain table.
---

<!--
The text of the launch page. Edit here, then run `python docs/site/build_page.py` from the
repository root; it regenerates index.html. Do not edit index.html by hand.

Conventions:

- Front matter: `title` is the browser tab title, `description` the search-engine summary.
- The `#` title line and everything up to the first `##` heading is the page header; the title
  is the `title-image` block (the logo in static/, with its alt text). The marker
  `tldr` comment opens the boxed TLDR; the header's paragraphs and list after it go in the box.
- `## Heading {#id}` starts a section; the id is used by the page's script and the URL. Keep it.
  `### Heading` is a subheading inside a section.
- Every paragraph inside a section is a lead paragraph (the page's body text style). Paragraphs in
  the header and footer are plain. A line `{.name}` directly above a paragraph gives it the class
  `name` instead (the header uses `{.sub}` and `{.byline}`).
- A `block: NAME` comment on a line of its own stands for a piece of HTML kept in
  page-blocks.html under that name (viewer panels, controls, figures, code listings, the page
  head and the noscript fallback). Inside a paragraph it is pasted inline (the version number,
  the dynamics rule the script fills in).
- Lines starting with `>` right after a block are that block's caption.
- The `footer` comment ends the sections; the paragraph after it is the page footer.
- Inline: **strong**, *emphasis*, `code`, [link text](url). Type the typographic apostrophe and
  other special characters directly; the build writes them as HTML entities. A plain `'` stays
  plain.
- Line breaks inside a paragraph do not matter; a blank line ends it.
-->

# <!-- block: title-image -->

{.sub}
Continuous-discrete-amortized active inference in JAX

{.byline}
Noumenal Labs · <!-- block: version --> · MIT

<!-- block: links -->

<!-- tldr -->

**TLDR.** `hybrid-aif` is a JAX library for active inference over models that mix discrete and
continuous state, and that allow for combining various inference methods and approximations (e.g.
closed-form messages alongside amortized recognition densities; conjugate parameter updates
alongside gradient-based learning; planning over discrete and continuous variables). Other core
features of the package:

- **Closed-loop active inference on mixture models.** For example, switching linear-Gaussian
  dynamics can be planned through, with mixtures collapsing to a fixed number of components each
  step. Scores on analytically intractable factors come back as intervals rather than point
  estimates: the mixture-entropy terms are bounded above and below in closed form, with planning
  ranking policies based on the midpoints of the bounds. The package supports growing mixtures for
  "structure learning", as in [AXIOM](https://arxiv.org/abs/2505.24784).
- **Expressive model and objective construction.** The usual expected free energy objective of
  active inference is a default, but information gain terms can be switched on or off per variable,
  and different objective terms such as entropy bonuses or cross-entropies with arbitrary sign can
  be used. Preferences or end-state constraints can be placed on any variable, including control
  states, and conditioned on any variable. Latent control variables and realized actions are
  distinct nodes in the graph, and any variable may play multiple roles simultaneously, e.g. a
  control state may be observed.
- **Robust agent/environment composition, hierarchical models, and scheduling.** One declaration
  serves as agent and as environment. Declared systems are joined in one of two ways: `merge` makes
  them one particle, such as a hierarchy whose slow tier reads the fast tier’s belief as sensing and
  sets its option; `couple` joins separate particles, which exchange only actions and sensations
  across their Markov blankets. This page replays a precomputed run of a two-tiered rover model so
  you can watch all of that happen without installing anything.
- **Model description.** Expected free energy can not only be "ablated" per term and per variable,
  but its components are (optionally) returned in every log. Each model can be scored
  (data-independently or on a dataset) for its free energy and minimum description length. Models
  can be rendered as JSON documents for easy modification and scoring using software agents. Bound
  pairs are returned in logs where available, and each number carries a derivation status
  (approximations are flagged as such, and categorized by quality and guarantee).

## A survey rover, one table, two readers {#hero}

A rover is mapping a field of three kinds of ground (firm, rough and ice) by driving over it and
learning which kind lies where. Its energy drains on inclines and rough ground and charges wherever
sunlight falls; there is no dock. It finds its position from a satellite fix, whose noise grows
where terrain blocks the sky, and it looks ahead through a forward camera. A fast rover tier steers
every tick. A slow tier, the scheduler, picks an *option* every twelve ticks, a choice held for that
long: survey one quarter of the field (a sector), or stop and charge.

What the rover learns is one table: for each of 256 cells, how likely each kind of ground is. The
model holds it as a *parameter variable*, a node of its own with a prior and an edge to every factor
that reads it. The rover's terrain factor reads it to predict the ground under it; the scheduler's
map factor reads the same table to value a sector. The model graph draws it once, with an edge to
each reader. Drag the tick, switch the setting, and watch the map fill in.

<!-- block: hero-viewer -->
> The field: the true ground (green-grey firm, tan rough, blue ice) with height contours, under the
> map the rover has learned so far (a cell is tinted by its most probable ground, more strongly the
> surer the rover is, and left clear until the rover has seen it), the sector grid with the option
> in force shaded, the rover with its heading, its belief about its position, and the plan it chose
> for the next three ticks (bold), with the planner’s other warm starts drawn fainter. Plans are
> rolled forward by the world’s rule: <!-- block: dynrule -->. The graph: the model, with the table
> `rover.field` read by `rover.terrain` in the fast tier and by `mission.map` in the slow one. The
> table is one node because the two tiers share it outright (one learned belief by design), not
> because one reads a copy of the other. The scheduler’s cell reading, the cell holding most of the
> rover’s position belief, is not drawn: it is a read across the tiers, taken once per decision, of
> a belief that stays the rover’s own. The belief over position is a weighted sum of Gaussians, and
> each such *mixture component* is drawn as one 2σ ellipse; they are long and thin by design,
> because the camera and the fix constrain position unequally.

## Two timescales {#timescales}

Run on its own, the rover tier maps its neighbourhood and then stalls. The reason is that *novelty*,
the information a plan expects to gain about the table, is local: once the cells the rover can reach
in a few ticks have been seen, nothing pulls it further. The scheduler scores a whole sector at
once. Its *pooled reading* of a sector is the twelve readings a survey takes, counted together as
one slow reading, so a trip to unsurveyed ground earns enough novelty to pay for its energy.

Both runs below use the same rover, model and world. In one the option is pinned to sector 1; in the
other the scheduler chooses it. Coverage, the fraction of cells whose most probable kind of ground
has probability above 0.6, ends at 0.45 against 0.24. Sector 4 is never chosen: at every decision
its energy risk outweighs its novelty. The rover reaches it by crossing.

<!-- block: timeline-figure -->
> Top to bottom: the option in force (a tick above the band marks each decision, every twelve
> ticks), the energy level against its four bands, and coverage over the run, solid for the selected
> setting and dashed for the other. The line follows the tick; drag on the chart to move it.

### The scheduler's decision

<!-- block: decision-figure -->

## The camera {#camera}

The rover's forward camera is a 48 by 48 greyscale picture looking along the field's +y axis. The
panels below show it at four times scale (192 pixels, no smoothing). The rover's model holds the
camera as a neural likelihood, a network that predicts the picture from the rover's position and the
ground type; inference runs that likelihood backwards, so the picture informs both the ground type
and the position.

<!-- block: camera-figure -->
> **Observed image:** what the camera sees now. **Decoder prediction:** the picture the model
> expects at the current belief. **Terrain readout:** the belief over the three kinds of ground, one
> bar each, with the true one marked. Overall brightness is the illumination at the rover’s
> position; the ground texture marks the terrain class; the top quarter is sky.

## Objectives as intervals {#objective}

Every plan is scored by every objective term, and every score is an interval. The bar spans the
lower and upper bound and the mark is the midpoint, which is what the planner ranks on. A term that
is exact is a point; a term over a mixture, such as the risk on the rover's position, carries a
bracket from closed-form entropy bounds. Terms that read the camera go through the decoder's
linearisation at the belief and are marked approximate. Switch tiers to compare the rover's gradient
planner with the scheduler's exact enumeration of option sequences.

<!-- block: objective-viewer -->

## Description length {#mdl}

What each part of the model costs to write down, in nats, grouped by tier. The description length is
the model's own complexity measure, computed from the same declaration that runs the agent. A
component declared as a fixed table, density or network is priced by its family alone and flagged
approximate. The parameter variable `rover.field` is priced once, under its own entry; the two
factors that read it, `rover.terrain` and `mission.map`, pay for their structure only.

<!-- block: mdl-table -->

## Install and quickstart {#start}

<!-- block: install-code -->

**The scheduler, as declared.** This is the slow tier from `examples/survey_scheduler.py`: three
factors (the cell the rover is in, its energy band and the terrain at that cell) plus the option as
a control node, two modalities (the cell reading, which is the cell the rover’s position belief puts
it in, and the energy gauge), and a novelty term on the map. The `readings` entry says a survey
pools twelve fast readings of the map into one slow reading, and the map's table is not its own: it
is the parameter variable `field`.

<!-- block: scheduler-code -->

**Joining the tiers.** From `examples/survey_run.py`: `merge` joins the scheduler and the rover into
one particle. `Link` from the scheduler's option action node makes the scheduler's option the
rover's control node, so the rover plans no decision for it. `Identify` identifies the two tiers'
tables as one, which is the single `rover.field` node in the graph above: the tiers share it
outright, because the rover's counts and the scheduler's novelty are one belief by design. The
scheduler also needs the rover's position, which it takes as a sensation: the cell holding most of
the position belief, passed in once per decision. The mission and rover views of the composite then
run on their own clocks, the rover every tick and the scheduler once per option.

<!-- block: merge-code -->

**Running it.** The scheduler alone needs no trained network, so it runs in a few seconds. This
builds it, feeds it one cell reading and one gauge reading, plans, and prints, for each first
option, the pooled novelty (nats of information about the field), the energy risk and the expected
free energy G of the best two-option sequence that starts with it. From the `examples` folder:

<!-- block: run-code -->

The whole showcase, both tiers over both settings on this page, is
`python examples/survey_run.py --level 2 --ticks 400 --out OUT_DIR`, then the same command with
`--assemble`, then `python examples/survey_trim.py OUT_DIR/bundle.json docs/site`;
`docs/site/README.md` says how the page's data is rebuilt from it.

## Citation {#cite}

<!-- block: citation -->

<!-- footer -->

Noumenal Labs, 2026. Most of this code was written by Claude through Claude Code under the direction
of the maintainers; the design records in
[docs/design/](https://github.com/noumenal-labs/hybrid-aif/tree/main/docs/design) show how each
decision was reached. Licensed MIT.
