# A categorical field over a continuous state

Status: design, 2026-11-05. Decided in dialogue; the showcase rebuild (a mapping rover with solar
charging) is its first customer.

## What is wanted

A tabular factor whose parent is the cell a continuous factor falls in. The customer is a
terrain field: one Dirichlet row per cell of a grid over the rover's field, so the terrain under
the rover is drawn from the row of the cell it stands in, the rows are learned by the conjugate
count update as the rover sees terrain, and the expected information gain about the rows (the
novelty term) is the epistemic drive that makes mapping a planning objective. The same parent
serves known tables too: an energy band whose drain and charge depend on where the rover is.

In factor-graph terms the new node is a categorical child of a continuous parent with indicator
features: `c = cell(x)`, a deterministic discretisation. It is the conjugate special case of the
discrete-given-continuous family; multinomial logistic regression (MNLR, de-risk report section
1) is the smooth-basis sibling, needs the Polya-Gamma augmentation, and cannot express a patch of
one terrain in a field of another without being handed indicator features. The indicator node is
built first and is not subsumed by MNLR; MNLR reuses its plumbing later.

## The messages

**Down, from the position to the cell (F1).** The belief over the cell is the mass of the
position belief in each cell. For a mixture it is the weight-sum over components of each
component's mass. The mass of a Gaussian component in a rectangular cell is taken as the product
of its two marginal masses (differences of the normal CDF at the cell edges), the correlation
dropped; the first and last cell on each axis absorb the tails. This is a smooth function of the
component means, so the velocity planner's gradient sees the field through it, and an uncertain
rover writes fractional counts across the cells it might be in, a bimodal one to two places. No
bilinear read, no spatial prior: cells are independent Dirichlets, and smoothness, if ever
wanted, enters as a basis change (piecewise-linear nodes), not a coupling prior.

**Up, from the cell to the position (F2, deferred).** The terrain seen under the rover is a
likelihood over position that is piecewise constant across the field: the field's probability of
that terrain at each cell. Multiplying a mixture by it reweights the components by the value at
each mean. Localisation by map is a bonus the messages give for free, never something a run
depends on; it is not built in the first round.

## Where it plugs in

The engine reads a tabular factor's parent beliefs in three places, and each takes a probability
vector (a `Categorical` or a raw vector): the transition prior (`_categorical_step` in
objectives/hybrid.py), the count statistics (`ConditionalDirichlet.stats` called from
system/faces/learn.py), and the transition novelty leaf (`transition_novelty_leaf` in
objectives/discrete.py, through `qs_prev`). Nothing today refuses a continuous factor in a tabular
factor's `B_deps`; it would fail on `.probs` downstream. The build:

- **Declaration.** `System(cells={"x": Grid(low, high, shape, axes)})`: the continuous factor's
  name, the grid's extent over the named axes of its state, and the number of cells per axis.
  A tabular factor may name that factor in `B_deps` only when a grid is declared for it; without
  one the refusal says so. The table's parent axis for that parent has one entry per cell, cells
  flattened row-major (`prod(shape)`), so every arity, learning and pricing rule sees one parent
  axis as it does for any categorical parent.
- **The read.** One function, `cell_mass(belief, grid) -> (cells,)`, over a `Gaussian` or
  `MixtureGaussian`; it is what the three sites receive as that parent's belief, wrapped as a
  `Categorical`. The sites learn nothing else: a cell parent is a categorical parent whose
  probability vector is computed rather than stored.
- **Order within a tick.** Categorical factors step first and read their parents' pre-transition
  beliefs; a mixture factor reads its discrete parents after they step. With the field as the
  terrain's parent and the terrain as the position's switching parent, this gives
  `terrain_t ~ field[cell(x_{t-1})]`: the terrain is read where the rover was one tick ago, and
  the dynamics then use it. The lag is one step of motion and is documented, not corrected.
  Learning uses the same pairing: the terrain posterior at `t` is counted into the cells of the
  position posterior at `t-1`.
- **Validation.** `validate_structure` keeps its rule that a continuous factor's parents are
  categorical; it gains the rule that a categorical factor's continuous parent must have a grid.
  The arity check counts the cell axis as one parent axis.
- **Novelty.** `transition:<field>` selects it as for any learnable transition; `qs_prev` for a
  cell parent is the cell mass of the predictive position belief at that step of the plan, so a
  plan's novelty is the expected information about the rows the rover will stand on.
- **Description length.** A cell parent's axis has no live-level mask (every cell is live); a
  learnable field is priced as rows times the tabular family cost plus the Dirichlet terms, as any
  table with a prior.
- **Faces.** `infer_states`, `update_empirical_prior`, `infer_policies`, `act` and `learn` all go
  through the three sites; `compiled()` needs nothing new. The fused tabular-only objective path
  is not reached by a system with a continuous factor.

## What the planner sees

The mass of a sharp belief deep inside one cell is flat in the mean: the normal CDF has
saturated at both edges, so the novelty term has an exactly zero gradient there, and a gradient
descent that starts from the zero plan stays put even when moving would earn several times the
information (a measured case: 0.29 against 0.05 nats, gradient 0.0 at the zero plan and about
0.7 at a plan of 0.3). Two consequences of the lag add to this: the last action of a plan moves
no read, so it has no novelty gradient, and with `policy_len=1` the term does not depend on the
action at all. The field's epistemic drive therefore needs `policy_len >= 2` and a planner that
does not start from the zero plan; the mixed and gradient planners' warm starts (`starts >= 2`,
a spread over the box) are that, and the showcase uses them. The predictive belief also widens
with process noise along a plan, so later steps see a smoother landscape than the first. No
tempering of the read is built; it is recorded here as the reason the planner settings are not
free.

## The conditional image preference (a second, smaller build)

The rover's preference for bright images is conditional on its energy band, so that mapping is not
coloured by a standing taste for light. Today a `ConditionalGaussian` target may condition on
control nodes only; a state parent is refused because the target is then a mixture at each step.
For the utility term (the expected log-density of the predicted observation under the target),
the parent-belief-weighted average over the levels is closed form and cheap, and that is what the
build adds: a Gaussian target with a categorical state parent, scored as the expected utility
over the parent's belief, on linear-Gaussian and neural modalities. The risk form (a divergence
to a mixture target) stays refused with the sentence it has now.

## What the showcase declares (level 1)

- `x`: position and velocity, a mode-aligned mixture switched by terrain; `v`: the velocity
  command (Box).
- `terrain`: categorical, parent `x` through a grid; its transition is the learnable field
  (a `ConditionalDirichlet` of shape `(3, cells)`), its initial prior uniform.
- `energy`: four bands, parents `(energy, x)` through the same grid; a known table built from the
  world's slope and illumination maps (drain on inclines, charge in light); a state target that
  prefers the full band.
- `fix`: a satellite position fix whose noise map is built from the height map's sky openness.
- `img`: the forward camera, a neural likelihood over `(x, terrain)`: ground texture marks the
  terrain, brightness is the illumination at the position (a smooth field the decoder learns);
  a Gaussian image target for bright images, conditional on `energy`.
- Objective terms: novelty on the field, the uncertainty term on `x`, the energy target, the
  conditional image utility. No exposure node.

Level 2 (a slow tier, mandatory) is designed after level 1 runs; see the memory of the decision.

## Out of scope here

MNLR; the upward message F2; a spatial prior over the field; a learnable energy table; cross-
factor continuous parents.
