# Design: slots and variables, the splice, the joint planner, and `run` over a closed System

*Status: BUILT 2026-11-09 to 2026-11-10 (chunk B of `engine-consolidation.md`, step 2), in six
commits: B1 2b7d4e7, B2 2a73a6b, B3 448c8f4, B4 1a574e3, B5 c4941d1, and B6 (this tree, not yet
committed). Designed by an Opus pass (revision 4, 2026-11-09) to Alex's rulings recorded in
`engine-consolidation.md` ("The bindings: identify, link, emit", "`merge` and `couple`",
"`run`"); each commit built, reviewed adversarially and fixed before the next began. The design
document, the build records and the reviews live in the build directory outside the repo; this
note records the chunk as built, with each ruling dated where it was made. Semantics:
`nesting-and-coupling.md` N1 to N10 and A11; `control-states.md` C13; `emissions-and-mirror.md`
(chunk A); `description-length.md` D2.*

Terms used throughout. A **slot** is a modality, or a control node, as a place a value arrives
at; a **variable** is what fills it: a state factor, a control node, an action node, an emission,
or a slot no identification names. An **out-port** is an action node or an emission (a value a
System generates); an **in-port** is a modality or a control node (a slot another System can
feed). A **leaf** is a component that is not itself composed of others. A **view** is
`subsystem(composite, params, leaf)`, one leaf's own `(System, SystemParams)` taken from the
composite's params, which are the **home** of every parameter the leaves share. A **driven**
control node is one whose level is the realised value of another component's action node or
emission; a **chosen** one is set by a planner; a **free** one is drawn from its habit
(`engine-consolidation.md`, the three regimes).

## 1. Slots and variables (B1)

**`System.observes`.** A static field, `tuple[tuple[str, str], ...]`, given as a mapping from a
slot to the variable it observes, empty by default; a slot the map does not name observes the
variable of its own name. Every slot keeps its own namespaced name (`agent.rover.gauge` stays the
key of `params.likelihood`, of the selectors, of the document and of the tests); the map is the
only place a slot and its variable differ. The variable set is derived, never stored. Structural
checks are in `System._check_observes` (keys are slots, targets are variables the System
generates, an in-port is named once) and port-type equality in `validation._validate_observes`.
Four readers of the map, and nobody else:

- `System.ports(params)`: a slot the map names is no in-port; every emission is an out-port typed
  by its kernel;
- the keyed faces (`infer_states`, `learn`, `replay`, and `infer_policies`'s `present=` since the
  B1 fix pass): observations and `present` flags are keyed by variable and one datum fans out to
  every slot observing it; a key naming an observing slot is refused, naming the variable; the
  datum of a variable the System generates is optional, its slot absent when it is left out;
- `describe` and `build`: an observing slot's likelihood record carries `observes: <variable>`
  where the two names differ, and the slot is no entry of `variables`;
- the description length: a slot the map names is no variable of its own (section 3 for the
  driven control node).

**`Emission(parents, realise="sample", kernel=None)`** (`hybrid_aif.reads`), a noun, is the
declaration of an emission (chunk A's `Emit` is gone, and a test asserts it). The kernel is
static on the declaration or in the params:

- `kernel=None`: the kernel is `params.emission[name]`, any likelihood family, and must be there
  (a missing one is refused);
- `kernel="identity"`: the emission is its one state factor (a level port for a categorical
  factor, its level or under `stats` its probability vector; a float port for a continuous one);
- `kernel=Grid(...)`: an indicator likelihood over the cells of one continuous factor's grid, a
  lookup from a point (`cells.cell_index`) and `cell_mass` from a belief.

*Ruling, 2026-11-09 (B1 review, finding 1): the identity is explicit.* The design read
`kernel=None` as the identity; the build did so for a single parent, which made a forgotten
kernel a silent noiseless copy of the state, a regression on chunk A's "one kernel per emission".
`None` therefore means the params hold the kernel, `"identity"` is spelt out, and the design
carries the dated note. A static-kernel emission reads state factors only (the identity of a
control node is its action node). `System.emit` realises static and params kernels alike and
always validates; `EMIT_TAG` is the tag a step's key is folded with before its emissions draw.

**Reads are emissions observed by a modality.** An own-belief read is an emission of the System
identified with one of its own modalities, written at construction:

```python
System(..., emissions={"cell": Emission(("x",), realise="mode", kernel=GRID)},
       observes={"cell_obs": "cell"})
```

A cross read, `Link(source, modality, realise, kernel)` under `merge`, is sugar for the same pair
across components: an emission of the source's component named by the modality's local name
(`rover.cell_obs` for `mission.cell_obs`), with the kernel and realisation, and an `observes`
entry of the composite. A second `Link` from the same source with the same kernel and realisation
reuses that emission (one out-port, two observers: B1 review, finding 2); a name taken by anything
else is refused.

**Dissolved in B1:** `System.lagged`, `reads.Read`, `read_datum`, `READ_TAG`,
`modality_key`, the `previous=` and `read_key=` arguments of every face, the carry's held
posterior `MindCarry.posterior`, and the face-internal datum helpers. Nothing computes a datum
inside the faces: the datum reaches inference as any observation does, from the caller for a hand
`infer_states` and `learn`, from the emissions `replay` realises after each tick from the
posterior and records (`Replay.emitted`, fed as the next tick's observation, absent at tick 0;
a `sample` emission without `key=` is refused), and under `run` from the step that owns the
emission (section 4). `mirror` drops every emission of the System and every `observes` entry (an
out-port: the mirror models what the System senses). The dated notes on S8
(`structure-learning.md`) and A11 (`nesting-and-coupling.md`) record the change of mechanism; the
semantics of S8 (lag one, a point datum, a soft categorical) hold.

**"A reader never sees a value that does not yet exist"** (Alex, 2026-11-09, retiring A11's fixed
lag). A read is the variable as it stands at the reader's turn; a reader that steps before the
source in the round's order sees the previous round's realisation, one that steps after sees this
round's. B1 kept the one-round lag in the composite form of `run`; `run` over a closed System
(B4) is where the rule is built, and section 4.5 gives the order it is read against.

**Description.** An emission with a static kernel is a family-less `actuator` record
(`options.kernel` `"identity"` or `"indicator"` with its grid, `options.realise` where not the
default); the likelihood record's `options.datum` is gone. No tagged release carried `lagged`.

**Equivalence pinned.** `tests/data/own_read_datum.npz`, recorded from a `git archive` of 61052fc
(chunk A) in a separate interpreter: `tests/test_own_read_equivalence.py` matches the datum, the
posterior, the log evidence and a five-tick replay bitwise for every read that draws no key
(`mode`, `stats`, the grid under both). A sampled read draws from `fold_in(step key, EMIT_TAG)`,
a key unlike chunk A's (an upgrading note in the capabilities bullet).

## 2. The splice and the bindings (B2; `merge` completed in B3)

**The splice** is the stage of `merge` that renames and concatenates the components' variables
and tables before any binding is applied: `merge._splice(subsystems, params) -> _Spliced`
(namespacing, offsets, tables, parameter variables and references, emissions, `observes`,
readings, recognition, per-leaf records, leaves of an inner splice re-qualified).
`merge._assemble(subsystems, params, bindings, *, coupled)` is the splice followed by the bindings
and the metadata; `merge` calls it with `coupled=False` and `couple` with `coupled=True`. From B3
on, `merge` is the splice plus the bindings and nothing more: it builds no joint planner.

**`couple` returns a System.** `couple(components, bindings=(), fixed=None) -> (System,
SystemParams)` is `merge` restricted to `Identify` of an out-port of one top-level component with
an in-port of another, plus a `Clamp` for each `fixed` entry (a discrete control node at a level
or a distribution). A structural `Link`, the read sugar, or an `Identify` of parameter variables
or latents is refused naming `merge`; refusals carry a `couple(): ` prefix. A merge may take a
coupling as a component, and a coupling a merge. The keywords of the wired form (`minds=`,
`processes=`, `links=`) return the `Wiring` until the deletions after chunk C; mixing the two
forms is a `TypeError`.

**Per-leaf configuration.** A leaf keeps its own planner configuration on its `Subsystem` record,
`config`: `policy_len`, `planner`, `gamma`, `num_iters`, `decomposition`, `planning_belief`,
`feedback`, `mixture_schedule`, `mixture_k`, `epistemic`, `terms`, `scoring`; and its recognition
variables, `Subsystem.recognition`. Its habit, precision, durations and scoring parameters are
`SystemParams.parts[<leaf>]`, a `PartParams(policy_prior, precision, durations, scoring)`, written
only where the leaf holds one of them (so a merge of plain components has `parts == {}` and its
params pytree is unchanged). The composite states a planner field where every leaf agrees and the
sentinel `UNSET` where they differ, and states no habit, precision, durations or scoring of its
own. `mixture_k` and `mixture_schedule` are refused at the splice when leaves differ (one belief
carry). `Subsystem.boundary` is `"coupled"` for a top-level component of a coupling.

*Rulings, 2026-11-09 (B2 review, findings 1 to 3):* the composite's own planner fields are
authoritative wherever they are set (edit them with `dataclasses.replace`); a leaf's record speaks
only where the composite holds `UNSET` or states nothing; a merged component brings its own
habit, precision and durations into its leaves' `parts` when coupled. And no face plans a
coupling as one System: `System._plannable(face, params)` refuses in `policies`, `infer_policies`,
`act`, `elapsed`, `replay` and the compiled faces when a planner field is `UNSET`, when `parts` is
not empty, or when more than one top-level component is coupled, naming `joint_planner` (B3) and
the leaf's view.

**A driven control node by identification.** `Identify(out_port, in_port)` (either argument
order) on a control node writes `(node, out_port)` into `observes`, exactly as on a modality. The
node stays in its own leaf, carries the out-port's *realised* value (C13), has `action_names`
entry `None`, gives no out-port, and is never chosen by a planner. `driven_nodes(composite,
name=None) -> {control node: driving variable}` lists them. `Identify("mission.a_option",
"rover.u_opt")` under `merge` and `Identify("agent.rover.a_v", "world.v")` under `couple` are one
form, and from B3 `Link(action node, control node)` under `merge` is the same binding
(`bind_port`). `FaceNames.action_keys` gives a driven node the key of its driving variable, so one
action keyed by the driver sets both nodes and a key naming the driven node is refused naming its
driver.

**`subsystem(composite, params, name, inputs=None)`.** `inputs` is keyed by variable (the action
node of a partner's node, or the variable that drives one of the leaf's own driven slots), the
value `int | Categorical | Array | None` (`None` the uniform average): an `int` slices every table
that reads the node and every carried target conditional on it, a distribution averages them, and
a driven slot not named stays an input with no action node. A driven continuous node is never
sliced. `partner_priors` and `partner_controls` are gone. A leaf that shares no latent with
another (every coupled leaf) gets its own System, unsliced, recognition and scoring included; the
amortized refusal (a stub would cut a recognition conditioning set) applies only to views with
stubs. A view under a distribution that would make a Gaussian a mixture (a target conditional on
the node, a linear-Gaussian transition reading it) raises `merge.MixtureRefusal`, a
`NotImplementedError` subclass (B6).

**The document.** A driven node is written under `relations.driven: {slot: driving variable}`,
with the action node it had before it was driven under `relations.driven_actions` (B3), so a
coupling round-trips with its records exact. The body key `components` (B2 fix pass) carries
`leaves` (per leaf its configuration, `precision`, `durations`, `policy_prior` logits and
`recognition`), `coupled` (the top-level coupled components) and `unset` (the planner fields the
composite holds `UNSET`, so a stated field survives `build`: B3 fix pass); a leaf's scoring
estimator terms are written in `objective.terms` tagged with their `leaf` (B3 fix pass), so a
merge or a coupling with a scoring leaf has a document. `build` restores `Subsystem.config`,
`recognition`, `boundary="coupled"` and `parts`. When a document holds no leaf records, `build`
gives each leaf the composite's own configuration (B6, gap G1: exact, since no leaf records means
every leaf agreed); `describe` writes leaf records whenever leaves differ on `epistemic` or
`terms`, which the composite holds joined rather than agreed (B6 fix pass 2).

## 3. The joint planner (B3)

**A composite is planned as one mind only on request.** `joint_planner(system, params, node=None,
*, inputs=None, **config) -> (System, SystemParams)` (`composition/joint.py`) builds the planning
problem of `node`, a component that is not a leaf, or the whole System for `None`. A leaf, an
unknown node and a System with no components are refused. It is public for hand loops, and `run`
calls it for a step given to a non-leaf node. `merge(...)` followed by `joint_planner` equals the
pre-B3 merge composite exactly (below), which is how every caller that planned a merge as one mind
was adapted in B3 (`examples/compose_minds.py`'s `merged`, `examples/survey_run.py`, twenty test
files).

- **Collapse.** Every driven control node whose driver is an action node inside the node leaves
  the System: its readers (`B_deps`, `A_deps` action columns, `C_deps`, `readings`, emission
  parents) read the driver's control node, the `D_deps`/`H_deps`/`B_deps` pads are re-cut, and
  the driven node's `observes` entry and box are dropped: one mind plans one decision. Refused: a
  kernel in `params.actuator` on the driver's action node that is not the identity (the intention
  and the value planned on would differ; an explicit identity is accepted, B3 fix pass), a target
  or durations on the driven node, a factor or an emission that would read one node twice. A
  driven node whose driver lies outside the node stays driven and takes `inputs=`, as in
  `subsystem`. An emission identified with a modality stays an emission and an `observes` entry:
  the joint model reads its own belief.
- **Configuration.** Each planner field, `mixture_schedule` and `mixture_k`: the argument, else
  the value the planned composite states, else the value the node's leaves agree on; leaves that
  differ are refused, naming the field and each leaf's value. *Ruled 2026-11-09 (B3 review, item
  10):* for a nested node "the composite" is the node being planned, so a field stated on an
  enclosing composite does not reach in. `epistemic` and `terms` are also arguments; `UNSET` as an
  argument is refused.
- **Composition**, from the leaves' records, the composite's own values authoritative where it
  states them: the habit is the log-space sum of the leaves' habits over the joint policy set
  (refused for a policy module, a continuous node, a leaf node the joint planner does not choose,
  a leaf `policy_len` unlike the joint one); one precision (all leaves or none, one value); one
  clock (durations on one node the planner chooses; two leaves with durations are refused); the
  scoring of the one leaf that declares it (two are refused). The result carries `parts == {}`.
- **Nested node.** The collapse runs on the whole System, then the node is cut out with
  `subsystem`, then the composition applies to the cut.

*Rulings, 2026-11-09 (Alex, on the B3 review's item 7): a control node and its action node are
one decision variable.* The action node is the control node's realised value, so the document and
the description length count the pair once. A driven control node is a slot of its driver's
variable, as an observing modality is: the document's `variables` drops the driven control node
and its inert action node, the variable keeps the driver's action node as its name (the kept
name; the readers of the slot name the slot as their parent, through `relations.driven`), and its
level count is paid once, on the driver's control node; the graph code's control count and every
table's candidate parents leave the driven node out. A coupling of `A` and `B` with `k`
identifications counts `|A| + |B| - k` decision variables: for the zoo coupling with the world's
node driven, 14 + 13 - 4 = 23, written as 24 names, the joint planner's own.

| zoo coupling, world's node driven | before the ruling | as built |
|---|---|---|
| document variables | 25 | 24 |
| `control:` addresses | `agent.u_location`, `world.u_location` (3.6994 each) | `agent.u_location` |
| graph code, control count `u` | 9.9284 (`u` = 2) | 9.0624 (`u` = 1) |
| candidate parents of ten tables | 6 | 5 (3.1621 in all) |
| structure total | 118.9946 (the world choosing its node) | 111.2671, the joint planner's |

The difference, 7.7275 nats, is exactly `log*(5)` (3.6994), the graph code (0.8660) and the
smaller parent alphabet (3.1621). Two invariants are pinned in `tests/test_splice.py`: the
description length of a driven coupling equals its joint planner's, and no identified variable
pays its level count twice. The 2026-11-09 note in `description-length.md` D2 states the rule and
withdraws B2's claim that the identification left the total unchanged. Alex's further point, that a
full description length should price the action node's kernel family (an identity is structure
only, a noisy table has parameters) and never by paying the control node's domain again, was
scheduled as its own commit at the end of the chunk (section 10).

**Merge as the splice plus the bindings.** B3 moved into `joint_planner` what `merge` held
transiently through B2 (the same-step collapse, the composition of habit, precision, clock and
scoring, the strict agreement of planner fields). What keeps `joint_planner(merge(...))` exact
lives in `bind_port`: a reader of the driven node's own action node follows the driver's; a
`Clamp` on a driver terminates the node it drives, in either order; `Subsystem.identified` and
`Subsystem.actions` record the pair and the inert action node.

**Pinned.** An oracle built each composite with `merge` on a `git archive` of the pre-B3 commit,
in its own interpreter, against `joint_planner(merge(...))` in the working tree: ten composites in
the build (the zoo, its same-step form, habits with precision and durations, the arena, the
robots) and twelve in the review (relays, clamps and chains in both orders), each identical in all
32 static fields, every params leaf and one planning step.

## 4. `run` over a closed System (B4; timing completed in B5 and B6)

### 4.1 Signature and steps

```python
run(system, params, ticks, key, steps, clocks=None, tick_order="simultaneous",
    carries=None, record_carries=False) -> Trajectory

Step = Callable[[System, Array, Mapping[str, Array | None] | None, Any, SystemParams],
                tuple[Mapping[str, Array], Any, SystemParams]]
```

A **closed System** is a merge or a coupling every modality of which is fed by an identification.
A **stepped node** is a key of `steps`: a leaf, or a non-leaf node then planned as one mind
through `joint_planner`. The keys are an antichain over the System's components that covers every
leaf. A step is `step(system, key, obs, carry, params) -> (out, carry', params')`: `system` and
`params` are its view taken from the home each round; `obs` maps each modality slot and each
driven control node, by the slot's own name, to the value of the variable it observes (`None`
where that variable has no value yet), and is `None` on the first call (4.4); `out` maps out-ports
to values, the action nodes of the control nodes it chooses and every one of its emissions; the
params returned are the object handed in when the step did not learn.

- **`plan(planner=None, learn=None, trace=False, replace=None, **act_kwargs)`**, the step of a
  planning mind, compiled per view (`eqx.filter_jit`, params an argument, equal views sharing one
  program): infer from the carried prior (a `None` datum is an absent modality), learn when
  `learn` is set and a previous round exists (`System.learn(..., prev_beliefs=carry.previous,
  action=carry.action, learn_state=...)`; `learn` is `True`, a mapping of `System.learn`
  keywords, or a sequence of slot keys), act, push the posterior, and realise every emission of
  the view from the posterior with `fold_in(key, EMIT_TAG)`. Its carry is `MindCarry` (`beliefs`,
  `learn_state`, `previous`, `action`, `trace`); `trace=True` keeps `Trace(q_pi, tree, policies)`.
  `replace` sets view fields, and on a non-leaf node the joint configuration. `plan()` is handed
  its view sliced at the realised levels of its driven discrete nodes; the step carries the mark
  `sliced = True`, which a wrapper must carry too (`functools.wraps` copies it), and a `plan()`
  handed an unsliced view with driven nodes is refused naming the mark (B4 review, item 5). A view
  with a driven continuous node is refused for `plan()`, naming a custom step.
- **`sample()`**, the step of a process: the first call is `System.sample_initial` (or the partial
  point state `carries=` seeds), each round `System.sample_step` on the unsliced view with the
  driven nodes from `obs`; it returns every emission realised by its declared rule and the action
  node of each control node nothing drives, drawn from its habit or uniformly; its carry is the
  point state. A driven node whose driver is an action node of the same System takes the driver's
  drawn level (`System._sample_controls`, B4 review, item 7); one nothing supplies is refused.
- **A custom step** is any function of that shape; it realises its emissions with
  `view.emit(fold_in(key, EMIT_TAG), params, {**beliefs, <control levels>})`.

### 4.2 The round body

Per stepped node, `run` holds its slots (modalities through `observes`, driven nodes through
`driven_nodes`), its chosen action nodes and its emissions. A turn: take the view (`subsystem` or
`joint_planner`, cached per input level and refreshed with `retake`; sliced at the realised
levels only for a sliced step, unsliced otherwise); supply `obs`; call the step; check what it
returned (`_check_out`: each key an out-port of the node, an action node one level in range or a
float vector of the node's width, an emission a level in range or a vector of the port's width,
every emission and action node present from round 0, naming the missing one); check the params
(differing only in parameter variables and the view's own learned tables) and write them back
(4.6); store the variables; realise the node's action nodes.

*Ruling under Alex's delegation, 2026-11-09 (B4 review, item 4, a design gap):* `plan(learn=...)`
writes back every own table it learned (a likelihood, a transition, an initial prior not filled
by a parameter variable) under its namespaced key, beside the parameter variables `write_back`
handles, and the cached views are dropped. A learned table whose shape differs from the home's
(the view was sliced at a driven variable) is refused, naming the table and the variable.

### 4.3 Realisation

An action node is realised from its node's `params.actuator` kernel at the intended level when
the node decides (`driver._actuate`), and stands until realised again; the index space is the
System's action nodes in order, the inert action nodes of driven nodes excluded. An emission is
realised by the step that owns it, from whatever that step carries: a sampler from its point, a
planner from its posterior. The runner holds no posterior to realise one from, so a step that
omits an emission is refused on the round it does. On a held round (section 5) every out-port
keeps its standing value. The value a driven node reads, and the level a sliced view is taken at,
is the realised one (C13), which differs from the intended level only under a non-identity
kernel.

### 4.4 The first call is the reset

Before round 0 `run` calls every stepped node once with `obs=None` and its seeded carry
(`carries.get(node)`, else `None`). What it returns stands entering round 0, its carry is its
state, and nothing is written back. `plan()` returns nothing and leaves the declared initial
priors as the first prior (so its emissions are absent entering round 0); `sample()` returns
`sample_initial`'s emissions and state; a custom step's contract is one line, `if obs is None:
return {}, carry, params`. The reset view is the uniform one; where `subsystem` refuses it with
`MixtureRefusal` (a Gaussian target or a linear-Gaussian transition conditional on a driven node),
`runner._reset_view` takes the view sliced at the first level of each driven input, which `plan()`
reads only for shapes (B6, gap G2, the survey rover's shape, and fix pass 2 for the transition).

### 4.5 Timing, as finally ruled on 2026-11-10

A round is one tick under every order and every stepped node steps once. The rule common to all
three orders: **a stepped node whose control node is identified with another stepped node's
action node steps after that action is realised in the round**, and its slot carries the realised
value. A **source** is a stepped node none of whose control nodes an action drives. A cycle of
action-to-control identifications is refused, and so is a leaf whose own action node drives its
own control node (B4 review, item 3).

- **`"simultaneous"`** (the default): two phases. Phase one is the sources, stepped on the round's
  boundary, their action nodes realised together with the last round key. Phase two is the driven
  nodes, in an order that respects the rule. A phase-two node reads a control node driven by
  another node's **action** as realised this round, and every other slot, a control node driven
  by an **emission** included, at the boundary, so what a simultaneous reader sees does not depend
  on the listing. Write-backs land at the round boundary (4.6).
- **`"sequential"`**: the derived depth-first order. The listing orders the sources only; each
  source steps in listing order, and each driven node steps as soon as the last of its drivers has
  realised, before the sweep returns to nodes made ready earlier: a chain follows its source, a
  join waits for its last driver and steps there, nodes made ready by one realisation keep
  listing order among themselves. A driven node's own place in the listing does not matter. Every
  read is the value as it stands.
- **`"async"`**: sequential with a fresh random permutation of the sources each round, drawn from
  the run key (a host generator seeded by `fold_in(key, N)`, one `permutation(len(sources))` per
  round, the draws of B4), then the same derived rule for the driven nodes: one total order per
  round, every read as it stands, no boundary read.

*Dated notes, the history of this rule.* Design 4.4 (revision 4) had three clauses the build
superseded. (1) "An emission identified with a control node is read as it stands": superseded on
2026-11-09 by a ruling under Alex's delegation (B5 review, findings 2 and 4), since reading it so
broke phase one's independence of the listing (two phase-one leaves read different values by their
order) and the boundary read a transition needs (a phase-two component's relay saw this round's
mode); the emission-driven control node reads at the boundary under simultaneous. (2) "A stepped
node listed before the node whose action drives it is refused, naming the reorder" under sequential:
superseded on 2026-11-10 by Alex's ruling of the derived order ("Option B": sequential stays a mode
and must work properly), after a regroup of `with_mirror`'s listing (each world leaf listed after
the model leaf driving it, built in the B5 fix pass) was reviewed as broken in three places and
lossy through the document, and was deleted. (3) Async "then the phase-2 nodes as under
simultaneous", which B4's fix pass had built with boundary reads in phase two: superseded on
2026-11-10 by Alex's ruling that async is sequential over a fresh random listing of the sources. His
reasons, as recorded: sequential is a systematic scan, async a random scan, and simultaneous a
synchronous update; async chooses a random but globally coherent update order, and all three stay
expressible, because a synchronous update of coupled local kernels admits period-two cycles that are
not fixed points of the kernels and need not reach the equilibrium a random scan reaches (the
Glauber-versus-Gibbs distinction, carried over from draws to steps). The derived order was first
built breadth-first by waves; on 2026-11-10 it was ruled depth-first under Alex's delegation, on his
principle that causation propagates from a source to its blanket before the sweep moves on (B5 order
review, S2: for S driving A and B, A driving C, the order is S, A, C, B, and B reads this round's
C). The write-back clause of 4.6 is Alex's ruling of the same day ("true simultaneity").

Keys follow listing position, not step order (design section 7, verified in the order review), so
the derived order changes no draw. Of the seven `run_parity` cases, five derive the listing's own
order and two (the same-step zoo, the pushers on a direct wire) derive another; neither changes
what any node reads, which a comparison of every recorded array against the pre-B5 tree showed.

### 4.6 Write-backs

*Ruling, Alex, 2026-11-10 ("true simultaneity"; origin the B6 review's item 4).* Under
`"simultaneous"` every node's view in a round is taken from the home as the round began; each
learner's write-back is computed against that home and held, and `_at_the_boundary` applies them
after the round's last realisation: a variable one node wrote takes its value; one several nodes
wrote takes the round-start value plus the sum of their deltas where it is Dirichlet counts
(`Dirichlet`, `ConditionalDirichlet`), the posterior given both nodes' data, and a sum that is
not positive is refused with `write_back`'s sentence naming the nodes (B6 fix pass 2). For any
other form, two `plan()` steps whose `learn` both reach the variable are refused before the first
round (`_learned_variables` reads `learn`: `True` every learnable slot, a sequence the slots it
names, a mapping its `lr` and `lr_by_slot`), and two other steps that both wrote it back in one
round are refused at that round's boundary; no marker attribute declares a custom step a
non-learner (ruled under delegation, 2026-11-10). Under `"sequential"` and `"async"` a write-back
is live: the next view is taken from the home as it then stands, which is what the composite form
did under every order.

### 4.7 The `Trajectory` tree

A `Trajectory` (`composition/runner.py`) is a tree with one node per component: `params` at the
root (the home after the last round), and per node `parts` and, keyed relative to the node (a
non-leaf node's mappings the prefixed union of its parts'), `received` and `present` (`ticks + 1`
entries, the last what the slots hold after the last round; an absent value is zeros), `emitted`
(realised action nodes and emissions, `ticks` entries, held values repeated), `intended`, `start`
(the out-ports holding a value entering round 0), `carries` (`ticks + 1` entries under
`record_carries=True`, the first after the reset), `carry` and `decided`. `trajectory[
"agent.rover"]` walks it by dotted path; a jointly planned node is a leaf of the tree. The
composite form's record is `CompositeTrajectory` and its step `MindStep`, until the deletions.

### 4.8 Keys

`key, k_reset = split(key)`; node `j` (in component order, a joint node at its node's place)
resets with `k_reset` for `j = 0` and `fold_in(k_reset, j)` after. Each round `key, k_tick =
split(key)` and `_round_keys(k_tick, N)` is `split(k_tick, 2N + 1)`: node `j` steps with key `j`,
its action nodes realise with key `N + j`, except phase one under simultaneous, which realises
with key `2N`. Realisation draws action node `i` of the `A` from `split(fold_in(k,
ACTUATOR_TAG), A)[i]`; an emission draws from its owner's step key folded by `EMIT_TAG`.
`_reset_keys`, `_round_keys` and `_sweep_generator` are the only places the layout lives, so a
parity test patches them to a pre-B layout (`tests/test_run_parity.py`, `_old_layout(M)`: minds
stepped with key `j` of `split(k_tick, 2M + 2)`, realised with key `M + j` or the last, the one
process with the last key) and compares bitwise against fixtures recorded at the commit before
the change.

### 4.9 Closedness and the up-front refusals

Each a sentence naming the fix: a modality no identification feeds (an open in-port); `steps`
that are not an antichain covering every leaf (a missing leaf, a leaf under a stepped node, an
unknown name), and a non-leaf node with a `sample()` leaf beneath it ("step that node's other
leaves instead": the joint view would take a process's tables as its model; a hand call of
`joint_planner` plans such a node as a self-model, design FLAG 3, accepted 2026-11-08); a
control node with no action node and no `observes` entry (B4 review, item 1); `clocks` or
`carries` naming no stepped node; a cycle of action drives; `tick_order` outside the three. A
control node a leaf chooses is listed among `ports(params).ins` (it can be driven), so closedness
is a property of the modalities alone (ruled 2026-11-09, B2 review, finding 8). A non-leaf node
stepped by `sample()` (a composite world) is allowed.

## 5. Clocks (B5)

**Durations imply holding** (design FLAG 5). A stepped node whose view declares
`params.durations` (one control node of the node carrying a duration per level) decides once per
option. On a decision round it steps, and `run` reads the duration of the level it chose
(`runner._chosen_duration`: the entry of `params.durations` at the intended level, read directly
so that an unsliced custom-step view with driven nodes needs no slicing, B5 order review S1),
times the node's `clocks` entry (rounds per unit of duration, 1.0 for a node it omits). A product
under one round or not a whole number is refused with `hold`'s arithmetic and sentences
(`driver._rounds_per_unit` and `_option_rounds`, shared with `hold`). `run` stores `_Held(remaining,
action)` in its own state, not the step's carry, so `carries[t].trace` of a held `plan(trace=True)`
node reads the decision's plan and `carries=` seeds a step, never a count. On a held round there
is no view, no step call, no write-back and no realisation; the node's out-ports keep their values
and the nodes it drives step on its standing action. The first round always decides.
`Trajectory.decided` records the decision rounds.

**`clocks=` is only the unit conversion.** Its keys name stepped nodes that declare durations (a
key naming a node without durations is refused, a typo guard the design did not state). Which
nodes declare durations is read from the home, without building a view (`_declared_durations`:
the System's own `params.durations` beneath the node where it holds any, the leaves' `parts`
otherwise, following `subsystem`'s precedence, B6 fix pass 2). Kept from `hold`: the refusal of a
non-identity actuator kernel on a clocked node's action nodes (one draw would stand for the whole
option; lifting it is a follow-on). Refused: durations on a control node
another node drives (the node chooses no option). A mirror carries no durations (a world steps
every round, B5 review, finding 1). A joint planner over leaves whose durations differ is refused;
leaves in different stepped nodes keep one clock each. `tests/test_step_params.py`, the one test
that declared durations as a counter, counts in `params.precision`.

## 6. The mirrors

**The whole mirror under `with_mirror`.** `with_mirror(system, params, *, per_component=False,
lower=True, param_key=None, names=("model", "world"))` couples `system` with `mirror(system,
params, lower=lower, param_key=param_key)`: each modality `observes` does not name is identified
with the world's emission of that name, each action node with the world's control node it
realises, and, for each control node `u` the model drives internally by its own action node `v`
(`driven_nodes(system)`), `Identify(model.v, world.u)` (B4 review, item 1, blocking: without it the
world's copy was drawn freely). For that binding `mirror` gives such a node an action node of its
own (`_action_name`: `u_x` gives `a_x`). The mirrored params' leaves are copied before `couple`, so
`lower=False` works. It is a convenience with no properties of its own: any System is closed by
it, and a world that is a model run forward is the mirror stepped by `sample()`. The whole mirror
of a merge holds every component in one leaf, so under sequential and async it cannot step one
component's factors at that component's turn; the mirror per component can.

**The mirror per component.** `mirror(system, params, per_component=True)` (delegating to
`component_mirror.mirror_components`) returns the *merge* of one mirror per component, a leaf
each, named as the component is, so a parameter variable several components read stays one. For
component `c`: its own factors, transitions, priors, control nodes and modalities are mirrored as
the whole mirror does, with no durations. A **foreign state parent** of a transition (the arena's
`lock_b` reading `b.hand`) becomes a driven control node `u_<owner>_<factor>` of `c` (in a tabular
table the parent's axis moves behind the others), fed by an identity emission of the owner,
`<factor>_read = Emission((factor,), "mode", "identity")`: the mode of a point is the point, exact,
no draw. A modality whose parents are **all states of one other component** moves to that owner
as an emission `<reader>_<modality>` (`world_emissions(system)` gives the names, and `with_mirror`
binds the model's modality to them), since an emission is the state showing itself and the state
is the owner's. A control node one component's action node drives inside the System is driven by
the same action node in the world (`drive_internal=True`), so a leaf sampled alone follows its
driver; `with_mirror(per_component=True)` passes `drive_internal=False` and drives the world's
copy from the model's action instead. `D_deps` holds one entry per node, control nodes included
(B5 fix pass 3).

**The mixed-modality limitation.** A modality with parents in two or more components
(compose_minds' `status` reads both robots' cells) stays an emission of its own component, its
foreign parents fed through the driven nodes. It reads the boundary value under simultaneous, and
under sequential and async whenever the owner steps after the reader in the round's order: one
round older than the System observes it. The root cause (B5 review, finding 2): a foreign factor
gets one driven node, and a transition needs the factor at the boundary while a modality needs it
after the owner has stepped, so no read-time rule serves both from one slot. *Ruled under Alex's
delegation, 2026-11-09:* relocate the single-owner modalities (exact) and record the mixed case as
a known limitation, per-use driven nodes being the roadmap item; changing the read rule was
refuted by probes in both of its forms.

**Refused**, each with its reason and roadmap item: a foreign state parent of a neural, nonlinear
or deterministic transition; a foreign discrete parent of a linear-Gaussian transition (a
continuous factor reads its discrete parents within the step, which the one driven node cannot
serve; B5 review, finding 3); a foreign continuous state parent; a foreign `D_deps` parent; a
transition reading a parameter variable whose axes would move; a habit on a component (it is over
the leaf's own policy set, which the mirror changes); a node shared by components (the joint
planner's collapse: mirror the merge); a modality reading another component's action node; nested
leaves; a System with no components.

**The arena parity.** `with_mirror(arena, per_component=True)`, listed as `couple` lists it (model
components, then world leaves), derives the sequential order `model.a, world.a, model.b, world.b,
model.arena, world.arena`, and is bitwise to the composite form's recorded run
(`tests/data/clock_parity.npz`) under simultaneous and sequential: both minds' choices and all six
world factors. Async is not comparable: the composite form permuted the minds only and stepped the
arena at a fixed turn, while `run` permutes every source, the arena's model leaf included; the
design's rule is the adjudicated one, and `test_async_is_reproducible_and_distinct` pins `run`'s
own.

## 7. The migration (B6, this tree)

**Onto `run`.** `examples/compose_minds.py` (`world(start)` the mirror with `lower=True`, eight
`Identify`s in `BINDINGS`, `coupled_steps(selection)` giving `plan()`, `plan()`, `sample()`,
positions from `rec["world"].carries`); `examples/hide_and_seek.py` and its notebook (`play` on
`run`, `tally` from the world's `start` and `emitted`); `demo.ipynb` section 9's prose; and the
test files that ran the composite or wired form, or called `agent_step`, `ancestral_step`,
`hold` or `particle` for a behaviour `run` has (the census is in the build record). The docs
follow, and the capabilities bullet carries a migration guide, one line per form.

**Pinned against the composite and wired drivers.** Each migrated test keeps its claim and is
pinned to a fixture recorded from a `git archive` of HEAD in a subprocess (the generator asserting
the archive's package is the one imported), compared bitwise under `_old_layout`'s patched keys:
`run_parity.npz` (B4, seven cases, 171 arrays), `clock_parity.npz` (B5), and three new fixtures,
`migration_couple.npz` (18 arrays), `migration_driver.npz` (70, every observations, realised and
states array compared) and `migration_particle.npz` (18). Where a test pinned the literal values
of the composite form's key stream, it pins `run`'s own layout beside the parity comparison.
Excluded, with reasons in the tests: async of the arena and of the held particle, and the float
composite's states under sequential.

**What differs from the old forms, pinned under the rulings.** A sequential sibling read that
steps after its source sees this round's emission (the composite form lagged a round). A driven
view gets this round's realised level under every order, a kernel on the read node included
(design FLAG 4; `test_particle` 252 and the kernel refusal inverted). A learned own table is
written back (4.2). In round 0 another node's action node is absent, not level 0. Under
simultaneous, write-backs land at the boundary (4.6). Moved numbers: `hide_and_seek.py`'s `main`
prints `caught 1, mean distance bin 1.71` (was 2, 1.29) and the notebook sweep moved from `3.0 ±
2.3 / 6.7 ± 2.9 / 0.9 ± 1.0` to `3.7 ± 1.1 / 6.8 ± 2.4 / 0.6 ± 0.7` ticks caught of 40, both from
the world's key use alone (the same games are bitwise under patched keys, and the old numbers come
back with the world on old keys); the prose claims hold. `demo.ipynb` section 9 moved nothing.

**What stays until the deletions after chunk C.** The survey (`examples/survey_run.py` and its
modules) runs on `particle`, `hold` and the wired form until chunk C migrates it, so these stay:
`run`'s dispatch to `_run_composite_or_wired`, `particle`, `ParticleCarry`, `hold`, `HoldCarry`,
`agent_step`, `ancestral_step`, `MindRecord`, `ProcessRecord`, `CompositeTrajectory`, `MindStep`,
`Wiring`, `Mind`, `Process`, `couple`'s wired keywords and `Subsystem.identified`. Tests whose
subject is the contract of one of these symbols are left in place and mapped to the `run` test
that covers the same rule (one added where none existed: an unknown component or port refused
listing the candidates, B6 review, item 4).

**The alias refusal.** *Ruled under Alex's delegation, 2026-11-10, consistent with the one
decision variable:* a driven control node and its driver's action node are one variable named by
the driver's action node, so an `Identify`, or a `Link` into a control node under `merge`, that
names the inert action node of a driven node is refused in either order of the bindings, naming
the driver's action node (`merge.py`, `no_inert_end` and `identified_outs`). A chain written
through a driven node, which `couple` rerouted silently to the first driver (found in the B5 order
review) and `merge`'s `Link` chain collapsed the same way, is refused; a join on one action node is
the spelling. The B3 test of a same-step chain inverts.

**G1 and G2**, the two gaps the first B6 pass found in `src` and its fix passes closed. G1:
`describe` then `build` of a merge of agreeing leaves dropped `Subsystem.config`, so `run` saw the
rebuilt merge as one leaf; `build` gives the leaves the composite's configuration (section 2), the
workarounds in two tests are gone, and the round-tripped merge coupled to its mirror runs bitwise
as the original. G2: a sliced step on a leaf whose Gaussian target (or linear-Gaussian
transition) is conditional on a driven node could not run, the durations scan and the reset call
both taking a uniform view; the scan reads the home (section 5) and the reset falls back on
`MixtureRefusal` (4.4). G2 was the survey rover's shape, so it blocked chunk C.

## 8. Departures from the design, and the rulings that caused them

Each departure is described where its subject is; this is the index, design section first.

- 1.3, `kernel=None` the identity: the identity is explicit (2026-11-09, section 1).
- 1.1, merge's refusals and habit composition going (FLAG 1): kept in `merge` through B2 so the
  composite form and the survey stayed bitwise, moved into `joint_planner` in B3 as designed;
  `mixture_k` and `mixture_schedule` stay refused at the splice (one belief carry).
- 1.2, a driven node written as a slot: B2 built it as a variable of its own, priced twice; one
  decision variable, Alex, 2026-11-09 (section 3).
- 3.2, a non-identity kernel refused: an explicit identity is accepted, any other refused on
  either end (2026-11-09, section 3).
- 4.2, parameter variables only: learned own tables are written back (2026-11-09, under
  delegation, section 4.2).
- 4.4, three clauses superseded and the order made depth-first (2026-11-09 and 2026-11-10,
  section 4.5).
- 4.6, deletions: none in chunk B (the survey gate runs on the old forms until chunk C); the names
  `Trajectory` and `Step` went to `run`'s, the composite form's becoming `CompositeTrajectory` and
  `MindStep`.
- 5, `_Held(remaining, action, realised)`: built without `realised`; `clocks` naming a node
  without durations is refused.
- 6, "`F` columns into `G`": a continuous factor's state parents are discrete only, and a foreign
  discrete parent is refused; single-owner modalities move to their owner (2026-11-09, under
  delegation, section 6).
- 6 and 9, the arena under three orders: bitwise under simultaneous and sequential, async not
  comparable (2026-11-10).
- 4.2, a write-back inside each turn: at the round boundary under simultaneous (Alex,
  2026-11-10, section 4.6).
- Names the design left open: `u_<owner>_<factor>`, `<factor>_read`, `<reader>_<modality>`,
  `_action_name`'s `a_x`, `merge.MixtureRefusal`, `relations.driven_actions`, `components.unset`.

## 9. Acceptance as run

At every commit: ruff check and format clean on `src`, `tests` and `examples/*.py`; the full
suite green (2 skipped, 8 deselected throughout); `examples/demo.ipynb` executed by nbconvert with
no error output and only the timings cell differing from the stored notebook; the survey
(`examples/survey_run.py --level 2`, both settings, 400 ticks) bitwise to chunk A's recorded
runs (`max abs diff 0.0`, no differing tick); the `as_env` fixtures and, from B1, the own-read
equivalence bitwise. Per commit, the suite count of the final gates and what was added:

- **B1 2b7d4e7**: 3,388 passed. `test_observes`, `test_own_read_equivalence` and its fixture; the
  recurrent-mixture acceptance of `test_structure_learning` at its recorded outcome.
- **B2 2a73a6b**: 3,416. `test_splice`; the 1,069 `merge` and `subsystem` results of the suite
  fingerprinted against HEAD, no System or params difference and 13 changed refusal sentences;
  two stored notebook numbers re-pinned (541.4 and 508.5, chunk A's `log*(E + 1)`, not B2's).
- **B3 448c8f4**: 3,454. `test_joint_planner`, `test_joint_description`; the oracle of section 3;
  HEAD's own tests run with `merge` wrapped in `joint_planner`, no System or params difference.
- **B4 1a574e3**: 3,543. `test_run`, `test_run_parity` and `run_parity.npz`, which the review
  regenerated from a fresh archive bit for bit.
- **B5 c4941d1**: 3,613. `test_clocks`, `test_clock_parity`, `test_component_mirror` and
  `clock_parity.npz`; the arena bitwise under simultaneous and sequential; `run_parity` untouched.
- **B6**: 3,661 (the main session's own rerun before the commit). The three `migration_*.npz`
  fixtures, every other fixture under `tests/data` unchanged; `examples/hide_and_seek.ipynb`
  re-executed with no error and stored with its outputs.

## 10. Open items, carried forward

- **Actuator pricing**, next, as its own commit: price an action node's kernel family (the
  identity is structure only, a noisy table has parameters) without paying the control node's
  domain again; it moves every model with an actuator, pinned numbers and the showcase included.
  The label of an emission's record and its price are settled with it (`description-length.md`
  D2, note of 2026-11-08).
- **Clock offsets**: a `clocks` entry with an initial offset, so two period-two nodes alternate
  rather than decide on the same rounds (agreed with Alex, 2026-11-10).
- **Per-edge delays in rounds** as the general form of the three orders, each order a particular
  assignment of delays (agreed with Alex, 2026-11-10).
- **The mixed-modality lag** of the mirror per component (section 6): per-use driven nodes.
- **The extra subsystem records of a coupling over a merge**: `describe` then `build` of
  `with_mirror(*merged())` gives two records the original lacks, `world.agent` and `world.world`
  (the whole mirror is one leaf, and the reader rebuilds components from name prefixes); `run` is
  unaffected.
- **The standalone-sampling edge of the whole mirror**: a whole mirror of a System with an
  internally driven node, sampled outside `with_mirror`, draws that node from the habit apart from
  its driver (inside `run` under `with_mirror` the binding drives it); the per-component mirror
  resolves it, and the whole mirror could by collapsing the node into its driver's, against the
  names `as_env` keys.
- **The keyed-core refactor**, queued after the arc (after the deletions that follow chunk C).
- Smaller, from the records: preferences (`obs_targets`, `C_deps`, `epistemic`) on an observing
  slot are scored as fresh observations (pre-existing, unruled); two leaves declaring scoring have
  no joint document form; a component whose driven node precedes a chosen node rebuilds with the
  chosen node first; `Subsystem.actions` names a driven node's inert action node, which no face
  resolves; the actuator refusal on clocked nodes waits to be lifted.
