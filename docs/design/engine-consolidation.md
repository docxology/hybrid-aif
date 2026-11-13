# Design: engine consolidation. One kind of participant, one run

*Status: PLAN 2026-11-06 (Alex + Fable), steps in order; step 1 APPROVED by Alex the same day
(names `step`, `agent_step`, `ancestral_step`; the composite form takes over `particle`'s loop).
Supersedes nothing: `nesting-and-coupling.md` N1 to N10 stand (what crosses, realisations,
reads, write-back as bookkeeping); this note is about the shape of the engine that carries them.
Customer: every caller of `run`, and the two-tier survey rover of the launch page.*

## What the engine should be

Alex's statement of the target, which this note takes as given:

1. a way to define a System as a factor graph;
2. a way to mirror a System as an environment (`as_env`);
3. a way to merge Systems (`merge`);
4. a way to couple Systems by wires (`couple`);
5. a way to run the whole on a clock, under the orders (simultaneous, sequential, async), with
   relative timescales.

Systems differ in three ways only. Some have a planner (a way of choosing actions that is not
message passing on the graph). Some are composite (built by `merge` or `couple`), which changes
nothing about running them. Some are open (unbound ports), and only that changes how a System
meets the rest: an open one cannot run until its ports are bound.

A coupling is one factorised function with one joint. What a wire fixes is how that joint is
handled at one edge: the receiving side conditions on a realised value (a delta), and no message
goes back. That is the `sample` realisation `nesting-and-coupling.md` N8 names for a wire, with
`mode` and `stats` as the alternatives. So `merge` and `couple` differ in a property of edges,
not of object types, and the tick orders are inference schedules over one graph.

## Where the engine departs from that today

- Two `run` forms. The composite form takes a `merge` result and one respond per component, runs
  the composite's own model forward as the world, and neither learns nor writes back. The wired
  form takes `(System, params, respond)` minds and `(GenerativeEnv, env_params)` processes under
  `couple`. Under the simultaneous order the two are bitwise the same; under sequential and
  async the composite form steps one component's factors at its turn, which the wired form
  cannot.
- Params baked into responds. A respond is `respond(key, obs, carry) -> (action, carry')` with
  its params closed over, so the runner never sees params and nothing under `run` can learn. The
  showcase runner first worked around this by hand, then N4 did so with a second respond shape
  (`ViewRespond`, params as an argument) and a wrapper (`particle`) that takes views from the
  composite's params each round and writes learning back.
- Two kinds of participant, mind and process, where a process is a System responding by
  sampling its model forward (`ancestral_respond` already is that).
- A slow clock as a wrapper (`hold`) the caller composes, rather than a clock the run is given.
- Vocabulary (mind, process, respond, carry, view, sibling read, hold, lagged, particle) used
  without definition where it appears, and more of it than the design needs.

## The steps, in order

Each step is small enough to vet on its own, keeps the test suite and the notebook gate green,
and says how the recorded survey runs are reproduced.

1. **Params flow through the runner.** DONE (f41330a). One step shape for every participant,
   `step(key, obs, carry, params) -> (action, carry', params')`; the runner holds params per
   participant and records them; the composite form learns.
2. **Worlds act, and one `run`.** One build, revised on 2026-11-07 after Alex's questions (below,
   "Step 2 in detail"): a world is a System that acts, so every port's direction is fixed by its
   kind; `as_env` becomes the mirror transform; the bindings become `Link`, `Identify` and `Clamp`;
   `couple` returns a System and `Wiring` goes; `run` takes a closed System, its params, a step per
   leaf component (`plan()`, `sample()` or custom), and clocks, and returns a `Trajectory` tree; the
   composite form, `particle`, `hold`, `agent_step` and `ancestral_step` go. The action node of a
   state factor (nesting-and-coupling.md N5) is built as part of it.
3. **Cleanup.** The vocabulary pass over the docs touched, and the deletions the build leaves.

Queued, not in this arc: moving the positional core to keyed indexing. Even then `emissions`
stays a field apart from `action_names`, for a reason that is not indexing: a control-parented
action node is an intention made real (a policy posterior, a duration, an actuator kernel keyed
by the chosen level, a place in the planner's control space), while an emission is the state
showing itself with no choice behind it, which is why nothing is scored about one (S2). The two
out-ports differ in what explains them (Alex, 2026-11-07).

## Step 1 in detail: params flow through the runner

**The step.** `Step = Callable[[key, obs, carry, params], tuple[action, carry, params]]`. A
participant that learns returns new params; one that does not returns the object it was given.
The runner treats "the same object" as "did not learn" (an identity test, no tree comparison).

**Constructors.** `agent_step(system, planner=None, **act_kwargs) -> Step` (today's
`agent_respond` without `params`); `ancestral_step(system) -> Step` (today's
`ancestral_respond` without `params`); `hold(step, ticks_per_unit=1) -> Step` (today's `hold`
without `system` and `params`). Checks that need concrete params (a `legal` mask that masks every
level, `durations` present and positive, an identity actuator under `hold`) run on the first
call, where the params are in hand and concrete, with the same messages as today.

**The runner.** `couple`'s mind entries stay `(System, params, step)`: params are declared where
the System is, and `run` takes them from the wiring as the participant's starting params. Each
round `run` hands a participant its current params and keeps what comes back. `MindRecord` gains
`params`, the participant's params after the last round (the starting object if it never
learned). Processes are unchanged in this step (step 3 makes them Systems).

The composite form keeps its signature, `run(composite, params, minds, ...)`, and takes each
component's step from the shared params each round exactly as `particle` does now: the view at
the levels the other components last chose, built once per level and refreshed from the home;
learning written back with the view that was handed out as `before`; a step whose returned
params differ outside its declared parameter variables refused. `_run_composite` and `particle`
share one per-round body. `Trajectory` gains `params`, the composite's params after the last
round.

**Deleted.** `ViewRespond`, `view_respond`, `RespondFn` as a second shape. `particle` stays
until step 2 (the showcase runs on it) but takes plain steps. `write_back` stays public for a
hand-written loop over views.

**Keys.** Unchanged. The key streams of both forms are as documented today, so every run that
does not learn is bitwise what it was, with one exception found in review and ruled on
2026-11-07: a composite in which a component's view reads a control node another component
chooses. Before this step that view kept the uniform prior its caller built, whatever the chooser
did; now `run` hands it the view at the level the chooser last chose, as `particle` already did
and as N4 says. That is a change of behaviour for such composites, accepted as the composite
form's semantics rather than made opt-in. Under the simultaneous order every step is handed the
levels as they stood when the round began; under sequential and async, as they stand at its
turn. A reader with no uniform view (its carried targets conditional on the chooser's node) is
supported only where the chooser is sure to have chosen before it responds in every round, that
is under sequential with the chooser listed first, or inside `particle`; under simultaneous and
async it is refused at construction. The level handed on is the chooser's intended level, which
is the realised one only under an identity actuator kernel, so a non-identity kernel on a
control node another component's view reads is refused.

**Migration.** `agent_respond(system, params, planner, **kw)` becomes `agent_step(system,
planner, **kw)` with params given to `couple` or `run`; `ancestral_respond(system, params)`
becomes `ancestral_step(system)`; `hold(system, params, respond, t)` becomes `hold(step, t)`;
a hand-written respond gains a `params` argument and returns it. Callers in the repo: two
examples (`compose_minds.py`, `hide_and_seek.py`), the survey runner, eleven test files, the
notebook if it drives a run. Outside callers of the 0.18 composition API change the same way.

**Acceptance.** Suite and notebook gate green; `examples/survey_run.py --level 2` at both
settings bitwise equal to the runner at the step's start over 400 ticks (the comparison script
of the N4 build); the key-stream tests of `test_driver.py` unchanged; a learner under the wired
form returns its learned params in the record (new test); a learner under the composite form
writes back and its sibling sees the learning (today's `test_particle.py`, retargeted).

## Step 1 as a caller sees it

**A planner coupled to a world, where the planner learns.** Today the learner keeps its own
params in a closure and the runner never sees them:

```python
state = {"params": params0}

def respond(key, obs, carry):                       # params closed over
    beliefs = system.compiled().infer_states(state["params"], obs, priors=carry.beliefs)
    state["params"] = system.compiled().learn(state["params"], beliefs, ...)
    action, carry = ...                                 # plan, act, push the prior
    return action, carry

wiring = couple(minds={"agent": (system, params0, respond)},
                processes={"world": (env, env_params)}, wires=[...])
rec = run(wiring, ticks=400, key=key)
learned = state["params"]                              # fished out of the closure
```

After step 1 the params come in and go out, and the record holds them:

```python
def step(key, obs, carry, params):
    beliefs = system.compiled().infer_states(params, obs, priors=carry.beliefs)
    params = system.compiled().learn(params, beliefs, ...)
    action, carry = ...
    return action, carry, params

wiring = couple(minds={"agent": (system, params0, step)},
                processes={"world": (env, env_params)}, wires=[...])
rec = run(wiring, ticks=400, key=key)
learned = rec["agent"].params                          # what the step last returned
```

A planner that does not learn is `agent_step(system, planner)`; it returns the params it was
given. A world component that only samples its model is `ancestral_step(system)`.

**Two planners on one merged System, one of them learning.** Today this needs the `particle`
wrapper, a `ViewRespond` for the learner, and `couple` to host it. After step 1 it is the
composite form itself:

```python
cs, cp = merge(
    subsystems={"mission": ms, "rover": rs},
    params={"mission": mp, "rover": rp},
    bindings=[
        ActionControl(driver="mission.u_option", control="rover.u_opt"),
        SharedParameter(left="rover.field", right="mission.field"),
        SiblingRead(modality="mission.cell_obs", source="rover.x", kernel=grid, realise="mode"),
    ],
)
mview, _ = subsystem(cs, cp, "mission")
rview, _ = subsystem(cs, cp, "rover", partner_priors={"mission.u_option": 0})

traj = run(cs, cp, {"mission": hold(agent_step(mview), ticks_per_unit=1),
                    "rover": rover_step(rview)},          # rover_step learns the field
           ticks=400, key=key, tick_order="sequential")
traj.params                                            # the composite's params, field learned
```

Each round `run` takes the mission's view and the rover's view from the composite's params as
they stand, with the option the mission last chose in force in the rover's view; the rover's
returned params are written back into the composite's `field`, and the mission's next view sees
the counts. The caller never takes a view's params or calls `write_back`. The world here is the
composite's own model run forward, as the composite form has always done; the survey runner,
whose world is a separate process, stays on `particle` under `couple` until step 2 lets a merged
System be a participant of a coupling directly.

**The survey runner in step 1.** The rover's and scheduler's steps lose nothing but the
`ViewRespond` wrapper; `hold(mview, mparams, scheduler, every)` becomes `hold(scheduler, every)`;
`particle(cs, cp, {"mission": slow, "rover": fast})` and the `couple` around it are as now. Both
settings reproduce today's runs bitwise over 400 ticks.

## Beyond step 1: the generic step (Alex, 2026-11-07)

Two cases the end state must handle, which fix what steps 2 and 3 aim at:

- A planner can be a node in an ancestral pass: an agent that receives observations and then
  acts, and that in the context of the pass functions as a prior over its action. So stepping a
  System forward realises each factor by its own rule: a table factor by a draw from the table, a
  control node that carries a planner by running it on what is realised so far and on its carried
  beliefs. `agent_step` and `ancestral_step` are then one function, the step of a System, and a
  planner is a rule attached to a control node, not a kind of participant. Step 1 keeps the two
  names because the runner still thinks in participants; they collapse in step 3.
- A coupling with open ports can be nested inside a broader System. So `couple` returns a
  System (step 2) whose wires are edges with their declared realisation, `merge` binds its open
  ports as it binds any, and the step of a composite is the scheduled stepping of its parts,
  recursively: the tick orders are the schedule at one level, a clock multiplier is a schedule
  entry saying a part steps every k rounds, and `run` is only "call the closed System's step for
  T ticks and record". The carry of a composite is the tree of its parts' carries. `particle`'s
  loop is this recursion written for one level.

## Step 2 in detail: worlds act, and one `run`

Revised 2026-11-07 after Alex's questions; supersedes the earlier "every participant is a System
with a step" and "one run" drafts.

**Ports are modalities in and action nodes out; a control node is never a port.** A control
node is an internal latent `c`, the model's control variable, in one of three regimes, and the
regime is all that distinguishes a planner from a world:

- *chosen*: a planner's policy posterior sets it (today's control node of an agent);
- *driven*: an input arrives and the node is inferred from it. Formally the input is a sensory
  modality `a` of the System, the received action, with the identity likelihood `P(a | c)`, so
  exact inference gives `q(c) = δ(c = a)`: the posterior mode is the slice of the transition the
  input selects, which is the mirror of the agent's own intention selecting a slice of its model
  of the dynamics. Under a soft input (`stats`, N8) `q(c)` is the probability vector and the
  transition is averaged over it, which is what `BoundaryPrior`'s vector case and a view's
  `partner_priors` already do;
- *free*: no datum and no planner, so `q(c)` is the prior over `c`, the habit, and a sampler
  draws from it (today's `ancestral_step`).

Every port's direction is then fixed by its kind: modalities in, action nodes out. A world is a
System that acts: its inputs are the input modalities of its control nodes, driven by links from
the agent's action nodes, and its outputs are action nodes whose parents are its state factors,
realised each round from an emission kernel (the thing a model calls its likelihood), and linked
to the agent's modalities. So every link runs from an action node to a modality, and in blanket
terms the world's inputs are its sensory states, as they should be. An ancestral sampler is an
actor, not a perceiver: the whole model run forward, every control node free. Nothing has to
declare what a step realises, so no object is needed around a step function: a custom step is a
plain callable.

**The shortcut the code takes.** The identity inference is a slice, so the engine keeps computing a
driven node's posterior by slicing (a level) or averaging (a vector) as it does now, with no
inference machinery on the input modality; the equivalence above is what makes the shortcut exact,
and it is stated where the slice is taken. The representation carries the scrupulous form:
`System.ports` lists each control node's input among the in-ports under the node's name (a port-list
convention, not a modality the inference engine sees: `"world.v"` is an in-port of the world),
descriptions and the link checks show it as one, and a non-identity input likelihood (actuator noise
on the receiving side) is a later option that changes nothing here. Walked by Alex and Fable before
this arc (the identity mapping into the state space) and re-derived on 2026-11-07; the shortcut is
the new piece.

**The action node of a state factor: an emission.** The keystone, N5 of `nesting-and-coupling.md`,
decided and not built. Declared on `System` as `emissions`, a mapping from a name to its parents
(state factors, and control nodes, which the mirror needs) or to `Emit(parents, realise="sample")`,
a field of its own rather than an entry of `action_names`, which is paired by position with the
control nodes in a dozen places. Its kernel, of any likelihood family, is `params.emission[name]`;
its port type is read off the kernel as a modality's is; `System.emit(key, params, values)` realises
it inside the step from a point state (a world) or a belief (an agent). It has no node index, so
nothing in the model reads or scores it (S2): a planner's view carries it as an output only.
`System.ports(params)` lists a System's in-ports (modalities, and control nodes under their own
names: the driven regime's input) and out-ports (action nodes and emissions). Described as an
actuator, the record widened to any out-port kernel with any parents (an emission is the paradigm
case of one, Alex 2026-11-07), and priced as a kernel of its family, so emissions cost description
length. Chunk A as built: `emissions-and-mirror.md` (2026-11-08).

**The mirror.** `mirror(system, params) -> (System, SystemParams)` is the transform behind `as_env`:
every modality becomes an emission parented by the modality's parents, with the likelihood as its
kernel; factors, transitions, priors, control nodes, the habit and durations stay (control nodes now
driven or free); targets, planner fields, reads and recognition are dropped; parameter variables are
KEPT (a world learns if its step learns; a perceiver that learns without planning is `plan()` on a
System with no free control nodes); a System that already has emissions is refused. `as_env` is the
mirror, then the variables lowered to tables (the posterior mean, or a draw with `param_key`) as
today, then the unchanged forward process, so it samples bitwise what it samples now.

**The bindings: identify, link, emit** (ruled by Alex 2026-11-08, superseding the three kinds of
2026-11-07 that chunk A built; chunk B rewrites to this). The graphical model of the whole has
variables and factors, and a factor sends messages both ways, always, with one exception: a variable
that is a *transducer*, one whose value is determined solely by its external source, absorbs the
messages sent into it. A modality identified with another component's out-port is such a variable:
on the receiving side it is clamped to what the source realised, so its likelihood's top-down
message into it is computed but changes nothing, while the same likelihood's upward message is what
inference uses. A likelihood whose child is a latent sends top-down messages as any factor does (a
predictive-coding hierarchy; the amortized recognition map is a piece of the process playing the
role of an approximate posterior). So flow along an identified variable is one way exactly when the
receiving variable is a transducer, which is ordinary message passing with observed variables, and
nothing in the vocabulary names a direction or sorts factors into kinds (Alex's refinement of
2026-11-08, replacing an earlier wording in terms of process and model factors).

- `Identify(a, b)`: two variables are one. Always two-way, by the nature of identity; what flows is
  decided by whether the receiving variable is a transducer. Identify an action node or an emission
  of one component with a modality or a control node of another and the flow is one way, since the
  receiving variable is a transducer: a coupling. Identify two latents, or two parameter variables,
  inside one model and the flow is both ways: a merge. Today's `SharedParameter`, `Wire`, and the
  same-step `ActionControl` (from the driver's *action node*, realised, to the driven control node:
  the driven regime) are all this. Identification collapses the two names to one variable, named by
  `a`, as it already does for parameter variables: every factor that read `b` reads `a`, the
  document writes the variable once, the MDL counts it once, and no alias survives; a leaf's view
  maps the variable back to the leaf's own local name as it strips prefixes today (Alex,
  2026-11-08). Designers should give the two ends the same local name; the identification joins the
  models at that node.

- `realise` lives on the factor that generates the variable, nowhere else: an action node's actuator
  kernel, an emission's `Emission(parents, realise="sample")` (the declaration is a noun,
  `Emission`, replacing chunk A's `Emit`) (`sample`, `mode`, `stats`, N8). A belief read, the
  scheduler reading the rover's position through the cell indicator, is an emission of the rover
  from its belief over position with the indicator as the kernel, realised by `mode`, identified
  with the scheduler's modality (chunk A's `emit` from a belief). `Link(source, target, realise,
  kernel)` as chunk A built it stays as sugar for exactly that pair of declarations, and is
  documented as sugar. An emission is realised from whatever its owner carries, a point for a
  sampled leaf and a belief for a planning leaf, so nothing declares which; the regime decides.

- `Link(parents, potential)`: a structural edge inside one model, a potential gaining parents
  (today's `ObservationLink`, the relay form of `ActionControl` from a state factor to a control
  node's latent, and `TargetEmission` with the potential spelled `"terminal:<factor>"`). Not
  sameness of variables: an edge. Two-way, since it lives inside one model.

- `Clamp(node, level)`: a control latent no planner chooses and nothing is identified with, held at
  a constant: a level, or a probability vector averaged over once at merge time (today's
  `BoundaryPrior`, and `couple`'s `fixed`). An identification with a constant, given its own name
  since a literal reads differently from a variable.

Three times exist in the language: a transition reads the previous round by definition, an
identified port is read as it stands at the reader's turn under the orders (section "The orders"),
everything else is same-round (a terminal target at the plan's horizon, not at a lag).

**`merge` and `couple`.** `merge(subsystems, params, bindings) -> (System, SystemParams)` is the
splice (namespacing, concatenation of tables, bindings rewritten into the graph) with the three
kinds above, and nothing more: it builds no joint planner. `couple(components, bindings, fixed) ->
(System, SystemParams)` is `merge` restricted to identifications of an out-port of one component
with an in-port of another, plus `Clamp`s for `fixed`; a structural `Link` or an identification of
latents or variables is refused, naming `merge`. Both return a System with the `subsystems`
metadata, which records the partition (whose internal states are whose: the fact a planner's view
and `describe`'s blankets are read from), so a coupling can be merged, coupled, described, priced
and nested like any System. `Wiring` goes. Planner-level configuration (horizon, precision, habit,
durations, planner fields) is per component and travels in the metadata; a joint planner over
several components exists only where elected, by a step on a non-leaf node in `run`
(`steps={"agent": plan()}` plans the merged composite as one mind and composes the habits over the
joint policy set; `steps={"agent.mission": plan(), "agent.rover": plan()}` plans the views), with
its configuration from the election's arguments, the parts' values where they agree, and a refusal
where they disagree and nothing was given. The `_agree` helper and the one-durations rule go: a
joint planner over parts whose durations differ is refused at election, not at merge. The same-step
identification `merge` performs today hands the driven node the sender's intended level, before its
kernel; `couple` hands the realised one, after it; C13 says the realised value crosses, so both
converge on the realised value. A `mirror` of a merged composite can be taken per component, each
part a leaf with the cross-component parents of its transitions as driven inputs, which is how a
self-modelled composite keeps within-round freshness under sequential and async (the arena of the
driver tests); and a plain convenience couples any System with its own mirror, with no special
properties. The joint planner's view of a node is built by `joint_planner(system, params, node,
**config)`, which a step on a non-leaf node calls; electing one over a node that contains a leaf
stepped by `sample()` is refused, since the joint view would take a process's tables as part of its
model (a self-model is a model leaf and elects cleanly). A coupled leaf's view is its own System
unsliced, so it accepts recognition and scoring as a wired mind does today; the splice carries them
per leaf. The MDL's graph code counts variables, not names (Alex, 2026-11-08).

**`run`.** `run(system, params, ticks, key, steps, clocks=None, tick_order="simultaneous",
carries=None, record_carries=False) -> Trajectory`, over a closed System. `steps` names every
leaf component (dotted through nested merges, `"agent.rover"`; a missing one is refused, nothing
is implicit) and gives each its step, `step(system, key, obs, carry, params) -> (action, carry',
params')`, handed its view's System and params each round (a step compiles on the System it is
handed, the same object every round at one level). Two default steps are provided, named by
what they do, and a custom one is any callable of that shape:

- `plan(planner=None, learn=None, trace=False)`: infer over the modalities with the carried
  beliefs as priors, choose the free control nodes with the planner (`None` is the enumerated
  default; a driven one is sliced by its input), push the empirical prior, realise the action
  nodes; `learn` names what to count and `trace` keeps the planner's metrics in the carry's
  `trace` slot;
- `sample()`: every control node driven or free, step the factors forward, draw the
  state-parented action nodes from their emission kernels and the control-parented ones from
  the driven or drawn levels (today's `ancestral_step`, now the full process step);
- a custom step: hand-written dynamics (the survey world) or hand-written learning (the survey
  rover); one that plans calls `system.act(planner=...)` itself.

`clocks` maps a component to its `ticks_per_unit`: the component decides once per option,
reading the duration off its chosen level, and holds the action for the rounds in between,
which is `hold` as an argument; `hold` goes. The per-round body is step 1's: each component's
view is taken from the shared params at the levels its linked control nodes stand at; its
realised in-ports (belief reads and port reads alike) are supplied; what it returns is written
back; its action nodes are realised and stand for the rest of the round. The orders mean what
they mean today. The composite form goes, and with it running a System against its own model as
a mode: that case is `couple({"model": (cs, cp), "world": mirror(cs, cp)}, links=...)`, written
out. `particle` goes: a merged composite inside a coupling is leaf components with their own
entries, the recursion of "Beyond step 1" being `run`'s body over the leaves. `agent_step` and
`ancestral_step` go, replaced by `plan()` and `sample()`, which take no System: `run` hands it.

**Closedness.** `run` refuses an open System: an observation modality no link feeds, and a link into
a modality a belief read already supplies. No control node can be unaccounted for, since every leaf
component plans, steps or samples, and a chosen node a planner must not plan over is a `Clamp`. The
checks of step 1 stay (a reader with no uniform view needs its chooser listed first under sequential
and is refused under the other orders; no non-identity actuator kernel on a node another view
reads).

**The trajectory.** As ruled: a tree mirroring the System's nesting, `params` at the root,
leaves with `received`, `emitted`, `intended`, `carries` (per round, opt-in) and `carry`,
dotted lookup, composite nodes' `received` and `emitted` as prefixed unions, diagnostics in the
carry's `trace` slot. `MindRecord`, `ProcessRecord` and today's `Trajectory` go.

**Keys.** The tick's key splits into `2N + 1` keys for `N` leaf components in the System's
component order: component `j` steps with key `j`, its action nodes are realised with key
`N + j` under sequential and async and every component's with the last key under simultaneous;
reads draw with `modality_key(k_tick, m)`. A new key stream for every run with a world, so no
bitwise check against today's runner. Acceptance: each pinned test re-pinned only after a probe
shows the old and new runs agree when handed the same per-round keys; the survey reproduced
statistically over several seeds (coverage at the four marks within the old runner's seed
spread, decisions on the slow clock, the energy floor); the page bundle regenerated from the new
runner at the end of the arc.

**The survey world as a System.** New work: factors for position, terrain and energy, the
control node for the velocity command, and three state-parented action nodes (fix, image,
gauge) with emission kernels of the right families and types; the hand-written dynamics stay as
its custom step. The tables of a world that is never sampled are placeholders, so the build gives
a world whose dynamics are hand-written a small constructor for its names and port types alone
(`ports(factors=..., controls=..., emissions=...)`) if the full `System` constructor proves
heavy for it, and a pymdp-style positional environment still enters through an adapter that
makes a step of it given the declaring System (today's `GenerativeEnv.wrap`).

**Step 2 as a caller sees it.** The two-tier survey rover, the whole of today's runner:

```python
cs, cp = merge(
    subsystems={"mission": ms, "rover": rs},
    params={"mission": mp, "rover": rp},
    bindings=[
        Identify("mission.a_option", "rover.u_opt"),  # the option, realised, drives u_opt
        Identify("rover.field", "mission.field"),
        Link(source="rover.x", target="mission.cell_obs", kernel=grid, realise="mode"),
        # sugar: an emission of the rover from its belief over x through the cell indicator,
        # realised by mode, identified with the scheduler's cell modality
    ],
)
ws, wp = wd.world_system()   # position, terrain, energy; v in; fix, img, gauge out
system, params = couple(
    components={"agent": (cs, cp), "world": (ws, wp)},
    bindings=[
        Identify("agent.rover.a_v", "world.v"),
        Identify("world.fix", "agent.rover.fix"),
        Identify("world.img", "agent.rover.img"),
        Identify("world.gauge", "agent.rover.gauge"),
        Identify("world.gauge", "agent.mission.gauge"),
    ],
)
rec = run(
    system, params, ticks=400, key=key, tick_order="sequential",
    steps={
        "agent.mission": plan(trace=True),       # the default step; the enumerated planner
        "agent.rover": rover_step,               # custom: learns the field, decodes the image
        "world": wd.world_step,                  # custom: the hand-written dynamics
    },
    clocks={"agent.mission": 1},                 # one option = its duration in rounds
    record_carries=True,
)
rec.params                                  # the whole System's params, the field as learned
rec["agent.mission"].emitted["option"]      # the options, held twelve rounds each
rec["agent.mission"].carries[t].trace.tree  # the scheduler's term tree at a decision
rec["world"].carries[t]["x"]                # the true position
describe(system)                            # the whole coupling as one model document
```

The scheduler needs no custom step: the default plans, and its metrics are in the trace. The
rover keeps one because it learns the field and decodes the image for the page; the world keeps
one because its dynamics are hand-written. A world that is a model run forward is
`mirror(system, params)` with `sample()` as its step.

**What the world's two functions are**, written out, since an example that calls convenience
functions hides what a user writes (Alex, 2026-11-07; the exact spelling of a state-parented
action node is the build's, N5 being unbuilt, but the size is honest):

```python
def world_system():
    """The world's ports: three state factors, one driven control node, three emissions."""
    system = System(
        state_names=("terrain", "x", "energy"),
        B_deps=((1,), (1,), (2, 0, 1)),               # terrain and x from x; energy from all
        controls=(FLOAT,),                            # v: the velocity command, width 2
        control_names=("v",), action_names=("a_v",),
        emissions={"fix": ("x",), "img": ("x", "terrain"), "gauge": ("energy",)},
    )
    params = SystemParams(
        control={"v": Box(-jnp.ones(2), jnp.ones(2))},
        emission={                                    # the kernels that type the out-ports
            "fix": LinearGaussianObservation(H=jnp.eye(4)[:2], c=jnp.zeros(2), noise=jnp.eye(2)),
            "img": LinearGaussianObservation(H=jnp.zeros((IMG_PIXELS, 4)), ...),
            "gauge": ConditionalCategorical(log_table=jnp.log(jnp.eye(4)), parent_ndims=1),
        },
        transition={}, initial_state_prior={},        # never sampled: the step is hand-written
    )
    return system, params


def world_step(system, key, obs, carry, params):
    """The true dynamics: the carry is the state; obs holds v; the emissions are returned."""
    state = carry if carry is not None else reset_state()
    k_move, k_obs = jax.random.split(key)
    x = move(state, obs["v"], k_move)                  # F, G, walls, stall, as today's step
    new = {"terrain": terrain_of(x[:2]), "x": x, "energy": energy_after(state, x)}
    return observe(k_obs, new), new, params            # {"fix", "img", "gauge"}, the state, params
```

Today's `world()` is the same three declarations plus a positional shell and a hand-built name
table, which the `GenerativeEnv` type imposes and which go. The rule for the showcase notebook
and the page: every convenience function an example calls is defined in the example, in a cell
the reader can see, or inlined.

**Open for Alex.**

- The binding kinds: `Identify` (two-way sameness; one-way flow follows from factor roles), the
  structural `Link`, `Emit` with the realisation, `Clamp`; the realised `Link` as sugar (ruled
  2026-11-08, replacing the three kinds of 2026-11-07).
- `steps` naming every leaf component, with `plan()` and `sample()` as the default steps (revised
  2026-11-07 after Alex's questions; `planners=` dropped).
- The key stream changes; the acceptance above replaces bitwise.
- The survey world declared as a System, the price of one kind of participant.
