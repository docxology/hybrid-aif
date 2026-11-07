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
and reproduces the recorded survey runs bitwise (level 1 and level 2, 400 ticks, against the
runner as it stands at the step's start) unless the step says why not.

1. **Params flow through the runner.** One step shape for every participant,
   `step(key, obs, carry, params) -> (action, carry', params')`. The runner holds params per
   participant across rounds and records them. Detailed below.
2. **One `run`.** The wired form absorbs the composite form's per-component stepping under
   sequential and async; the composite form goes. A merged System with several planners runs
   under `run` by taking each component's view from the shared params each round, which is
   `particle`'s body moved to where it belongs; `particle` goes. `couple` returns a System whose
   wires are edges with a declared realisation; `Wiring` goes.
3. **Processes are Systems.** A participant of `couple` is a System with params and a step; a
   world is a System whose step samples its model forward (`as_env` is the face it uses).
   `GenerativeEnv` stops being a participant type.
4. **Clocks in `run`.** `hold` becomes a per-participant clock argument; the duration is still
   read off the chosen option's `durations`.
5. **Vocabulary.** Each term defined where first used in the docs; the terms that no longer
   name anything dropped (mind, respond, process, particle as API words).

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
cs, cp = merge({"mission": ms, "rover": rs}, {"mission": mp, "rover": rp},
               [ActionControl("mission.u_option", "rover.u_opt"),
                SharedParameter("rover.field", "mission.field"),
                SiblingRead("mission.cell_obs", "rover.x", grid, "mode")])
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
