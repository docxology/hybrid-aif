# Design: emissions, the mirror, and the three bindings

*Status: BUILT 2026-11-08 (chunk A of `engine-consolidation.md`, step 2). Designed by an Opus
pass on 2026-11-07, ruled by Alex and Fable the same day, built and reviewed on 2026-11-08; the
design document and the review were lost with a session's scratch directory and this note is
their regeneration from the record, as built. Semantics: `nesting-and-coupling.md` N1, N5, N8,
A11; `control-states.md` C12, C13; `action-node-modalities.md` S2; `engine-consolidation.md`
"Ports are modalities in and action nodes out; a control node is never a port".*

## 1. The emission: an action node of a state factor

**What it is.** An out-port of a System realised from its state rather than from a choice. A
control-parented action node is an intention made real: a policy posterior, a duration, an
actuator kernel keyed by the chosen level, a place in the planner's control space. An emission is
the state showing itself, with no choice behind it, which is why nothing in the emitting model
reads or scores it (S2). The two out-ports differ in what explains them, so they are declared
apart whatever the core's indexing (`engine-consolidation.md`, queued item).

**Declaration.** `System.emissions`, a mapping from a name to its parents (a draw) or to
`Emit(parents, realise="sample")` (`hybrid_aif.reads.Emit`, beside `Read`), stored as `(name,
Emit)` pairs in the order given. Parents are state factors and control nodes, named, in the order
of the kernel's parent axes (control nodes because the mirror needs them: a modality that read a
control node becomes an emission that reads it). Not an entry of `action_names`: that list is
paired by position with the control nodes in about ten places (`FaceNames.actions[k]` with
`kinds[k]`, `_actuator`, merge's node zip, `describe`, `keyed.actions_in/out`, `act`, the driver's
action check, `A_deps`' offset by the factor count), and an emission is paired with nothing.

**Kernel.** `SystemParams.emission[name]`, any likelihood family (tabular, linear-Gaussian,
neural), learnable forms and `Parameter` references included (the design refused them; the
build accepts them because the mirror of a learning agent must be a valid System and `emit` and
the forward process use the point form). Validation checks that each emission has a kernel of a
likelihood family that gives a port type, with arity and per-axis sizes matching its parents. A
control node only an emission reads is sized by the kernel (`control_space(..., emissions=)`).

**Port type and ports.** Read off the kernel as a modality's is (`_observation_type`): `(LEVEL,
levels)` for a table, `(FLOAT, width)` for a linear-Gaussian or neural kernel. `System.ports(
params) -> Ports(ins, outs)`: `ins` are the modalities and each control node under its own name
(the driven regime's input, a port-list convention: `A_deps` and `modality_names` do not grow and
no identity-likelihood modality exists for the inference engine, which would have had to share
the node's name, be skipped by every face that iterates modalities, and read a control-node
parent `A_deps` refuses); `outs` are the action nodes, typed as their control nodes, and the
emissions. A port has no node index: a port is its name.

**Realisation.** `System.emit(key, params, values) -> {name: value}` inside the step, not the
runner: `values` holds each parent by name as a point (a world's state, a realised control level)
or as a belief (an agent's posterior). With `split(key, E)[e]` for emission `e`: from points,
`sample` conditions the kernel on the point and draws (exactly the forward process's observe
step); from a belief, `sample` draws the parents with a folded key and then the kernel, so a
zero-entropy belief reproduces the point draw; `mode` and `stats` from a belief are the
predictive, a table contracted against the parents or a linear-Gaussian mixture moment-matched.

**Isolation (S2).** Refused with a sentence, each: an emission as the source of a `lagged`
read; as a `C_deps` key or parent; as a `readings` entry; as `branch_on`; as an `obs_targets`,
`state_targets` or `terminal_targets` key; as an `epistemic` key; in a merge, as a belief read's
source, in a structural tuple, into a control node or a terminal target, as a link's target, or
under `Clamp` or `Identify`. `act` returns action nodes only; the default `learn` leaves a
learnable emission kernel unchanged; `_is_hybrid` ignores emissions, so a tabular agent with a
Gaussian emission stays on the tabular path.

**Merge and views.** `merge` namespaces emissions and their kernels; a control parent follows a
same-step identification; a parent removed by a relay or a clamp is refused. `Subsystem.emissions`
records a component's own; `subsystem` carries them as outputs only and refuses an emission that
reads a partner's control node.

**Description and MDL.** No label of its own: the actuator record is widened to any out-port
kernel with any parents, the categorical kernel on its own control node being the special case
(Alex: an emission is the paradigm case of an actuator). Under an emission the record takes any
family of the likelihood menu, parents are latents and control nodes at the same step,
`options.realise` is written when not the default; `LinearGaussianObservation` and
`MLPConditional` gained the actuator label. Every pinned description fixture stayed valid. The
MDL prices `emission:<name>` as a likelihood of its family (an emission costs exactly what a
same-declaration modality costs), and the graph code counts emissions as `log*(E + 1)` always,
the prefix-free form, which moved every model by `log*(1)`, about 1.05 nats; two pinned tests and
seven stored numbers in `demo.ipynb` moved by exactly that. Actuator kernels themselves stay
unpriced (`description-length.md` D2, note of 2026-11-08).

## 2. The mirror

`hybrid_aif.system.mirror.mirror(system, params, *, lower=False, param_key=None) -> (System,
SystemParams)`: the world a System models, as a System that acts. Every modality becomes an
emission of the same name over the same parents, with the likelihood as its kernel (a modality
that observed an action node becomes an emission reading the control node it realises). Kept:
factors, transitions, priors, `D_deps`, `dims`, `cells`, control nodes, `durations`, `control`,
and the parameter variables (`parameters`, `references`) as they are, since a world learns if its
step learns. Kept but inert: the action nodes with an empty actuator, one per control node being
required. Dropped: targets, `C_deps`, `H_deps`, `epistemic`, `terms`, `scoring`, planner fields,
`readings`, `recognition`, `lagged` (a read is a datum source for inference; today's forward
process samples such a modality from its likelihood, which is what keeps `as_env` bitwise).
Refused: a System that already declares emissions.

**The habit.** `policy_prior` is converted at mirror time, on the original System's policy set,
into a categorical over the distinct first-step level tuples, each holding the mass of the
policies whose first step equals it; the mirror's `policy_len` is 1. A habit that cannot be so
converted is refused at mirror time. (The review's finding: the mirror drops the planner, so a
sophisticated agent's habit over five policies met twenty-five enumerated ones.)

**Lowering.** `lower=True` replaces every learnable declaration with its posterior mean, or
with one draw per slot under `param_key`, and drops `parameters` and `references` so the forward
process sees plain tables (the review's finding: kept references put the Dirichlet back and the
process refused it). A `param_key` without `lower` is refused.

**Sampling forward.** `System.sample_initial(key, params, state=None)` and
`System.sample_step(key, params, state, inputs)` run a System's emissions forward from a point
state: a control node an input drives takes the input; a free one is drawn from the habit with
`split(fold_in(key, CONTROL_TAG), K)[k]` (`CONTROL_TAG` in `envs.generative`), uniform without
one, so the process's own key stream is untouched; every emission is realised by its declared
rule, `mode` and `stats` included (a world with a `mode` emission must step). With every node
driven these equal `as_env`'s reset and step.

**`as_env` over the mirror, bitwise.** `as_env` is `mirror(lower=True, param_key)` followed by
the unchanged forward process (`System._forward`: the discrete process when transitions, priors
and emissions are all tabular, the hybrid one otherwise), with the actuator re-applied at the
env face and the faces keyed by the original System's names. Key use of the process, unchanged:
reset splits the key into one per factor plus one, the modalities from the first; step draws the
actuator from `fold_in(key, ACTUATOR_TAG)`, then splits into an observation key and a state key,
factors from the state key and modalities from the observation key. Eight cases recorded from the
previous commit before any change (`tests/data/as_env_*.npz`, `tests/mirror_cases.py`: a tabular
T-maze; a modality over an action node; a non-identity actuator; `D_deps`; `param_key`; a
switching linear-Gaussian with a categorical; an MLP likelihood; a masked `factors=` step) are
reproduced exactly; the review added three (a discrete and a continuous node with a
linear-Gaussian transition; the lagged learnable system under `param_key`; shared transition and
likelihood variables under `param_key`), also exact over thirty ticks.

## 3. The three bindings

*Superseded in part on 2026-11-08 (`engine-consolidation.md`, "The bindings: identify, link,
emit"): `Identify` is the only binding that joins variables and is always two-way, one-way flow
following from factor roles; a realised `Link` is sugar for an emission identified with a modality;
the structural `Link` stays. What follows is the code as chunk A left it, which chunk B rewrites.*

Replacing `ObservationLink`, `SiblingRead`, `Wire`, `ActionControl`, `SharedParameter`,
`BoundaryPrior` and `TargetEmission`, removed without aliases (a test asserts they are gone):

- **`Link(source, target, realise=None, kernel=None)`**, an arc of the graphical model of the
  whole; `source` a node or a tuple of nodes. From a realised port (an action node; an emission
  once the runner realises one) to a modality or to a control node: the port's realised value,
  with no `realise` of its own, since the port is realised once by its kernel (C13) and every
  link from it carries that value (today's `Wire`, and `ActionControl`'s same-step form, spelled
  from the driver's *action node*: `Link("mission.a_option", "rover.u_opt")`). From a state
  factor to a modality with `realise` given (`sample`, `mode` or `stats`, no default) and an
  optional `kernel` (a `Grid` over a continuous source): a belief read, one way, the source's
  posterior of the previous round realised (today's `SiblingRead`). From state factors to a
  modality with no `realise`: a structural arc, the sources the modality's complete parent list
  with messages both ways (today's `ObservationLink`); a `kernel` without `realise` is refused so
  the two cannot be confused. From a state factor to a control node: the relay form of
  `ActionControl`, structural, read a step later. From factors to `"terminal:<factor>"`
  (`"terminal:agent.z"` under a prefix): the terminal target of that factor conditioned on them
  (today's `TargetEmission`); `terminal` joins the key kinds of the selector grammar for this
  use only, and `epistemic=("terminal:*",)` stays refused.
  Refused: `realise` on a link from a realised port; a control node as a source (the sentence
  names its action node); an action node or emission into a modality under `merge` (the
  sentence names `couple`); a relay with `realise`.
- **`Identify(a, b)`**: two parameter variables become one, keeping `a`'s declaration (today's
  `SharedParameter`). Only variables: driving a control node is a `Link`.
- **`Clamp(node, level)`**: a control latent no planner chooses and no link drives, clamped at a
  constant: a level (every reading table sliced), or a probability vector averaged over once at
  merge time (today's `BoundaryPrior`, and `couple`'s `fixed`).

Composites built with the old classes and with the new forms are bitwise equal in every static
field and every params leaf. `couple(links=[Link(...)])` replaces `wires=`, with `Wiring.links`;
a mind's out-ports under `couple` are its action nodes only, and a link from a mind's emission
is refused with a sentence, until the runner realises emissions (`engine-consolidation.md`,
step 2's remaining chunks). The boundary classification of A11 is unchanged: a structural link,
a relay or an identification shares a latent (`shared`); a belief read or an action-node link is
blanket-shaped (`blanket`).

## 4. Acceptance as run

Ruff clean; the full suite green; the notebook gate green; the survey showcase at 400 ticks,
both settings, bitwise equal to the previous commit (no key stream moves in this chunk); 79 new
tests (`test_emissions` 38, `test_mirror` 20, `test_bindings` 21) plus the review's probes
folded into the fix pass. Open: actuator kernels unpriced by the MDL; a dotted emission name on
a non-composite System is accepted as modality names are; the mirror also drops
`external_inputs`, `decomposition` and `gamma`, which no probe exercised.
