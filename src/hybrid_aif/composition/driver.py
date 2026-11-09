"""The driver: participants on a clock, in either of two forms.

A participant of a round is a :data:`Step`, ``step(key, obs, carry, params) -> (action, carry',
params')``: it is handed its parameters each round and returns them, the very same object when it
did not learn, so a learner and a non-learner are one shape and the runner keeps what comes back.
A step that holds a planner is informally a mind. :func:`agent_step` and :func:`ancestral_step`
build the two ends of that spectrum, and :func:`hold` slows a step to one decision per option.
:func:`run` closes the loop in one of two forms:

- **Wired** (:func:`~hybrid_aif.composition.couple.couple`): separate particles, minds and
  processes joined by typed wires, from an action node to a process input or a mind's modality,
  and from a process observation to a mind's modality. A mind is declared as ``(System, params,
  step)``; ``run`` starts it from those params, hands it the current ones each round and records
  the last ones. Each action node is realised from the mind's ``params.actuator`` (as declared
  at ``couple``) when the mind
  chooses, and every wire from it carries the realised value; the processes step on their
  realised inputs. The record is one entry per participant, :class:`MindRecord` or
  :class:`ProcessRecord`.
- **Composite** (:func:`~hybrid_aif.composition.merge.merge`): one particle, whose components
  are the steps' views; the world is the composite's ``as_env``, with the wires by name:
  observations route in by the ``subsystems`` metadata (each step sees its own modalities, keyed
  by name); actions route out by action-node name, each step's action nodes merged into the
  composite action, realised at its nodes, on which the world moves. The composite's params are
  the one parameter home: each round every step is handed the view of its component taken from
  the home, with each control node another component chooses set to the level it last chose, a
  learner's returned parameter variables are written back to the home
  (:func:`~hybrid_aif.composition.merge.write_back`) and the next view is taken from the new
  home. The record is a :class:`Trajectory`, whose ``params`` is the home after the last round.

A merged composite is also one participant of the wired form: :func:`particle` turns it, with one
step per view, into a single step for :func:`~hybrid_aif.composition.couple.couple`, so a particle
meets an outside world through wires; it runs the same per-round body as the composite form, with
the params the wiring hands it as the home. The composite form's world is the composite's own
model; a particle with an outside world is the wired form.

In both forms a realised action is drawn once, when its step chooses, and every wire from the
node carries it for the rest of the round; no other step's choice redraws it.

A belief is internal: each participant's posterior lives in its own carry and never crosses as a
distribution. Participants reach each other only through the wires (in the composite form, through
the world, or through a sibling-belief read: a modality that observes an emission of another
component, which that component's step realised from its posterior and carries, by ``sample``,
``mode`` or ``stats``), and what a wire carries is the realised value of an action node, which
may be the belief's statistics.

A round is one tick of time under every order, and every factor steps once per round; the
orders are algorithms for one update, not amounts of time. ``"simultaneous"`` (default): every
step responds to the round's boundary, then every process (in the composite form, the world)
steps once on the round's actions. ``"sequential"``: steps run in declaration order (the
order given to :func:`~hybrid_aif.composition.couple.couple`, or the order of ``minds`` in the
composite form), each reading current values, fresh where the source has already stepped this
round: in the wired form a direct wire carries an earlier mind's act to a later mind, and the
processes step once at the end of the round; in the composite form each component's factors
step at that component's turn, and a component with no step goes at the end. ``"async"``: as
sequential, in a fresh random order per round (reproducible from the run key). Each records
one row per tick. The driver is a Python loop, so carries need no shared structure; each
step may be jitted.

Design: docs/design/nesting-systems.md (architecture sketch; tick semantics);
docs/design/control-states.md C13; nesting-and-coupling.md N1, N3, N4.
"""

import dataclasses
import math
import numbers
from collections.abc import Callable, Mapping
from typing import Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.composition.couple import Wiring
from hybrid_aif.composition.merge import _inputs_of, retake, subsystem, write_back
from hybrid_aif.controls import FLOAT, LEVEL
from hybrid_aif.distributions import Categorical
from hybrid_aif.envs.generative import ACTUATOR_TAG
from hybrid_aif.planning.base import Planner
from hybrid_aif.reads import EMIT_TAG
from hybrid_aif.system import Belief, LearnState, System, SystemParams, keyed

TickOrder = Literal["simultaneous", "sequential", "async"]


class MindCarry(NamedTuple):
    """One mind's internal state between ticks, the only place its beliefs live.

    ``beliefs`` is ``None`` before the first tick (use the declared initial priors): the empirical
    prior the next tick filters from, the last posterior pushed through its transition under the
    chosen action. ``emitted`` maps each emission of the mind's System to the value its last round
    realised from the posterior (``System.emit``, with the step's key folded by ``EMIT_TAG``), which
    is what a modality that observes the emission takes as its datum on the next round (a custom
    step that keeps none, or a carry seeded through ``carries=`` without one, is read as absent on
    the next round). ``learn_state`` holds the carry of ``System.learn`` once a mind learns online.
    Carries are never merged, so two minds share no leaf.

    Design: nesting-and-coupling.md N1.
    """

    beliefs: Mapping[str, Belief] | None = None
    learn_state: LearnState | None = None
    emitted: Mapping[str, Array] | None = None


#: One participant's round: ``obs`` is its own modalities keyed by name, ``action`` an int32 level
#: per action node (a float vector for a continuous one), keyed by action-node name as
#: :meth:`System.act` returns it. The params returned are the very same object as the ones given
#: unless the step learned; the runner treats identity as "did not learn".
Step = Callable[
    [Array, Mapping[str, Array], MindCarry, SystemParams],
    tuple[Mapping[str, Array], MindCarry, SystemParams],
]


def _check_legal(system: System, params: SystemParams, legal, who: str) -> None:
    """Refuse a ``legal`` mask that masks every level, on the concrete mask: inside the compiled
    tick the mask is traced and cannot be refused there."""
    if system._control_layout(system._params_in(params, who)) is None and not isinstance(
        legal, jax.core.Tracer
    ):
        system._legal_logits(system._params_in(params, who), legal, system.policies(params), who)


def agent_step(system: System, planner: Planner | None = None, **act_kwargs) -> Step:
    """A full agent as a :data:`Step`: infer, act, carry the empirical prior, hand the params back.

    ``system`` is the participant's System (typically
    :func:`hybrid_aif.composition.merge.subsystem`'s view): ``infer_states`` with the carried
    beliefs as priors, ``act``, then ``update_empirical_prior`` under the chosen action, and the
    emissions of the System realised from the posterior into the carry (``MindCarry.emitted``).
    The params come with each call and are returned as given, since this step never learns; a
    learner is a hand-written step. ``act_kwargs`` pass through to :meth:`System.act`; a ``legal``
    among them (keyed by control node, or one bool per policy) is checked on the first call,
    where the params are concrete. A System with no designations gives a Bayesian perceiver (flat
    ``q_pi``, the same filtering). A modality whose observation is ``None`` (a sibling's belief
    the driver could not yet read) is absent that round. The step carries ``system`` as an
    attribute, which :func:`hold` reads.
    """
    kwargs = tuple(sorted(act_kwargs.items()))
    legal = act_kwargs.get("legal")
    checked = legal is None

    def step(key, obs, carry, params):
        nonlocal checked
        if not checked:
            _check_legal(system, params, legal, "agent_step")
            checked = True
        action, carry = _agent_tick(system, params, planner, kwargs, key, obs, carry)
        return action, carry, params

    step.system = system
    return step


@eqx.filter_jit
def _agent_tick(system, params, planner, kwargs, key, obs, carry):
    """One compiled tick: infer, act, push; ``params`` is an argument, so the program is shared."""
    #  a modality the driver left absent this round (a sibling's belief not yet carried) is None
    absent = {m: False for m, v in obs.items() if v is None}
    beliefs, _ = system.infer_states(params, obs, priors=carry.beliefs, present=absent or None)
    action, _, _ = system.act(key, params, beliefs, planner=planner, metrics=False, **dict(kwargs))
    pushed = system.update_empirical_prior(params, beliefs, action)
    if not system.emissions:
        return action, carry._replace(beliefs=pushed)
    names = system._names()
    values = {
        **beliefs,
        **{u: action[a] for u, a in zip(names.controls, names.actions, strict=True) if a in action},
    }
    emitted = system._emit(jax.random.fold_in(key, EMIT_TAG), params, values)
    return action, carry._replace(beliefs=pushed, emitted=emitted)


def ancestral_step(system: System) -> Step:
    """An env-like component as a carry-free :data:`Step`.

    The action is drawn from the declared prior over the participant's own control nodes and
    emitted keyed by their action nodes: ``policy_prior``'s first step if declared, else uniform
    per node. A component with no control nodes responds with the empty action. The carry and the
    params pass through. The level counts are read from the params on the first call, where they
    are concrete (a System with continuous control nodes (A9) is refused there); the habit
    (``policy_prior``) is read from the params handed in on every call.
    """
    fixed: list = []

    def prepare(params):
        """The static structure: the draws' level counts and the policy enumeration."""
        if any(k != "level" for k in system._space(params).kinds):
            raise NotImplementedError(
                "ancestral_step draws a discrete control level per control node, and this System "
                "has continuous control nodes (A9); supply a Step that samples from their boxes "
                "in params.control instead"
            )
        if system._table.sugar is not None:  # one draw per state factor, as the sugar lays them out
            levels = tuple(
                n if c else 1
                for n, c in zip(
                    system._num_controls(params, "ancestral_step"),
                    system._ctl("ancestral_step"),
                    strict=True,
                )
            )
        else:
            levels = system._control_levels(system._params_in(params, "ancestral_step"))
        fixed.append([levels, None])

    names = system._names()

    def step(key, obs, carry, params):
        del obs
        if not fixed:
            prepare(params)
        entry = fixed[0]
        levels = entry[0]
        policy_prior = params.policy_prior
        if policy_prior is not None:
            if entry[1] is None:
                entry[1] = system.policies(params)
            logits = (
                policy_prior.log_probs
                if isinstance(policy_prior, Categorical)
                else jnp.asarray(policy_prior)
            )
            chosen = entry[1][Categorical(logits=logits).sample(key), 0].astype(jnp.int32)
            return keyed.actions_out(names, chosen), carry, params
        keys = jax.random.split(key, len(levels))
        draws = [jax.random.randint(k, (), 0, n) for k, n in zip(list(keys), levels, strict=True)]
        drawn = jnp.stack(draws).astype(jnp.int32) if draws else jnp.zeros((0,), jnp.int32)
        if system._table.sugar is None:
            return keyed.actions_out(names, drawn), carry, params
        return keyed.factor_actions_out(names, drawn), carry, params

    step.system = system
    return step


class Trajectory(NamedTuple):
    """What :func:`run` produced, recorded at tick boundaries.

    ``states``/``observations`` carry ``ticks + 1`` rows per composite factor/modality, keyed by
    name (row 0 is the reset); ``actions`` maps each composite action node (by name) to its
    ``(ticks,)`` intended levels, the composite action the minds emitted as of each tick's last
    sub-step, and ``realised`` to the realised levels standing after the tick's last sub-step,
    on which the world moved (equal to ``actions`` for an action node without a kernel in
    ``params.actuator``); ``carries`` holds
    each mind's final carry.
    """

    states: dict[str, Array]
    observations: dict[str, Array]
    actions: dict[str, Array]
    carries: dict[str, MindCarry]
    realised: dict[str, Array] | None = None
    params: SystemParams | None = None


class MindRecord(NamedTuple):
    """What a wired :func:`run` recorded of one mind.

    ``observations`` maps each of its modalities to ``ticks + 1`` rows as the mind received
    them: row ``t < ticks`` is what it responded to at tick ``t``, row ``ticks`` what its
    in-ports hold after the last tick. Under ``"sequential"`` and ``"async"`` a later mind's row
    ``t`` already holds the acts earlier minds made at tick ``t``. Under ``"simultaneous"`` the
    rows are the tick boundaries a :class:`ProcessRecord` keeps. ``actions`` and ``realised``
    map each of its action nodes to ``ticks`` rows: the intended action it emitted, and the
    value the node realised from it, standing after the tick's last step. ``carry`` is its
    final carry, and ``params`` its params after the last round (the starting object if its
    step never learned).
    """

    observations: dict[str, Array]
    actions: dict[str, Array]
    realised: dict[str, Array]
    carry: MindCarry
    params: SystemParams


class ProcessRecord(NamedTuple):
    """What a wired :func:`run` recorded of one process: ``ticks + 1`` rows per factor
    (``states``) and per emitted modality (``observations``), keyed by name; row 0 is the
    reset, row ``t + 1`` the end of tick ``t``."""

    states: dict[str, Array]
    observations: dict[str, Array]


class HoldCarry(NamedTuple):
    """A held step's carry: the wrapped step's own, the rounds still to hold, the held action."""

    inner: MindCarry
    remaining: int
    action: Mapping[str, Array]


def hold(
    step: Step,
    ticks_per_unit: numbers.Real | Array = 1,
    *,
    system: System | None = None,
) -> Step:
    """A step that decides once per option: ``step`` held for the option's duration.

    On a decision round the wrapper calls ``step``, stores its action and reads the duration with
    :meth:`System.elapsed` at the params it was handed; the action is then returned unchanged,
    with the params handed in, for ``duration * ticks_per_unit`` rounds in all (the decision
    round included), counting down on the host and calling nothing, so the participant infers and
    plans once per option. The first round always decides, whether ``run`` seeds a bare
    :class:`MindCarry` or ``carries=`` supplies one; the wrapped carry is a :class:`HoldCarry`,
    and a ``HoldCarry`` supplied in ``carries=`` resumes a count. What a decision round returns
    for the params is the wrapped step's, so a held learner learns once per option.

    The System that reads the duration is ``step.system`` (:func:`agent_step` and
    :func:`ancestral_step` set it; a hand-written step sets the attribute) or the ``system``
    keyword. The params it is handed must declare ``params.durations``; this and the rest of
    what needs concrete params is checked on each decision round, the first of which is the first
    call. Durations are in the preference's unit; ``ticks_per_unit`` converts that unit to rounds
    (a positive number, checked at construction), and a duration whose product with it is shorter
    than one round or not a whole number of rounds (to a relative 1e-5) is refused when the
    action is read. A non-identity actuator kernel (``params.actuator``) on any action node is
    refused, since the driver realises every action node every round, so a draw would repeat each
    round of the option; an identity kernel is accepted.

    In the composite form (:func:`~hybrid_aif.composition.merge.merge`) the held step's System
    must own no world process: its factors are beliefs and the world is a separate component,
    or the composite environment would apply the option's whole-step kernel once per round of
    the fast tier. The wired form (:func:`~hybrid_aif.composition.couple.couple`) has no such
    restriction. This is not checked. Design: step-durations.md T9; nesting-and-coupling.md N4.
    """
    scalar = isinstance(ticks_per_unit, numbers.Real) or (
        isinstance(ticks_per_unit, np.ndarray | Array) and ticks_per_unit.ndim == 0
    )
    if isinstance(ticks_per_unit, bool | np.bool_) or not scalar:
        raise ValueError(
            f"hold: ticks_per_unit={ticks_per_unit!r} is not a number; pass a positive real "
            "number or a scalar array, the rounds one unit of duration lasts"
        )
    ticks_per_unit = float(ticks_per_unit)
    if not ticks_per_unit > 0 or not math.isfinite(ticks_per_unit):
        raise ValueError(
            f"hold: ticks_per_unit={ticks_per_unit!r} is not positive and finite; pass the "
            "number of rounds one unit of duration lasts"
        )
    attached = getattr(step, "system", None)
    if system is not None and attached is not None and system is not attached:
        raise TypeError(
            "hold: system= is a different System from step.system; pass one, or the one the "
            "step is built over"
        )
    system = system if system is not None else attached
    if not isinstance(system, System):
        raise TypeError(
            "hold: the step names no System to read durations from; build it with agent_step or "
            "ancestral_step, set step.system, or pass system="
        )

    def held(key, obs, carry, params):
        if isinstance(carry, HoldCarry) and carry.remaining > 0:
            return carry.action, carry._replace(remaining=carry.remaining - 1), params
        node = _durations_node(system, params)
        for action_node, kernel in system._actuator(params, "hold").items():
            if _is_identity(kernel):
                continue  # an identity kernel changes nothing
            raise ValueError(
                f"hold: action node {action_node!r} has a non-identity actuator in "
                "params.actuator; a held mind is refused under one, since the driver realises "
                "every action node every round, so a held node's draw would be repeated each "
                "round of the option and the duration is read off the realised level "
                "(step-durations.md T9)"
            )
        inner = carry.inner if isinstance(carry, HoldCarry) else carry
        action, inner, learned = step(key, obs, inner, params)
        unit = float(system.elapsed(action, params))
        value = unit * ticks_per_unit
        rounds = round(value)
        if value < 1 - 1e-5:
            raise ValueError(
                f"hold: the duration of control node {node!r} at the chosen level is {unit:g}, "
                f"which times ticks_per_unit={ticks_per_unit} is shorter than one round"
            )
        if abs(value - rounds) > 1e-5 * max(1.0, abs(value)):
            raise ValueError(
                f"hold: the duration of control node {node!r} at the chosen level is "
                f"{unit:g}, which times ticks_per_unit={ticks_per_unit} is "
                "not a whole number of rounds"
            )
        return action, HoldCarry(inner, rounds - 1, action), learned

    held.system = system
    return held


class ParticleCarry(NamedTuple):
    """A particle's carry: the carry of each step inside it, the control levels in force, and the
    previous round's composite action.

    ``standing`` maps a control node a step chose to the level it chose last (the level the steps
    that read it are handed); ``last`` is the composite action of the round before, which a read
    of an action node takes as its datum. The parameter home is not in the carry: it is the params
    the runner hands the particle each round, and the record's ``params`` after the last.

    Design: nesting-and-coupling.md N4.
    """

    minds: Mapping[str, object] = {}
    standing: Mapping[str, int] = {}
    last: Mapping[str, Array] | None = None


class _ParticleStep:
    """The step of :func:`particle`: a callable that also says which of its composite's
    modalities it supplies itself (``supplied``), so :func:`couple` asks for no wire to them."""

    def __init__(self, step, supplied: frozenset[str]):
        self._step = step
        self.supplied = supplied

    def __call__(self, key, obs, carry, params):
        return self._step(key, obs, carry, params)


class _Body:
    """The per-round body both forms of a merged composite share: what each step is handed and
    what is done with what it returns.

    Built once for a composite, its params (which type it), and a step per component. A turn
    (:class:`_Round`) supplies the step's reads, hands it the view of its component taken from
    the parameter home at the levels the other components last chose, writes returned params back
    to the home, and checks the action. ``in_order`` says the steps take turns in the order of
    ``minds`` every round (``particle``, and ``run`` under ``"sequential"``), so a chooser listed
    first has chosen before a reader that needs its level responds.
    """

    def __init__(
        self,
        composite: System,
        params: SystemParams,
        minds: Mapping[str, Step],
        who: str,
        in_order: bool = True,
        tick_order: str = "sequential",
    ):
        self.composite = composite
        self.minds = minds
        self.layout = _layout(composite, minds, who)
        layout = self.layout
        face = who.removesuffix("()")
        typed = composite._names(composite._params_in(params, face))
        self.node_types = {
            a: (typed.kind(k), typed.sizes[k]) for k, a in enumerate(layout.names.actions)
        }
        order = list(minds)
        nodes = composite._table.control_names
        self.partner = {
            name: tuple(nodes[k] for ks in _inputs_of(composite, name).values() for k in ks)
            for name in order
        }
        kernels = composite._actuator(composite._params_in(params, face), who)
        for name in order:
            for node in self.partner[name]:
                chooser = layout.claimed[node]
                kernel = kernels.get(layout.action_of[node])
                if kernel is not None and not _is_identity(kernel):
                    raise ValueError(
                        f"{who}: the view of component {name!r} reads the control node {node!r}, "
                        f"and its action node {layout.action_of[node]!r} has a non-identity "
                        "kernel in params.actuator; the level handed on is the level the chooser "
                        "intended, which differs from the realised one under such a kernel, so "
                        "use an identity kernel there or do not read the node"
                    )
                if in_order and order.index(chooser) < order.index(name):
                    continue
                try:
                    subsystem(composite, params, name)
                except ValueError as err:
                    if in_order:
                        fix = (
                            f"list {chooser!r} before {name!r} in minds, so its choice is in hand "
                            f"before {name!r} responds"
                        )
                    else:
                        fix = (
                            f"tick_order={tick_order!r} does not guarantee that {chooser!r} "
                            f"has chosen before {name!r} responds; use tick_order='sequential' "
                            f"with {chooser!r} listed first, or fix the node with a "
                            "Clamp, or make the targets not conditional on it"
                        )
                    raise ValueError(
                        f"{who}: component {name!r} reads the control node {node!r}, which "
                        f"component {chooser!r} chooses, and has no view before that choice is "
                        f"made ({err}); {fix}"
                    ) from err
        self.supplied = frozenset(m for reads in layout.reads_in.values() for m, *_ in reads)
        #  (name, levels) -> (home, view System, view params): built once per set of levels and
        #  refreshed from the home when it changes, so equal Systems share one compiled program
        self._cache: dict[tuple, tuple] = {}

    def view(self, name: str, home: SystemParams, standing: Mapping[str, int]):
        """``name``'s view System and params taken from ``home``, with each control node another
        component chooses set to the level in ``standing`` (uniform before any choice)."""
        levels = tuple((n, standing.get(n)) for n in self.partner[name])
        entry = self._cache.get((name, levels))
        if entry is None:
            action = self.layout.action_of
            priors = {action[n]: lv for n, lv in levels if lv is not None}
            system, vparams = subsystem(self.composite, home, name, inputs=priors or None)
        elif entry[0] is home:
            return entry[1], entry[2]
        else:
            system = entry[1]
            vparams = retake(self.composite, system, entry[2], home)
        self._cache[(name, levels)] = (home, system, vparams)
        return system, vparams


def _is_identity(kernel) -> bool:
    """Whether an action kernel is the identity table (square, equal to the identity)."""
    table = np.asarray(kernel)
    return (
        table.ndim == 2
        and table.shape[0] == table.shape[1]
        and bool(np.allclose(table, np.eye(table.shape[0])))
    )


class _Round:
    """One round of a :class:`_Body`: the carries, the parameter home and the control levels in
    force, mutated turn by turn. ``live`` and ``standing`` are updated in place; ``home`` is
    rebound when a step learns."""

    def __init__(
        self,
        body: _Body,
        live: dict[str, object],
        home: SystemParams,
        standing: dict[str, int],
        last: Mapping[str, Array] | None,
        frozen: bool = False,
    ):
        self.body, self.live, self.home, self.standing = body, live, home, standing
        #  the levels the views are taken at: those standing when the round began if ``frozen``
        #  (every step reads the round's boundary), else as they stand at each turn
        self.levels = dict(standing) if frozen else standing
        self.last = last
        #  the emissions the sources carried when the round began (a read is one tick behind)
        self.begun = {
            holder: _emitted(live[holder])
            for reads in body.layout.reads_in.values()
            for _, holder, _ in reads
            if holder is not None
        }

    def turn(self, name: str, key: Array, obs: Mapping[str, Array]) -> dict[str, Array]:
        """``name``'s step called on ``obs`` (its own modalities picked out, its reads supplied)
        and its view; returns its checked action."""
        body, layout = self.body, self.body.layout
        mods, ports = layout.routes[name]
        system, given = body.view(name, self.home, self.levels)
        #  a modality the view feeds from its own variable is keyed by the variable, and the
        #  world's sample of it is not the datum
        renamed = dict(system.observes)
        seen = {m: obs[m] for m in mods if m in obs and m not in renamed}
        for m, datum in _supplied(layout.reads_in[name], self.begun, self.last).items():
            seen[renamed.get(m, m)] = datum
        chosen, self.live[name], learned = body.minds[name](key, seen, self.live[name], given)
        if learned is not given:
            _check_parameters_only(system, name, given, learned)
            self.home = write_back(body.composite, self.home, name, given, learned)
        action = _check_action(
            chosen, ports, name, layout.action_of, {a: body.node_types[a] for a in ports}
        )
        for node, owner in layout.claimed.items():
            if owner == name and body.node_types[layout.action_of[node]][0] == LEVEL:
                self.standing[node] = int(action[layout.action_of[node]])
        return action


def particle(
    composite: System,
    params: SystemParams,
    minds: Mapping[str, Step],
) -> Step:
    """A merged composite with one step per view, as one step for :func:`couple`.

    The composite is one particle: its modalities are the participant's in-ports and its action
    nodes the out-ports (``couple((composite, params, particle(...)))``, a world joined by
    wires). Inside it each round runs the body of :func:`run`'s composite form, in the order of
    ``minds`` (a step listed later sees an earlier step's decision of this round; the order is the
    particle's own, not a tick order):

    - every read of a minded component is supplied from the carries: the emission its observed
      variable names, as the holder's step realised it the round before, an action node's level
      of the round before, ``None`` where nothing is carried yet (the modality is absent that
      round);
    - each step is handed the view of its component taken from the home (the composite's params
      as the runner holds them, which are the ones handed to the particle), with each control
      node another component chooses that its view reads set to the level that component last
      chose (uniform before any choice): the per-level views are built once on first use
      (``subsystem``) and refreshed from the home (``retake``) when it changes, equal Systems and
      the shapes of one compiled program;
    - when a step returns params other than the ones it was given, they are written back to the
      home (:func:`~hybrid_aif.composition.merge.write_back`, with the view it was given as
      ``before``, so the stale-``before`` rule holds by construction); the next step to respond
      is handed a view taken from the new home, so an ``Identify`` needs no bookkeeping.
      The params returned may differ from the ones given in parameter variables only, else this
      raises. The particle returns the home, so the record of the coupling holds the learned
      composite params.

    A step's key is the particle's for the first step in ``minds`` and ``fold_in(key, i)`` for the
    ``i``-th after it. Every component needs a step (a component that chooses nothing takes
    :func:`ancestral_step` or a step returning no action), and every control node must be chosen
    by a listed step. A view that reads a control node another component chooses, and whose
    targets are conditional on it, has no view before that component has chosen: list the chooser
    first, which this checks at construction. A step wrapped in :func:`hold` composes as in the
    composite form. ``params`` types the composite and is used to check the order; each step's
    carry is ``carry.minds[name]``.

    Design: nesting-and-coupling.md N4, N9.
    """
    body = _Body(composite, params, minds, "particle()")
    order = list(minds)
    unminded = sorted(set(body.layout.table) - set(minds))
    if unminded:
        raise ValueError(
            f"particle(): component(s) {unminded} have no step in minds; every component of a "
            "particle needs one (ancestral_step(...) or a step returning no action for a "
            "component that chooses nothing)"
        )

    def step(key, obs, carry, home):
        if not isinstance(carry, ParticleCarry):
            carry = ParticleCarry()
        live = {n: carry.minds.get(n, MindCarry()) for n in order}
        this = _Round(body, live, home, dict(carry.standing), carry.last)
        action: dict[str, Array] = {}
        for i, name in enumerate(order):
            mind_key = key if i == 0 else jax.random.fold_in(key, i)
            action.update(this.turn(name, mind_key, obs))
        return action, ParticleCarry(live, this.standing, action), this.home

    return _ParticleStep(step, body.supplied)


def _own_fields(system: System, view_params: SystemParams) -> dict[str, object]:
    """The fields of positional view params that no parameter variable fills, by name (a
    transition by its factor, a likelihood by its modality): everything but ``parameters``,
    ``references`` and the reader positions the references refill."""
    refilled = {(slot, i) for slot, i, _ in view_params.references}
    named = {"transition": system._table.state_names, "likelihood": system.modality_names}
    out: dict[str, object] = {}
    for field in dataclasses.fields(view_params):
        name = field.name
        value = getattr(view_params, name)
        if name in ("parameters", "references"):
            continue
        if name in named:
            for i, entry in enumerate(value):
                if (name, i) not in refilled:
                    out[f"{name}[{named[name][i]!r}]"] = entry
        else:
            out[name] = value
    return out


def _check_parameters_only(
    system: System, name: str, given: SystemParams, learned: SystemParams
) -> None:
    """Refuse learned params that differ from the given ones outside the parameter variables.
    Leaves are compared by identity first and by value only where the identity changed."""
    before = _own_fields(system, system._params_in(given, "run"))
    after = _own_fields(system, system._params_in(learned, "run"))
    differ = [] if before.keys() == after.keys() else ["the set of fields"]
    for field in before.keys() & after.keys():
        la, ta = jax.tree_util.tree_flatten(before[field])
        lb, tb = jax.tree_util.tree_flatten(after[field])
        if ta != tb or len(la) != len(lb):
            differ.append(field)
            continue
        for x, y in zip(la, lb, strict=True):
            if x is y:
                continue
            try:
                same = bool(jnp.array_equal(x, y, equal_nan=True))
            except TypeError:
                same = x == y
            if not same:
                differ.append(field)
                break
    if differ:
        raise ValueError(
            f"the step of component {name!r} returned params that differ from the ones it was "
            f"given in {sorted(differ)}; on a composite only declared parameter variables are "
            "written back to the home, so a table that is learned must be declared as a "
            "Parameter variable"
        )


def _durations_node(system: System, params: SystemParams) -> str:
    """The control node that carries ``params.durations``, or a refusal when none is declared."""
    declared = system._durations(system._params_in(params, "hold"), "hold")
    if not declared:
        raise ValueError(
            "hold: this System declares no params.durations, so there is no option to hold; "
            "a step with no durations is a plain step"
        )
    ((node, _),) = declared.items()
    return node


def _emitted(carry) -> Mapping[str, Array] | None:
    """The emissions a mind realised in its last round, by name: a :class:`MindCarry`'s, or the
    wrapped mind's under :func:`hold`; ``None`` before the mind has any (or for a mind whose
    carry keeps none)."""
    while carry is not None and not hasattr(carry, "emitted"):
        carry = getattr(carry, "inner", None)
    return None if carry is None else carry.emitted


def _driven_reads(
    composite: System,
    minds: Mapping[str, Step],
    routes: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]],
    emitters: Mapping[str, tuple[str, ...]],
) -> dict[str, list[tuple[str, str | None, str]]]:
    """Every read the driver supplies, by reading mind: ``(modality, holder mind, variable)``. The
    holder is the mind whose component declares the emission the modality observes (the reader
    itself for an emission of its own view), and ``None`` for an action node, whose datum is the
    previous round's chosen level. A step's view never fills its own reads (it takes no carry of
    past rounds), so the driver supplies all of them."""
    out: dict[str, list] = {name: [] for name in minds}
    actions = set(composite.names.actions)
    for modality, variable in composite.observes:
        reader = next((n for n in minds if modality in routes[n][0]), None)
        if reader is None:
            continue
        holder = None
        if variable not in actions:
            holder = next((n for n in minds if variable in emitters[n]), None)
            if holder is None:
                raise ValueError(
                    f"modality {modality!r} observes {variable!r}, an emission of a component "
                    "that has no mind, so there is no carried emission to read; give that "
                    "component a step"
                )
        out[reader].append((modality, holder, variable))
    return out


class _Layout(NamedTuple):
    """What the composite drivers share: the routing of a merged composite's minds."""

    table: dict
    names: object
    action_of: dict[str, str]
    routes: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]
    claimed: dict[str, str]
    factors_of: dict[str, tuple[str, ...]]
    unminded: tuple[str, ...]
    reads_in: dict[str, list]


def _layout(composite: System, minds: Mapping[str, object], who: str) -> _Layout:
    """The routing of ``minds`` over a merged composite: each mind's modalities and action nodes
    by composite name, the control node each mind chooses, the factors that step at its turn,
    and the reads the driver supplies. Refuses a composite with no ``subsystems`` metadata, a name
    that is no component, and a control node no mind chooses."""
    if composite.subsystems is None:
        raise ValueError(
            f"{who} routes by the subsystems metadata, and this System carries none; build the "
            "composite with hybrid_aif.composition.merge.merge()"
        )
    table = dict(composite.subsystems)
    unknown = sorted(set(minds) - set(table))
    if unknown:
        raise ValueError(
            f"minds names {unknown}, which are not components of this composite; its "
            f"components are {sorted(table)}"
        )
    c_names = composite._names()
    c_mods, c_facs = c_names.modalities, c_names.factors
    action_of = dict(zip(c_names.controls, c_names.actions, strict=True))
    routes: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    claimed: dict[str, str] = {}
    for name in minds:
        chosen = [node for node in table[name].controls if node in action_of]
        # the mind's modalities and its action nodes, by composite name
        routes[name] = (
            tuple(c_mods[m] for m in table[name].modalities),
            tuple(action_of[node] for node in chosen),
        )
        for node in chosen:
            claimed[node] = name
    unclaimed = [node for node in c_names.controls if node not in claimed]
    if unclaimed:
        raise ValueError(
            f"control node(s) {unclaimed} are chosen by no mind, so nothing would ever emit "
            "their action; give the owning component a step (ancestral_step(...) for an "
            "env-like one), or fix the node with a Clamp naming it at merge time"
        )
    #  which composite factors step at which component's turn (sequential, async); the factors
    #  of the components with no mind step at the end of the tick
    factors_of = {name: tuple(c_facs[i] for i in table[name].factors) for name in minds}
    minded = {f for fs in factors_of.values() for f in fs}
    unminded = tuple(f for f in c_facs if f not in minded)
    return _Layout(
        table,
        c_names,
        action_of,
        routes,
        claimed,
        factors_of,
        unminded,
        _driven_reads(composite, minds, routes, {n: table[n].emissions for n in minds}),
    )


def _supplied(
    reads: list,
    begun: Mapping[str, Mapping[str, Array] | None],
    last: Mapping[str, Array] | None,
) -> dict[str, Array | None]:
    """The data of one mind's reads this round, by composite modality: the emission its variable
    names as the holder's carry held it when the round began (``begun``, by holder mind), an action
    node's level of the round before (``last``); ``None`` where the source has nothing yet."""
    out: dict[str, Array | None] = {}
    for modality, holder, variable in reads:
        if holder is None:
            out[modality] = None if last is None else last[variable]
            continue
        carried = begun[holder]
        out[modality] = None if carried is None else carried.get(variable)
    return out


def run(
    composite: "System | Wiring",
    params: SystemParams | None = None,
    minds: Mapping[str, Step] | None = None,
    ticks: int | None = None,
    key: Array | None = None,
    tick_order: TickOrder = "simultaneous",
    state: Mapping[str, Array] | None = None,
    carries: Mapping[str, MindCarry] | None = None,
    param_key: Array | None = None,
    *,
    states: Mapping[str, Mapping[str, Array]] | None = None,
):
    """Run steps and processes for a number of ticks; see the module docstring.

    **Wired form**, ``run(wiring, ticks=..., key=..., tick_order="simultaneous", carries=None,
    states=None) -> dict[str, MindRecord | ProcessRecord]``: ``wiring`` is
    :func:`~hybrid_aif.composition.couple.couple`'s result, and ``ticks`` and ``key`` are given
    by keyword; ``carries`` seeds the minds' carries and ``states`` the processes' resets (a
    point state keyed by factor name), each keyed by participant name. Each mind's params start
    as the ones its entry in :func:`couple` declares; every round its step is handed the current
    ones and what it returns is kept (:attr:`MindRecord.params` holds the last; a step that
    returns the object it was given did not learn). Every process steps once
    per tick, at the end, on the acts of the tick. Under ``"simultaneous"`` every mind responds
    to the tick's boundary; under ``"sequential"`` and ``"async"`` a mind's action nodes are
    realised as it chooses, so a later mind sees an earlier mind's act this tick through a
    direct wire. A mind's observation row ``t`` is what it responded to at tick ``t``, and row
    ``ticks`` its in-ports after the last tick; a process's rows are the tick boundaries (row 0
    the reset, row ``t + 1`` the end of tick ``t``). Under ``"sequential"`` and ``"async"`` a
    later mind's row ``t`` therefore already holds the acts earlier minds made at tick ``t`` on a
    direct wire. Before a mind has chosen, its action nodes intend level 0 (zeros, for a
    continuous node), undrawn, and every wire and process reached from one carries that null
    value as given. A node is realised when its mind chooses, and the processes step on the
    realised values standing at the end of the tick, once.

    Keys: the run key splits into ``(key, k_reset)``; process ``p`` (in :func:`couple` order)
    resets with ``k_reset`` for ``p = 0`` and ``fold_in(k_reset, p)`` after. Each tick
    ``(key, k_tick) = split(key)`` and ``k_tick`` splits into ``2 M + 2`` keys for ``M`` minds:
    mind ``j`` responds with key ``j``; its action nodes are realised with key ``M + j`` under
    sequential and async, and every mind's with the last key under simultaneous, which is also
    the tick's step key under every order. Realising with key ``k`` draws node ``i`` of the ``N``
    (minds in :func:`couple` order, each mind's action nodes in its node order) from its
    kernel's column at the intended level with key ``i`` of ``split(fold_in(k, ACTUATOR_TAG),
    N)``; the step with key ``k`` steps process ``p`` with ``k`` for ``p = 0`` and
    ``fold_in(k, p)`` after, on its realised inputs. A wiring of one process, a merged
    composite's ``as_env``, with the minds on its ``subsystem`` views and the wires by name,
    reproduces the composite form bitwise under ``"simultaneous"`` when the minds are given in
    the composite's component order, so that the action nodes, and the kernels keyed by them,
    fall in the composite's action order, and no component's view reads a control node another
    component chooses (the composite form routes such a choice into the reading view, which a
    view fixed by the caller does not); under the other orders the composite form steps each
    component's factors at its turn, which a wiring of one process cannot, and the two differ.
    The action kernels of a mind (``params.actuator``) are those of its starting params, fixed
    when :func:`~hybrid_aif.composition.couple.couple` runs, so a step that learns a kernel
    changes nothing the wired form realises; a step that returns anything other than a
    ``SystemParams`` (when it does not return the params it was given) is refused naming the mind.

    **Composite form**, ``run(composite, params, minds, ticks, key, tick_order="simultaneous",
    state=None, carries=None, param_key=None) -> Trajectory``: the composite's world is its own
    model (``as_env``), run forward from ``params``. ``minds`` maps a subsystem name to its
    :data:`Step`, in the order ``"sequential"`` follows. ``params`` is the parameter home, and
    each round a step is handed the view of its component taken from the home as it stands
    (:func:`~hybrid_aif.composition.merge.subsystem`, with each control node another component
    chooses that the view reads at the level that component last intended, uniform before any
    choice; built once per set of levels and refreshed with ``retake``). Under ``"simultaneous"``
    the levels are those standing when the round began, since every step reads the round's
    boundary; under ``"sequential"`` and ``"async"`` they are as they stand at the step's turn.
    A non-identity ``params.actuator`` kernel on a node another component's view reads is refused
    at construction, since the level handed on is the intended one, not the realised one. A step
    that returns params
    other than the ones it was given has them written back to the home
    (:func:`~hybrid_aif.composition.merge.write_back`) and the next view is taken from the new
    home; it may differ from what it was given in declared parameter variables only, else this
    raises. The world does not learn: it keeps running the starting ``params``. A view that reads a
    control node whose targets are conditional on it has no view before that node's chooser has
    chosen, so it is supported only where the chooser has chosen before the reader responds in
    every round: under ``"sequential"`` with the chooser listed first (checked up front); under
    ``"simultaneous"`` and ``"async"`` it is refused up front. ``Trajectory.params`` is the
    home after the last round. :func:`particle` is the same per-round body as one step of a
    coupling. Every control node of the composite must be chosen by a named step, or this raises
    (give the component a step, or fix the node with a
    :class:`~hybrid_aif.composition.merge.Clamp`); a component that chooses no control
    node needs none. ``state`` overrides the env's initial draw with a known point state keyed
    by composite factor name, ``carries`` seeds the minds' carries, and ``param_key`` draws
    ground-truth tables from learnable declarations, as in :meth:`System.as_env`. Under
    ``"sequential"`` and ``"async"`` the composite steps a component's factors at that
    component's turn, on the realised actions standing then (every other mind's last realised
    action, level 0 before a mind has chosen), and the factors of the components that have no
    step at the end of the tick; under ``"simultaneous"`` the action is re-zeroed each tick and
    the whole composite steps once on it. Every partial step emits the composite's observations
    afresh, so a later mind reads a stepped factor's new value and a new draw of any noisy
    modality, including one over factors that did not step. The composite's kernels
    (``params.actuator``) are drawn at the nodes with the wired form's keys over the composite's
    action order, and the world is stepped on the realised action with an ``env_params`` whose
    actuator mapping is empty, so the world draws nothing.

    Every read of a minded component (``System.observes`` of the composite: a component's own
    emission or action node, a belief :class:`~hybrid_aif.composition.merge.Link`, a read inside
    an inner composite that is one mind) is supplied to its mind each round, as an ordinary
    observation, in place of the world's sample of that modality. A read of an emission takes the
    value the holder's step realised from its posterior the round before (``MindCarry.emitted``,
    as it stood when the round began, so a read is one tick behind whatever the order); a read of
    an action node takes the level chosen the round before. On the first round, or when the source
    carries no emission, the entry is ``None`` (the modality is absent; :func:`agent_step` filters
    on it); a holder mind must exist. Design: nesting-and-coupling.md N3.
    """
    if isinstance(composite, Wiring):
        if any(v is not None for v in (params, minds, state, param_key)):
            raise TypeError(
                "run(wiring, ticks=..., key=...) takes ticks and key by keyword and nothing "
                "else by position, and its minds, their params and the processes come from "
                "couple(), so pass no params, minds, state or param_key"
            )
        if ticks is None or key is None:
            raise TypeError("run(wiring, ticks=..., key=...) needs both ticks and key")
        return _run_wired(composite, ticks, key, tick_order, carries, states)
    if states is not None:
        raise TypeError(
            "run(composite, ...) takes states= only with a Wiring; seed the composite's reset "
            "with state= keyed by composite factor name"
        )
    missing = [
        n
        for n, v in (("params", params), ("minds", minds), ("ticks", ticks), ("key", key))
        if v is None
    ]
    if missing:
        raise TypeError(f"run(composite, params, minds, ticks, key) is missing {missing}")
    return _run_composite(
        composite, params, minds, ticks, key, tick_order, state, carries, param_key
    )


def _run_composite(
    composite: System,
    params: SystemParams,
    minds: Mapping[str, Step],
    ticks: int,
    key: Array,
    tick_order: TickOrder = "simultaneous",
    state: Mapping[str, Array] | None = None,
    carries: Mapping[str, MindCarry] | None = None,
    param_key: Array | None = None,
) -> Trajectory:
    """Run ``minds`` against the composite for ``ticks``: :func:`run`'s composite form, whose
    docstring describes it. The per-round body is :class:`_Body`'s, shared with :func:`particle`.

    ``minds`` is keyed by subsystem name, in the order ``"sequential"`` follows. Every control
    node of the composite must be chosen by a named step, or this raises (give the component a
    step, or fix the node with a :class:`~hybrid_aif.composition.merge.Clamp`); a
    component that chooses no control node needs none. ``state`` overrides the env's initial draw
    with a known point state keyed by composite factor name, ``carries`` seeds the minds'
    carries, and ``param_key`` draws ground-truth tables from learnable declarations, as in
    :meth:`System.as_env`. Under ``"sequential"`` and ``"async"`` a component's factors step at
    its turn on the realised actions standing then (every other mind's last realised action,
    level 0 before a mind has chosen), and a component with no mind steps at the end of the
    tick, each partial step emitting the observations afresh; under ``"simultaneous"`` the
    action is re-zeroed each tick and the composite steps once on it.
    """
    if tick_order not in ("simultaneous", "sequential", "async"):
        raise ValueError(
            f"tick_order must be 'simultaneous', 'sequential', or 'async', got {tick_order!r}. "
            "A round is one tick under every order and every factor steps once per round. "
            "simultaneous = every mind reads the round's boundary, then the composite steps once "
            "on the merged actions (synchronous field update; simultaneous-move games); "
            "sequential = minds respond in declaration order and each component's factors step "
            "at its turn, reading values fresh where the source has stepped this round (fixed "
            "turn-taking); async = as sequential in a RANDOM permutation per round (no ordering "
            "privilege; the statistically faithful mode for equilibrium/Ising-like composites, "
            "where synchronous updates do not preserve the stationary distribution). The choice "
            "changes the game structure (design: nesting-systems.md, requirement 5 + "
            "tick-semantics note)."
        )
    body = _Body(composite, params, minds, "run()", tick_order == "sequential", tick_order)
    layout = body.layout
    c_names = layout.names
    c_mods, c_facs = c_names.modalities, c_names.factors
    factors_of, unminded = layout.factors_of, layout.unminded

    env, env_params = composite.as_env(params, param_key)
    key, k_reset = jax.random.split(key)
    obs, world = env.reset(k_reset, env_params, state)
    live = {name: MindCarry() for name in minds}
    live.update(carries or {})
    home, standing = params, {}

    node_types = body.node_types

    def no_op() -> dict[str, Array]:
        return {
            k: jnp.zeros((), jnp.int32)
            if node_types[k][0] == LEVEL
            else jnp.zeros((node_types[k][1],), jnp.float32)
            for k in c_names.actions
        }

    action = no_op()
    realised = no_op()
    #  the actuator draws at the node; the world steps on the realised action under params whose
    #  actuator is empty, so the world draws nothing
    kernels = dict(env_params.actuator or {})
    quiet = dataclasses.replace(env_params, actuator={})
    order = {a: i for i, a in enumerate(c_names.actions)}
    states_log, obs_log, action_log, realised_log = [world], [obs], [], []

    def actuate(k: Array, nodes: tuple[str, ...]) -> dict[str, Array]:
        """``nodes`` realised from the intended ``action`` at their composite indices."""
        drawn = _actuate(
            k,
            len(c_names.actions),
            [(order[a], kernels[a], action[a]) for a in nodes if a in kernels],
        )
        return {a: drawn.get(order[a], action[a]) for a in nodes}

    names = list(minds.keys())
    # tick_order="async": round orders from a host-side generator seeded by the run key
    # (folded, so the per-tick key stream is untouched).
    sweep_seed = jax.random.key_data(jax.random.fold_in(key, len(names)))
    sweep_rng = np.random.default_rng(np.asarray(sweep_seed, np.uint32).ravel())
    for _ in range(ticks):
        key, k_tick = jax.random.split(key)
        keys = list(jax.random.split(k_tick, 2 * len(minds) + 2))
        if tick_order == "simultaneous":
            action = no_op()
        #  the previous round's chosen actions: a read of an action node takes them as its datum
        this = _Round(
            body,
            live,
            home,
            standing,
            action_log[-1] if action_log else None,
            frozen=tick_order == "simultaneous",
        )
        if tick_order == "async":
            # a fresh random permutation per round: no fixed ordering privilege
            perm = [int(i) for i in sweep_rng.permutation(len(names))]
        else:
            perm = list(range(len(names)))
        for j in perm:
            name = names[j]
            ports = layout.routes[name][1]
            action = {**action, **this.turn(name, keys[j], obs)}
            if tick_order in ("sequential", "async"):
                k_step = keys[len(minds) + j]
                realised = {**realised, **actuate(k_step, ports)}
                if factors_of[name]:
                    obs, world = env.step(k_step, world, realised, quiet, factors_of[name])
        home = this.home
        if tick_order == "simultaneous":
            realised = actuate(keys[-1], c_names.actions)
            obs, world = env.step(keys[-1], world, realised, quiet)
        elif unminded:
            obs, world = env.step(keys[-1], world, realised, quiet, unminded)
        states_log.append(world)
        obs_log.append(obs)
        action_log.append(action)
        realised_log.append(realised)

    return Trajectory(
        states={f: jnp.stack([s[f] for s in states_log]) for f in c_facs},
        observations={m: jnp.stack([o[m] for o in obs_log]) for m in c_mods},
        actions={
            k: jnp.stack([a[k] for a in action_log]) if action_log else jnp.zeros((0,), jnp.int32)
            for k in c_names.actions
        },
        carries=live,
        params=home,
        realised={
            k: jnp.stack([a[k] for a in realised_log])
            if realised_log
            else jnp.zeros((0,), jnp.int32)
            for k in c_names.actions
        },
    )


def _run_wired(
    wiring: Wiring,
    ticks: int,
    key: Array,
    tick_order: TickOrder = "simultaneous",
    carries: Mapping[str, MindCarry] | None = None,
    states: Mapping[str, Mapping[str, Array]] | None = None,
) -> dict[str, "MindRecord | ProcessRecord"]:
    """:func:`run`'s wired form."""
    if tick_order not in ("simultaneous", "sequential", "async"):
        raise ValueError(
            f"tick_order must be 'simultaneous', 'sequential', or 'async', got {tick_order!r}; "
            "every process steps once per tick, at the end; simultaneous = every mind responds "
            "to the tick's boundary; sequential = minds respond in couple() order, a later one "
            "seeing an earlier one's act on a direct wire; async = as sequential in a random "
            "order per tick"
        )
    mind_names, proc_names = list(wiring.minds), list(wiring.processes)
    for given, label, table in (
        (carries, "carries", mind_names),
        (states, "states", proc_names),
    ):
        unknown = sorted(set(given or {}) - set(table))
        if unknown:
            kind = "minds" if label == "carries" else "processes"
            raise ValueError(
                f"run(): {label} names {unknown}, which are not {kind} of this wiring; its "
                f"{kind} are {table}"
            )

    m_count = len(mind_names)
    #  the action nodes, in the fixed order the actuator keys follow
    nodes = [(n, a) for n in mind_names for a in wiring.minds[n].system.names.actions]
    port_types = {
        n: {a: wiring.types[f"{n}.{a}"] for a in wiring.minds[n].system.names.actions}
        for n in mind_names
    }
    action_of = {
        n: dict(
            zip(
                wiring.minds[n].system.names.controls,
                wiring.minds[n].system.names.actions,
                strict=True,
            )
        )
        for n in mind_names
    }
    intended = {n: {a: wiring.null(f"{n}.{a}") for a in port_types[n]} for n in mind_names}
    realised = {n: dict(intended[n]) for n in mind_names}

    def process_key(k: Array, p: int) -> Array:
        return k if p == 0 else jax.random.fold_in(k, p)

    key, k_reset = jax.random.split(key)
    obs: dict[str, dict[str, Array]] = {}
    world: dict[str, dict[str, Array]] = {}
    for p, n in enumerate(proc_names):
        proc = wiring.processes[n]
        obs[n], world[n] = proc.env.reset(
            process_key(k_reset, p), proc.params, (states or {}).get(n)
        )

    def value(port: str) -> Array:
        who, name = port.split(".", 1)
        if who in realised:
            return realised[who][name]
        return obs[who][name]

    wired = {
        n: tuple(m for m in wiring.minds[n].system.modality_names if f"{n}.{m}" in wiring.feeds)
        for n in mind_names
    }

    def mind_inputs(n: str) -> dict[str, Array]:
        return {m: value(wiring.feeds[f"{n}.{m}"]) for m in wired[n]}

    def process_inputs(n: str) -> dict[str, Array]:
        out = {}
        for a in wiring.processes[n].env.names.actions:
            port = f"{n}.{a}"
            source = wiring.feeds[port]
            out[a] = wiring.fixed[port] if source is None else value(source)
        return out

    index = {node: i for i, node in enumerate(nodes)}

    def realise(k: Array, chosen: list[str]) -> None:
        """The action nodes of the ``chosen`` minds realised from their kernels."""
        draws = [
            (index[(n, a)], wiring.minds[n].kernels[a], intended[n][a])
            for n in chosen
            for a in port_types[n]
            if a in wiring.minds[n].kernels
        ]
        drawn = _actuate(k, len(nodes), draws)
        for n in chosen:
            realised[n] = {a: drawn.get(index[(n, a)], intended[n][a]) for a in port_types[n]}

    def step_processes(k: Array) -> None:
        """Every process stepped once on its realised inputs."""
        for p, n in enumerate(proc_names):
            proc = wiring.processes[n]
            obs[n], world[n] = proc.env.step(
                process_key(k, p), world[n], process_inputs(n), proc.params
            )

    live = {n: MindCarry() for n in mind_names}
    live.update(carries or {})
    current = {n: wiring.minds[n].params for n in mind_names}
    received_log: dict[str, list] = {n: [] for n in mind_names}
    action_log: dict[str, list] = {n: [] for n in mind_names}
    realised_log: dict[str, list] = {n: [] for n in mind_names}
    states_log = {n: [world[n]] for n in proc_names}
    obs_log = {n: [obs[n]] for n in proc_names}

    sweep_seed = jax.random.key_data(jax.random.fold_in(key, m_count))
    sweep_rng = np.random.default_rng(np.asarray(sweep_seed, np.uint32).ravel())
    for _ in range(ticks):
        key, k_tick = jax.random.split(key)
        keys = list(jax.random.split(k_tick, 2 * m_count + 2))
        if tick_order == "async":
            perm = [int(i) for i in sweep_rng.permutation(m_count)]
        else:
            perm = list(range(m_count))
        for j in perm:
            n = mind_names[j]
            received = mind_inputs(n)
            received_log[n].append(received)
            chosen, live[n], learned = wiring.minds[n].step(keys[j], received, live[n], current[n])
            if learned is not current[n]:
                if not isinstance(learned, SystemParams):
                    raise ValueError(
                        f"run(): the step of mind {n!r} returned {type(learned).__name__} as its "
                        "params; a step returns the SystemParams it was given, or learned ones"
                    )
                current[n] = learned
            intended[n] = _check_action(
                chosen, tuple(port_types[n]), n, action_of[n], port_types[n]
            )
            if tick_order in ("sequential", "async"):
                realise(keys[m_count + j], [n])
        if tick_order == "simultaneous":
            realise(keys[-1], mind_names)
        step_processes(keys[-1])
        for n in mind_names:
            action_log[n].append(dict(intended[n]))
            realised_log[n].append(dict(realised[n]))
        for n in proc_names:
            states_log[n].append(world[n])
            obs_log[n].append(obs[n])
    for n in mind_names:
        received_log[n].append(mind_inputs(n))

    def stacked(rows: list, keys_: tuple[str, ...], empty) -> dict[str, Array]:
        return {k: jnp.stack([r[k] for r in rows]) if rows else empty(k) for k in keys_}

    out: dict[str, MindRecord | ProcessRecord] = {}
    for n in mind_names:
        actions = tuple(port_types[n])

        def empty(a: str, n: str = n) -> Array:
            null = wiring.null(f"{n}.{a}")
            return jnp.zeros((0, *null.shape), null.dtype)

        out[n] = MindRecord(
            observations=stacked(received_log[n], wired[n], None),
            actions=stacked(action_log[n], actions, empty),
            realised=stacked(realised_log[n], actions, empty),
            carry=live[n],
            params=current[n],
        )
    for n in proc_names:
        names = wiring.processes[n].env.names
        out[n] = ProcessRecord(
            states=stacked(states_log[n], names.factors, None),
            observations=stacked(obs_log[n], names.modalities, None),
        )
    return out


def _actuate(k: Array, n: int, draws: list[tuple[int, Array, Array]]) -> dict[int, Array]:
    """The actuator's draws at the node: for each ``(i, kernel, intended)``, a level drawn from
    the kernel's column at the intended level with key ``i`` of
    ``split(fold_in(k, ACTUATOR_TAG), n)``, keyed by ``i``. No draw, no key."""
    if not draws:
        return {}
    keys = jax.random.split(jax.random.fold_in(k, ACTUATOR_TAG), n)
    out = {}
    for i, kernel, level in draws:
        log_p = jnp.log(jnp.asarray(kernel, jnp.float32))
        column = jnp.moveaxis(jnp.take(log_p, jnp.asarray(level, jnp.int32), axis=-1), 0, -1)
        out[i] = jax.random.categorical(keys[i], column).astype(jnp.int32)
    return out


def _check_action(
    chosen: Mapping[str, Array],
    ports: tuple[str, ...],
    name: str,
    action_of: Mapping[str, str] | None = None,
    types: Mapping[str, tuple[str, int | None]] | None = None,
) -> dict[str, Array]:
    """The mind's action, checked: exactly its action nodes, each one integer level (shape
    ``()``), or with ``types`` naming a node continuous, a float vector of its width; a key
    naming a control node is refused naming its action node."""
    if not isinstance(chosen, Mapping):
        raise TypeError(
            f"mind {name!r} returned a positional action; a step returns its action keyed "
            f"by action node, {list(ports)} (convert with view.from_positional('actions', ...))"
        )
    crossed = [k for k in chosen if k in (action_of or {}) and action_of[k] in ports]
    if crossed:
        raise ValueError(
            f"mind {name!r} returned actions[{crossed[0]!r}], a control node; key the action by "
            f"its action node {action_of[crossed[0]]!r}"
        )
    missing = [k for k in ports if k not in chosen]
    unknown = [k for k in chosen if k not in ports]
    if missing or unknown:
        raise ValueError(
            f"mind {name!r} returned an action keyed {list(chosen)}, missing {missing} and "
            f"unknown {unknown}; its action nodes are {list(ports)}"
        )
    out = {}
    for k in ports:
        level = jnp.asarray(chosen[k])
        if types is not None and types[k][0] == FLOAT:
            width = types[k][1]
            if level.shape != (width,) or not jnp.issubdtype(level.dtype, jnp.floating):
                raise ValueError(
                    f"mind {name!r} returned actions[{k!r}] of dtype {level.dtype} and shape "
                    f"{tuple(level.shape)}; a continuous action node takes a float vector of "
                    f"its width, shape ({width},)"
                )
            out[k] = level
            continue
        if level.shape != ():
            raise ValueError(
                f"mind {name!r} returned actions[{k!r}] of shape {tuple(level.shape)}; an action "
                "node takes one level per tick, shape ()"
            )
        if not jnp.issubdtype(level.dtype, jnp.integer):
            raise ValueError(
                f"mind {name!r}: actions [{k!r}] are not integer ({level.dtype}); a discrete "
                "control's action is a level index"
            )
        if types is not None:
            size = types[k][1]
            if not 0 <= int(level) < size:
                raise ValueError(
                    f"mind {name!r} returned actions[{k!r}] = {int(level)}; action node {k!r} has "
                    f"{size} levels, so emit a level in [0, {size})"
                )
        out[k] = level.astype(jnp.int32)
    return out
