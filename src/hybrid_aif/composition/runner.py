"""``run`` over a closed System: every leaf stepped, one value per variable per round.

A **closed System** is a merge or a coupling in which every modality is fed by an identification
(``System.observes``): nothing is observed from outside. It is run by naming a step for each of
its **stepped nodes**, the leaves (components that are not themselves composed) or a non-leaf
node planned jointly as one mind (:func:`~hybrid_aif.composition.joint.joint_planner`); the keys
of ``steps`` form an antichain that covers every leaf. A step is the function
``step(system, key, obs, carry, params) -> (out, carry, params)``:

- ``system`` and ``params`` are its view (the leaf's :func:`~hybrid_aif.composition.merge.subsystem`
  view, or the joint view) and the view's params, taken from the parameter home each round;
- ``obs`` maps each of its modality slots and each control node a variable drives, by the slot's
  own name, to the value of the variable that slot observes (``None`` where the variable has no
  value yet); ``obs`` is ``None`` on the step's first call, its reset;
- ``out`` maps its out-ports to values: the action nodes of the nodes it chooses (each an
  intended level, or a float vector for a continuous node) and its emissions;
- the params returned are the very object handed in when the step did not learn; a learner's
  returned parameter variables are written back to the home
  (:func:`~hybrid_aif.composition.merge.write_back`).

:func:`plan` builds the step of a planning mind, :func:`sample` the step of a process (a world
sampled forward from its declarations); a custom step is any function of that shape.

**Variables.** A System's variables that a step generates are its action nodes and its emissions;
each has one value per round. An action node is realised from its node's ``params.actuator``
kernel (a draw at the intended level) when the node decides, and stands until realised again; an
emission is realised by the step that owns it, from the point or the belief that step carries. A
modality, or a control node, that ``System.observes`` names reads the variable it observes as it
stands at its reader's turn, and never a value that does not yet exist.

**Timing.** A round is one tick under every order, and every stepped node steps once. A stepped
node whose control node is identified with another stepped node's action node steps after that
action is realised in the round, and the slot carries the realised value. ``"simultaneous"``
(default): the nodes none of whose control nodes is driven by an action node step on the round's
boundary (phase one), their action nodes realised together; the others step after, in an order
that respects the rule (phase two). A node of phase two reads a control node driven by another
node's action as realised this round, and every other slot, an emission-driven control node
included, at the boundary. ``"sequential"``: the listing orders the sources, the nodes none of
whose control nodes an action drives, which step in listing order, each reading what stands and
realising its action nodes as it chooses; each driven node steps as soon as the last of its
drivers has realised, before the sweep returns to nodes made ready earlier (a chain follows its
source, a join waits for its last driver and steps there, nodes made ready by the same realisation
keep listing order), so a driven node's own place in the listing does not matter. ``"async"``:
sequential with a fresh random listing of the sources each round (reproducible from the run key,
host generator seeded by ``fold_in(key, N)``), one total order per round, every node reading what
stands at its turn and each driven node placed by the same derived rule; there is no boundary
read. Sequential is a systematic scan, async a random scan,
simultaneous a synchronous update. A cycle of action-to-control identifications is refused.

**Write-backs.** Under ``"simultaneous"`` every node's view in a round is taken from the parameter
home as the round began, and the params each learner returns are applied at the round's boundary,
after its last realisation and before the next round's reads. Two nodes of one round that wrote
back one parameter variable are combined only where it is Dirichlet counts: the round-start
concentration plus the sum of their deltas, the posterior given both nodes' data, and a sum that
is not positive is refused as ``write_back`` refuses one; for any other form two ``plan()`` steps
whose ``learn`` both reach the variable are refused before the first round, and two other steps
that both wrote it back in one round are refused at that round's boundary. Under
``"sequential"`` and ``"async"`` a write-back is live: the next view is taken from the home as it
then stands.

**Clocks.** A stepped node whose view declares durations (``params.durations``, one control node
of the node carrying a duration per level) decides once per option. On a decision round it steps
and ``run`` reads the duration of the level it chose (the entry of ``params.durations`` at the
intended level of the node that carries them),
times the node's entry of ``clocks`` (the rounds one unit of duration lasts, 1.0 by default); the
product must be a whole number of rounds of at least one, and the node does not step again until
that many rounds, the decision round included, have passed. On a held round there is no view, no
step call, no write-back and no realisation: the node's action nodes and emissions keep their
values, and a variable it drives keeps its level. The first round always decides. The count is
``run``'s own and is not in the step's carry, so ``carries`` seeds a step and never a count. A node
with durations refuses a non-identity actuator on its action nodes, and ``Trajectory.decided``
records which rounds decided.

**Keys.** The run key splits into ``(key, k_reset)``; stepped node ``j`` (in the order of the
System's components, a joint node at the place of its node) calls its reset step with ``k_reset``
for ``j = 0`` and ``fold_in(k_reset, j)`` after. Each round ``(key, k_tick) = split(key)`` and
``k_tick`` splits into ``2N + 1`` keys for ``N`` stepped nodes: node ``j`` steps with key ``j``;
its action nodes are realised with key ``N + j``, except the nodes of phase one under
``"simultaneous"``, realised together with the last key. Realising with key ``k`` draws action
node ``i`` of the ``A`` action nodes of the System (in order, the action nodes of driven control
nodes excluded) from its kernel's column at the intended level with key ``i`` of
``split(fold_in(k, ACTUATOR_TAG), A)``. An emission is realised with the owner's step key folded
by ``EMIT_TAG``.

Design: docs/design/engine-consolidation.md; nesting-and-coupling.md N3, N4, N5; control-states.md
C13.
"""

import dataclasses
import fnmatch
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.composition import driver as _driver
from hybrid_aif.composition.driver import (
    MindCarry,
    TickOrder,
    _actuate,
    _check_legal,
    _check_parameters_only,
    _is_identity,
)
from hybrid_aif.composition.joint import joint_planner
from hybrid_aif.composition.merge import (
    SEPARATOR,
    MixtureRefusal,
    _inputs_of,
    _subsystem_meta,
    driven_nodes,
    retake,
    subsystem,
    write_back,
)
from hybrid_aif.controls import FLOAT, LEVEL
from hybrid_aif.distributions import Categorical, ConditionalDirichlet, Dirichlet
from hybrid_aif.planning.base import Planner
from hybrid_aif.reads import EMIT_TAG
from hybrid_aif.system import System, SystemParams

__all__ = ["Step", "Trace", "Trajectory", "plan", "run", "sample"]

#: ``step(system, key, obs, carry, params) -> (out, carry, params)``: see the module docstring.
Step = Callable[
    [System, Array, "Mapping[str, Array | None] | None", Any, SystemParams],
    tuple[Mapping[str, Array], Any, SystemParams],
]


class Trace(NamedTuple):
    """What a planning step kept of one round's plan (``plan(trace=True)``, in ``MindCarry.trace``).

    ``q_pi`` is the posterior over the policy set, ``tree`` the term tree of the objective
    (``System.act``'s metrics), ``policies`` the enumerated policy set ``q_pi`` is over
    (``System.policies``, ``None`` unless the plan is over an enumerated set).
    """

    q_pi: Categorical
    tree: Any
    policies: Array | None


# ------------------------------------------------------------------------------------ plan


def _learn_arguments(learn) -> tuple | None:
    """``plan(learn=...)`` as the keywords of ``System.learn``, a hashable tuple of items."""
    if learn is None or learn is False:
        return None
    if learn is True:
        return ()
    if isinstance(learn, Mapping):
        return tuple(sorted(learn.items()))
    if isinstance(learn, Sequence) and not isinstance(learn, str):
        return (("lr", 0.0), ("lr_by_slot", tuple(sorted((k, 1.0) for k in learn))))
    raise TypeError(
        f"plan(learn={learn!r}): learn is None, True (every learnable slot), a mapping of "
        "System.learn keywords, or a sequence of slot keys"
    )


def _learn_keywords(items: tuple) -> dict[str, Any]:
    out = dict(items)
    if isinstance(out.get("lr_by_slot"), tuple):
        out["lr_by_slot"] = dict(out["lr_by_slot"])
    return out


@eqx.filter_jit
def _plan_tick(system, params, planner, kwargs, trace, learn, key, seen, carry):
    """One compiled round: infer, learn, act, push, emit; ``params`` is an argument, so equal
    views share one program. ``seen`` is keyed by variable where the view observes its own."""
    absent = {m: False for m, v in seen.items() if v is None}
    beliefs, _ = system.infer_states(params, seen, priors=carry.beliefs, present=absent or None)
    learned, learn_state = None, carry.learn_state
    if learn is not None and carry.previous is not None:
        extra = {} if carry.beliefs is None else {"priors": carry.beliefs}
        out = system.learn(
            params,
            seen,
            beliefs,
            prev_beliefs=carry.previous,
            action=carry.action,
            learn_state=carry.learn_state,
            **extra,
            **_learn_keywords(learn),
        )
        learned, learn_state = out if isinstance(out, tuple) else (out, carry.learn_state)
    held = params if learned is None else learned
    action, q_pi, tree = system.act(
        key, held, beliefs, planner=planner, metrics=trace, **dict(kwargs)
    )
    pushed = system.update_empirical_prior(held, beliefs, action)
    emitted = {}
    if system.emissions:
        names = system._names()
        values = {
            **beliefs,
            **{
                u: action[a]
                for u, a in zip(names.controls, names.actions, strict=True)
                if a in action
            },
        }
        emitted = system._emit(jax.random.fold_in(key, EMIT_TAG), held, values)
    stored = carry._replace(
        beliefs=pushed, learn_state=learn_state, previous=beliefs, action=action, trace=None
    )
    return action, emitted, stored, learned, (q_pi, tree) if trace else None


class _Plan:
    """The step :func:`plan` builds: a callable that also says what ``run`` needs of it. ``sliced``
    marks a step that is handed a view sliced at its driven control nodes; a wrapper around it
    copies the mark with ``functools.wraps``."""

    def __init__(self, planner, learn, trace, replace, act_kwargs):
        self.sliced = True
        self.planner = planner
        self.learn = _learn_arguments(learn)
        self.trace = bool(trace)
        self.replace = dict(replace or {})
        self.act_kwargs = tuple(sorted(act_kwargs.items()))
        self._replaced: dict[int, tuple[System, System]] = {}
        self._policies: dict[int, tuple[System, Any]] = {}

    def _view(self, system: System) -> System:
        if not self.replace:
            return system
        held = self._replaced.get(id(system))
        if held is None or held[0] is not system:
            held = (system, dataclasses.replace(system, **self.replace))
            self._replaced[id(system)] = held
        return held[1]

    def _enumerated(self, system: System, params: SystemParams):
        """The enumerated policy set of the view, computed once per view (``None`` unless the
        plan is over an enumerated set)."""
        held = self._policies.get(id(system))
        if held is None or held[0] is not system:
            policies = None
            if self.planner is None and system.planner == "enumeration":
                kinds = system._space(system._params_in(params, "plan")).kinds
                policies = None if FLOAT in kinds else system.policies(params)
            held = (system, policies)
            self._policies[id(system)] = held
        return held[1]

    def __call__(self, system, key, obs, carry, params):
        system = self._view(system)
        driven = [
            u
            for u, a in zip(system._table.control_names, system.action_names, strict=True)
            if a is None
        ]
        if driven:
            raise ValueError(
                f"plan(): the view has driven control node(s) {driven}, which a planner is handed "
                "sliced at the driver's level. A step that wraps plan() must carry its "
                "`sliced = True` mark: decorate the wrapper with functools.wraps(plan(...)) or "
                "set wrapper.sliced = True"
            )
        carry = MindCarry() if carry is None else carry
        if obs is None:
            legal = dict(self.act_kwargs).get("legal")
            if legal is not None:
                _check_legal(system, params, legal, "plan")
            if self.learn is not None and carry.learn_state is None:
                carry = carry._replace(learn_state=system.init_learn_state(params))
            return {}, carry, params
        renamed = dict(system.observes)
        seen = {renamed.get(slot, slot): value for slot, value in obs.items()}
        action, emitted, stored, learned, planned = _plan_tick(
            system,
            params,
            self.planner,
            self.act_kwargs,
            self.trace,
            self.learn,
            key,
            seen,
            carry._replace(trace=None),
        )
        if planned is not None:
            stored = stored._replace(
                trace=Trace(planned[0], planned[1], self._enumerated(system, params))
            )
        return {**action, **emitted}, stored, params if learned is None else learned


def plan(
    planner: Planner | None = None,
    learn=None,
    trace: bool = False,
    replace: Mapping[str, Any] | None = None,
    **act_kwargs,
) -> Step:
    """The step of a planning mind: infer, learn, act, push the prior, emit.

    Built once, compiled per view (equal views share one program, ``params`` being an argument).
    Each round, on its view: infer the posterior from the carried prior (a modality whose value is
    ``None`` is absent that round; a modality that observes an emission or action node of the view
    takes the datum of that variable); learn when ``learn`` is set and a previous round exists
    (``System.learn`` with the previous posterior and action); choose an action
    (``System.act(key, params, beliefs, planner=planner, **act_kwargs)``); push the posterior
    through the transition under the action as the next prior; realise every emission of the view
    from the posterior (the step's key folded with ``EMIT_TAG``). It returns the action nodes and
    the emissions, and the params it was given, or the learned ones. The carry is a
    :class:`~hybrid_aif.composition.driver.MindCarry`, ``None`` meaning a fresh one; the step's
    first call (``obs is None``) returns nothing and leaves the declared initial priors as the
    first round's prior.

    ``learn`` is ``None`` (nothing is learned), ``True`` (every learnable slot), a mapping of
    ``System.learn`` keywords, or a sequence of slot keys (those slots learn at rate 1, the rest at
    0). ``trace=True`` keeps each round's :class:`Trace` in ``carry.trace``. ``replace`` sets
    fields of the view (``{"policy_len": 2}``) before it plans; on a non-leaf node it is also
    where the joint view's configuration is given (as ``joint_planner`` takes it). ``act_kwargs``
    pass to ``System.act`` (``selection=``, ``legal=``, ...); a ``legal`` is checked on the first
    call. A view with a driven continuous control node is refused, since a planner holds a driven
    input at one level: use a custom step.
    """
    return _Plan(planner, learn, trace, replace, act_kwargs)


# ---------------------------------------------------------------------------------- sample


@eqx.filter_jit
def _sample_tick(system, params, key, state, inputs):
    emitted, state = system.sample_step(key, params, state, inputs)
    controls = system._sample_controls(key, params, inputs)
    names = system._table.control_names
    actions = {
        a: controls[u] for u, a in zip(names, system.action_names, strict=True) if a is not None
    }
    return emitted, state, actions


class _Sample:
    """The step :func:`sample` builds."""

    def __call__(self, system, key, obs, carry, params):
        if obs is None:
            emitted, state = system.sample_initial(key, params, state=carry)
            return emitted, state, params
        nodes = set(system._table.control_names)
        inputs = {k: v for k, v in obs.items() if k in nodes and v is not None}
        emitted, state, actions = _sample_tick(system, params, key, carry, inputs)
        return {**emitted, **actions}, state, params


def sample() -> Step:
    """The step of a process: its System sampled forward from its declarations.

    The first call (the reset) draws the initial state, or takes the partial point state the
    carry was seeded with through ``run(carries=...)``, and returns the emissions of that state.
    Each round calls :meth:`System.sample_step` on the unsliced view with the control nodes that
    other nodes drive taken from ``obs`` (a driven node's value as the driver realised it),
    returning every emission realised by its declared rule from the stepped state and the action
    node of each control node that no variable drives, drawn from its habit or uniformly. The
    carry is the point state, keyed by factor name.
    """
    return _Sample()


# -------------------------------------------------------------------------- the record


class Trajectory(NamedTuple):
    """What :func:`run` recorded: a tree with one node per component of the System.

    The root holds ``params`` (the parameter home after the last round); every node holds its
    ``parts`` (its components, by their name relative to the node) and the records below, each a
    mapping keyed by name relative to the node, a non-leaf node's mappings being the prefixed
    union of its parts':

    - ``received`` and ``present``: the value each modality slot, and each control node a
      variable drives, held for its step, and whether it held one, ``ticks + 1`` entries (entry
      ``t`` is what the step was handed at round ``t``, the last what the slots hold after the
      last round); an absent value is zeros;
    - ``emitted``: the realised value of each action node and emission of the node, ``ticks``
      entries;
    - ``intended``: each action node as its step chose it, ``ticks`` entries (equal to ``emitted``
      where no kernel in ``params.actuator`` moved the draw);
    - ``start``: the out-ports that held a value entering round 0, as the reset call left them;
    - ``carries`` and ``carry``: a stepped node's carry after each round (``ticks + 1`` entries,
      the first after the reset call) when ``record_carries`` asked for them, else ``None``, and
      its carry after the last round; for a node that is not itself stepped, a mapping from each
      stepped node beneath it, by relative name;
    - ``decided``: a bool per round, whether the node stepped (``False`` on a round a node that
      declares durations spent inside its option; ``True`` every round of any other node). On such
      a round ``received`` holds what the node's in-ports held, ``intended`` its held action and
      ``emitted`` its standing values.

    ``trajectory["agent.rover"]`` walks the tree by dotted path; a name that is no node is
    refused, naming the nodes. A node that is planned jointly is a leaf of the tree.
    """

    params: SystemParams | None
    parts: dict[str, "Trajectory"]
    received: dict[str, Array]
    present: dict[str, Array]
    emitted: dict[str, Array]
    intended: dict[str, Array]
    start: dict[str, Array]
    carries: Any
    carry: Any
    decided: Any

    def __getitem__(self, path):
        if not isinstance(path, str):
            return tuple.__getitem__(self, path)
        node, rest = self, path
        while rest:
            names = rest.split(SEPARATOR)
            for i in range(1, len(names) + 1):
                head = SEPARATOR.join(names[:i])
                if head in node.parts:
                    node, rest = node.parts[head], SEPARATOR.join(names[i:])
                    break
            else:
                raise KeyError(
                    f"{path!r} is no node of this trajectory at {rest!r}; "
                    f"the nodes here are {sorted(node.parts)}"
                )
        return node


# --------------------------------------------------------------------------- the nodes


def _reset_keys(key: Array, count: int) -> list[Array]:
    """The keys of the reset calls: node ``j`` takes ``key`` for ``j = 0``, ``fold_in(key, j)``
    after."""
    return [key if j == 0 else jax.random.fold_in(key, j) for j in range(count)]


def _round_keys(key: Array, count: int) -> list[Array]:
    """The keys of one round for ``count`` stepped nodes: node ``j`` steps with key ``j`` and
    realises with key ``count + j``; key ``2 * count`` realises phase one under simultaneous."""
    return list(jax.random.split(key, 2 * count + 1))


def _sweep_generator(key: Array, count: int) -> np.random.Generator:
    """The host generator the async order draws its permutations from, seeded by the run key
    (folded by the node count, so the per-round key stream is untouched)."""
    seed = jax.random.key_data(jax.random.fold_in(key, count))
    return np.random.default_rng(np.asarray(seed, np.uint32).ravel())


@dataclasses.dataclass
class _Node:
    """A stepped node and what ``run`` needs to route to it."""

    name: str
    index: int
    step: Callable
    joint: bool
    reads: dict[str, str]  # slot -> the variable it observes (modalities and driven nodes)
    driven: dict[str, str]  # the control nodes a variable outside the node drives
    inputs: tuple[str, ...]  # the variables a sliced view takes a level of, in order
    chosen: tuple[str, ...]  # the action nodes of the nodes it chooses
    emissions: tuple[str, ...]
    phase_two: bool = False

    @property
    def outs(self) -> tuple[str, ...]:
        return (*self.chosen, *self.emissions)


def _check_order(tick_order: str) -> None:
    if tick_order not in ("simultaneous", "sequential", "async"):
        raise ValueError(
            f"tick_order must be 'simultaneous', 'sequential', or 'async', got {tick_order!r}. "
            "A round is one tick under every order and every stepped node steps once. "
            "simultaneous = the nodes no variable drives step on the round's boundary, then the "
            "driven nodes step on what was just realised; sequential = the nodes no action drives "
            "step in the order of the System's components and each driven node steps as soon "
            "as the last node that drives it has realised, before the sweep returns to nodes made "
            "ready earlier, each reading what stands; async = sequential with a "
            "RANDOM order of the nodes no action drives each round (reproducible from the run "
            "key), driven nodes placed by the same rule, each reading what stands"
        )


def _structure(system: System, params: SystemParams, steps: Mapping[str, Callable]) -> list[_Node]:
    """The stepped nodes of ``system`` in component order, checked: ``steps`` an antichain over
    the System's components that covers every leaf, every modality fed, a joint node holding no
    sampled leaf."""
    if system.subsystems is None:
        raise ValueError(
            "run: this System has no components (it is not a merge or a coupling), so there is "
            "no leaf to give a step; build the System with couple() or merge(), or with_mirror()"
        )
    if not isinstance(steps, Mapping) or not steps:
        raise TypeError(
            "run: steps is a mapping from a leaf (or a non-leaf node planned jointly) to its "
            "step: plan(), sample() or a function step(system, key, obs, carry, params)"
        )
    table = dict(system.subsystems)
    order = list(table)
    leaves = [n for n, sub in table.items() if sub.config]
    unknown = sorted(set(steps) - set(table))
    if unknown:
        raise ValueError(
            f"run: steps names {unknown}, which are no component of this System; its leaves are "
            f"{leaves} and its nodes {sorted(set(order) - set(leaves))}"
        )
    for name, step in steps.items():
        if not callable(step):
            raise TypeError(
                f"run: steps[{name!r}] is a {type(step).__name__}; give plan(), sample() or a "
                "function step(system, key, obs, carry, params)"
            )

    def beneath(node: str) -> list[str]:
        return [n for n in order if n.startswith(f"{node}{SEPARATOR}")]

    for name in steps:
        for below in beneath(name):
            if below not in steps:
                continue
            if isinstance(steps[below], _Sample):
                raise ValueError(
                    f"run: steps names {name!r}, planned as one mind, and {below!r} beneath it "
                    "with sample(): the joint view would take the process's tables as part of its "
                    f"model; step the other leaves of {name!r} instead, and give {below!r} its "
                    "own sample()"
                )
            raise ValueError(
                f"run: steps names {name!r} and {below!r} beneath it; the keys of steps are an "
                f"antichain, so step {name!r} as one node or its parts one by one, not both"
            )
    covered = {
        leaf
        for leaf in leaves
        if any(leaf == k or leaf.startswith(f"{k}{SEPARATOR}") for k in steps)
    }
    missing = [leaf for leaf in leaves if leaf not in covered]
    if missing:
        raise ValueError(
            f"run: steps has no step for the leaf {missing[0]!r}"
            + (f" (nor for {missing[1:]})" if missing[1:] else "")
            + "; every leaf of a closed System is stepped, by its own key or by the key of a "
            "node it lies under"
        )
    observes = dict(system.observes)
    open_slots = [m for m in system.modality_names if m not in observes]
    if open_slots:
        raise ValueError(
            f"run: modality {open_slots[0]!r} is observed by no identification"
            + (f" (nor are {open_slots[1:]})" if open_slots[1:] else "")
            + ", so the System is open; feed it with Identify(<out-port>, "
            f"{open_slots[0]!r}) (an action node or an emission of another component), or "
            "couple the System to its mirror with with_mirror()"
        )
    names = system._names()
    action_of = dict(zip(names.controls, names.actions, strict=True))
    orphans = [u for u, a in action_of.items() if a is None and u not in observes]
    if orphans:
        raise ValueError(
            f"run: control node {orphans[0]!r} has no action node and no variable drives it, so no "
            "step would give it a value; identify an action node (or emission) of another "
            "component with it, hold it with a Clamp, or, for a model's own internal "
            "identification, close the System with with_mirror()"
        )
    nodes: list[_Node] = []
    for index, name in enumerate(n for n in order if n in steps):
        meta = _subsystem_meta(system, name)
        mods = [system.modality_names[i] for i in meta.modalities]
        chosen = tuple(action_of[u] for u in meta.controls if action_of.get(u) is not None)
        emissions = tuple(meta.emissions)
        inside = set(chosen) | set(emissions)
        joint = not table[name].config
        own = {slot: v for slot, v in driven_nodes(system, name).items() if v in chosen}
        if own and not joint:
            slot = next(iter(own))
            raise ValueError(
                f"run: node {name!r} has its control node {slot!r} driven by its own action node "
                f"{own[slot]!r}, a cycle of length one; drive the node from another component"
            )
        driven = {
            slot: variable
            for slot, variable in driven_nodes(system, name).items()
            if variable not in inside
        }
        reads = {m: observes[m] for m in mods if m in observes}
        reads.update(driven)
        valid = _inputs_of(system, name)
        inputs = tuple(v for v in valid if v not in inside)
        nodes.append(
            _Node(
                name,
                index,
                steps[name],
                joint,
                reads,
                driven,
                inputs,
                chosen,
                emissions,
            )
        )
    return nodes


def _check_timing(
    nodes: list[_Node],
) -> tuple[list[_Node], Callable[[Sequence[_Node]], list[_Node]]]:
    """Mark phase two (a driven control node whose driver is another node's action node) and
    return ``(phase two in an order that respects the rule, derive)``; refuse a cycle.
    ``derive(sources)`` is the order of one sweep of ``"sequential"`` or ``"async"``: the nodes no
    action drives (the sources) in the order given, and each driven node as soon as the last of
    its drivers has realised, before the sweep returns to nodes made ready earlier; the nodes one
    realisation makes ready keep listing order."""
    owner = {a: n for n in nodes for a in n.chosen}
    drivers: dict[str, set[str]] = {}
    for node in nodes:
        for variable in node.driven.values():
            if variable in owner:
                drivers.setdefault(node.name, set()).add(owner[variable].name)
    by_name = {n.name: n for n in nodes}
    for name, owners in drivers.items():
        by_name[name].phase_two = True
        if name in owners:
            raise ValueError(
                f"run: node {name!r} has a control node driven by its own action node; an "
                "identification of a node's action node with its own control node is a cycle"
            )
    placed: list[_Node] = []
    waiting = [n for n in nodes if n.phase_two]
    done = {n.name for n in nodes if not n.phase_two}
    while waiting:
        ready = next((n for n in waiting if all(o in done for o in drivers.get(n.name, ()))), None)
        if ready is None:
            raise ValueError(
                f"run: the identifications of action nodes with control nodes between "
                f"{sorted(n.name for n in waiting)} form a cycle, so no node can step after the "
                "one that drives it; break the cycle"
            )
        waiting.remove(ready)
        placed.append(ready)
        done.add(ready.name)

    def derive(sources: Sequence[_Node]) -> list[_Node]:
        """The order of one sweep: ``sources`` as given, and after each node the nodes it makes
        ready (the driven nodes whose last driver it was), in listing order, each followed by the
        nodes it makes ready in turn before the next of its siblings, depth first."""
        derived: list[_Node] = []
        done: set[str] = set()
        held = [n for n in nodes if n.phase_two]

        def step(node: _Node) -> None:
            derived.append(node)
            done.add(node.name)
            ready = [n for n in held if all(o in done for o in drivers[n.name])]
            for n in ready:
                held.remove(n)
            for n in ready:
                step(n)

        for source in sources:
            step(source)
        return derived

    return placed, derive


# ---------------------------------------------------------------------------- the views


class _Views:
    """The view of each stepped node, taken from the parameter home: built once per set of input
    levels and refreshed from the home when it changes, so equal Systems share one compiled
    program."""

    def __init__(self, system: System, steps: Mapping[str, Callable]):
        self.system = system
        self._config = {
            name: {k: v for k, v in getattr(step, "replace", {}).items() if k in _JOINT_CONFIG}
            for name, step in steps.items()
        }
        self._cache: dict[tuple, tuple] = {}

    def clear(self) -> None:
        """Forget every view, so the next is built from the home as it now stands."""
        self._cache.clear()

    def get(self, node: _Node, home: SystemParams, levels: tuple):
        held = (node.name, tuple((v, _hashable(lv)) for v, lv in levels))
        entry = self._cache.get(held)
        if entry is None:
            inputs = {v: lv for v, lv in levels} if levels else None
            if node.joint:
                view, vparams = joint_planner(
                    self.system, home, node.name, inputs=inputs, **self._config[node.name]
                )
            else:
                view, vparams = subsystem(self.system, home, node.name, inputs=inputs)
        elif entry[0] is home:
            return entry[1], entry[2]
        else:
            view = entry[1]
            vparams = retake(self.system, view, entry[2], home)
        self._cache[held] = (home, view, vparams)
        return view, vparams


def _hashable(level):
    """A level (an ``int``, ``None`` or a probability vector) as a dictionary key."""
    return tuple(np.asarray(level).tolist()) if isinstance(level, Array) else level


_JOINT_CONFIG = (
    "policy_len",
    "planner",
    "gamma",
    "num_iters",
    "decomposition",
    "planning_belief",
    "feedback",
    "mixture_schedule",
    "mixture_k",
    "epistemic",
    "terms",
)


def _null(kind_size: tuple[str, int]) -> Array:
    kind, size = kind_size
    if kind == LEVEL:
        return jnp.zeros((), jnp.int32)
    return jnp.zeros((size,), jnp.float32)


class _Held(NamedTuple):
    """A clocked node's option in force: the rounds left to hold after this one and the intended
    action it chose."""

    remaining: int
    action: Mapping[str, Array]


def _check_clocks(clocks: Mapping[str, float] | None, names: set[str], clocked: set[str]):
    """The unit conversion of each clock, a positive finite number of rounds per unit of
    duration, defaulting to 1.0 for a node that declares durations; a key that is no stepped node
    or a node that declares no durations is refused."""
    if clocks is not None and not isinstance(clocks, Mapping):
        raise TypeError(
            f"run: clocks is a {type(clocks).__name__}; pass a mapping from a stepped node to the "
            "rounds one unit of duration lasts"
        )
    clocks = dict(clocks or {})
    unknown = sorted(set(clocks) - names)
    if unknown:
        raise ValueError(
            f"run: clocks names {unknown}, which are no stepped node; the stepped nodes are "
            f"{sorted(names)}"
        )
    idle = sorted(set(clocks) - clocked)
    if idle:
        raise ValueError(
            f"run: clocks names {idle}, whose params declare no durations, so there is no option "
            "to convert; declare durations for the node's control node, or drop the entry"
        )
    return {n: _driver._rounds_per_unit(clocks.get(n, 1.0), f"run: clocks[{n!r}]") for n in clocked}


def _chosen_duration(view: System, vparams: SystemParams, chosen: Mapping[str, Array]) -> float:
    """The duration of the option ``chosen``: the entry of ``params.durations`` at the intended
    level of the action node of the control node that carries them, read directly so that a view
    with driven control nodes (a custom step's is unsliced) needs no slicing."""
    ((node, table),) = view._durations(view._params_in(vparams, "run"), "run").items()
    action = view.action_names[view._table.control_names.index(node)]
    return float(jnp.asarray(table)[jnp.asarray(chosen[action], jnp.int32)])


# ---------------------------------------------------------------------------------- run


def _run(
    system: System,
    params: SystemParams,
    ticks: int,
    key: Array,
    steps: Mapping[str, Callable],
    clocks: Mapping[str, float] | None = None,
    tick_order: TickOrder = "simultaneous",
    carries: Mapping[str, Any] | None = None,
    record_carries: bool = False,
) -> Trajectory:
    """:func:`~hybrid_aif.composition.runner.run`, the engine."""
    _check_order(tick_order)
    if not isinstance(system, System):
        raise TypeError(f"run: the first argument is a {type(system).__name__}, not a System")
    if not isinstance(params, SystemParams):
        raise TypeError(f"run: params is a {type(params).__name__}, not SystemParams")
    if isinstance(ticks, bool) or not isinstance(ticks, int | np.integer) or ticks < 0:
        raise ValueError(f"run: ticks={ticks!r} is not a non-negative integer")
    system._params_in(params, "run")
    nodes = _structure(system, params, steps)
    names = {n.name for n in nodes}
    unknown = sorted(set(carries or {}) - names)
    if unknown:
        raise ValueError(
            f"run: carries names {unknown}, which are no stepped node; the stepped nodes "
            f"are {sorted(names)}"
        )
    second, derive = _check_timing(nodes)
    for node in nodes:
        if getattr(node.step, "sliced", False):
            floats = [s for s in node.driven if _kind(system, params, s) == FLOAT]
            if floats:
                raise NotImplementedError(
                    f"run: node {node.name!r} is stepped by plan() and its control node "
                    f"{floats[0]!r} is driven by a continuous variable; a planner holds a driven "
                    "input at one level, so give the node a custom step"
                )
    first = [n for n in nodes if not n.phase_two]
    count = len(nodes)
    action_nodes = [a for a in system.action_names if a is not None]
    index = {a: i for i, a in enumerate(action_nodes)}
    kernels = system._actuator(params, "run")
    ports = system.ports(params)
    types = dict(ports.outs)
    views = _Views(system, steps)
    declared_by = {n.name: _declared_durations(params, n.name) for n in nodes}
    clocked = {name for name, declared in declared_by.items() if declared}
    scales = _check_clocks(clocks, names, clocked)
    for node in nodes:
        if node.name in clocked:
            declared = next(iter(declared_by[node.name]))
            if declared in node.driven:
                raise ValueError(
                    f"run: node {node.name!r} declares durations on its control node {declared!r}, "
                    "which another node's variable drives, so the node chooses no option and has "
                    "nothing to hold; drop the durations from this node's params (a mirror "
                    "carries none), or declare them on the node that chooses the option"
                )
    for node in nodes:
        if node.name not in clocked:
            continue
        for a in node.chosen:
            if a in kernels and not _is_identity(kernels[a]):
                raise ValueError(
                    f"run: node {node.name!r} declares durations and action node {a!r} has a "
                    "non-identity actuator in params.actuator; a decision held for its duration "
                    "is realised once, so the kernel's draw would stand for the whole option and "
                    "the duration is read off the intended level; give the node an identity "
                    "actuator, or drop its durations"
                )

    if tick_order == "simultaneous":
        _check_shared_learning(nodes, views, params)

    values: dict[str, Array | None] = {v: None for n in nodes for v in n.outs}
    live_carry = {n.name: (carries or {}).get(n.name) for n in nodes}
    home = params
    key, k_reset = jax.random.split(key)
    # ---- the reset calls
    for node, k in zip(nodes, _reset_keys(k_reset, count), strict=True):
        view, vparams = _reset_view(views, node, home)
        out, live_carry[node.name], _ = node.step(view, k, None, live_carry[node.name], vparams)
        out = _check_out(node, out, types, system, reset=True)
        values.update(out)
    start = {n.name: {v: values[v] for v in n.outs if values[v] is not None} for n in nodes}
    sweep = _sweep_generator(key, count)

    received = {n.name: [] for n in nodes}
    emitted = {n.name: [] for n in nodes}
    intended = {n.name: [] for n in nodes}
    decided = {n.name: [] for n in nodes}
    held: dict[str, _Held] = {}
    carry_log = {n.name: [live_carry[n.name]] for n in nodes}

    def stands(node: _Node, variable: str) -> bool:
        """Whether ``node`` reads ``variable`` as it stands now rather than at the round's
        boundary. Under ``"sequential"`` and ``"async"`` every read stands; under
        ``"simultaneous"`` only a control node driven by an action node does, reading the value
        realised this round."""
        return tick_order != "simultaneous" or (
            variable in index and variable in node.driven.values()
        )

    def read(node: _Node, source: Mapping[str, Any], at_boundary: Mapping[str, Any]) -> dict:
        return {
            slot: (source if stands(node, variable) else at_boundary).get(variable)
            for slot, variable in node.reads.items()
        }

    def realise(k: Array, chosen: Mapping[str, Array], members: Sequence[str]) -> None:
        draws = [(index[a], kernels[a], chosen[a]) for a in members if a in kernels and a in chosen]
        drawn = _actuate(k, len(action_nodes), draws)
        for a in members:
            if a in chosen:
                values[a] = drawn.get(index[a], chosen[a])

    def turn(node: _Node, k: Array, at_boundary: Mapping[str, Any]) -> dict[str, Array]:
        """``node`` stepped on what it reads, its return checked, its emissions stored; the
        action nodes it chose (intended) are returned, to be realised by the caller. Under
        ``"simultaneous"`` the view is taken from the home as the round began and a write-back is
        held for the round's boundary; otherwise both are live."""
        nonlocal home
        base = round_home if tick_order == "simultaneous" else home
        seen = read(node, values, at_boundary)
        wanted = ()
        if getattr(node.step, "sliced", False):
            wanted = tuple(
                (v, _level_of((values if stands(node, v) else at_boundary).get(v)))
                for v in node.inputs
            )
        view, vparams = views.get(node, base, wanted)
        sliced = {s for s, v in node.driven.items() if v in {w for w, _ in wanted}}
        obs = {s: x for s, x in seen.items() if s not in sliced}
        out, live_carry[node.name], learned = node.step(
            view, k, obs, live_carry[node.name], vparams
        )
        if learned is not vparams:
            if not isinstance(learned, SystemParams):
                raise ValueError(
                    f"run: the step of {node.name!r} returned {type(learned).__name__} as its "
                    "params; a step returns the SystemParams it was given, or learned ones"
                )
            learned, written = _own_slots(view, node, vparams, learned, wanted, base)
            _check_parameters_only(view, node.name, vparams, learned)
            new = write_back(system, base, node.name, vparams, learned)
            if tick_order == "simultaneous":
                pending.append((node.name, new, written))
            else:
                home = new if not written else dataclasses.replace(new, **written(new))
                if written:
                    views.clear()
        received[node.name].append(seen)
        out = _check_out(node, out, types, system, reset=False)
        for e in node.emissions:
            values[e] = out[e]
        chosen = {a: out[a] for a in node.chosen}
        intended[node.name].append(chosen)
        if node.name in clocked:
            rounds = _driver._option_rounds(
                _chosen_duration(view, vparams, chosen),
                scales[node.name],
                f"run: the duration of control node {_driver._durations_node(view, vparams)!r} "
                f"of {node.name!r}",
                f"its clock of {scales[node.name]} rounds per unit",
            )
            held[node.name] = _Held(rounds - 1, chosen)
        return chosen

    def decide(node: _Node, k: Array, at_boundary: Mapping[str, Any]) -> dict[str, Array]:
        """``turn``, except that a clocked node inside its option steps nothing: it keeps the
        action it holds (its action nodes stand realised), and counts one round down."""
        state = held.get(node.name)
        if state is not None and state.remaining > 0:
            held[node.name] = state._replace(remaining=state.remaining - 1)
            received[node.name].append(read(node, values, at_boundary))
            intended[node.name].append(state.action)
            decided[node.name].append(False)
            return {}
        decided[node.name].append(True)
        return turn(node, k, at_boundary)

    round_home = home
    pending: list = []
    for round_ in range(ticks):
        key, k_tick = jax.random.split(key)
        keys = _round_keys(k_tick, count)
        boundary = dict(values)
        round_home = home
        pending.clear()
        if tick_order == "simultaneous":
            chosen: dict[str, Array] = {}
            for node in first:
                chosen.update(decide(node, keys[node.index], boundary))
            realise(keys[2 * count], chosen, [a for n in first for a in n.chosen])
        else:
            sources = (
                [first[int(i)] for i in sweep.permutation(len(first))]
                if tick_order == "async"
                else first
            )
            for node in derive(sources):
                realise(
                    keys[count + node.index], decide(node, keys[node.index], boundary), node.chosen
                )
        if tick_order == "simultaneous":
            for node in second:
                realise(
                    keys[count + node.index], decide(node, keys[node.index], boundary), node.chosen
                )
            if pending:
                home = _at_the_boundary(system, round_home, pending, round_)
                if any(written for _, _, written in pending):
                    views.clear()
        for node in nodes:
            emitted[node.name].append({v: values[v] for v in node.outs})
            carry_log[node.name].append(live_carry[node.name])

    final = {n.name: read(n, values, values) for n in nodes}
    return _assemble_record(
        system,
        params,
        home,
        nodes,
        ticks,
        types,
        ports,
        start,
        received,
        final,
        emitted,
        intended,
        carry_log if record_carries else None,
        live_carry,
        decided,
    )


def _own_slots(view, node, given, learned, wanted, home):
    """The tables a view learned that are its own, split off for the home.

    Returns ``(learned', write)``: ``learned'`` is ``learned`` with each own table (a likelihood,
    transition or initial prior that no parameter variable fills) put back to ``given``'s, so that
    what remains differing is the parameter variables, and ``write`` maps the home to the fields
    that hold the learned tables under their namespaced keys, or is ``None`` when none changed. A
    learned table whose shape differs from the home's (the view was sliced at a driven variable)
    is refused, naming the table and the variable."""
    filled = {(slot, i) for slot, i, _ in given.references}
    changed: dict[str, dict[str, Any]] = {}
    keep = {}
    for field, names in (
        ("likelihood", view.modality_names),
        ("transition", view._table.state_names),
        ("initial_state_prior", view._table.state_names),
    ):
        before, after = getattr(given, field), getattr(learned, field)
        pairs = {}
        for i, name in enumerate(names):
            if (field, i) in filled:
                continue
            if _same_tree(before[name], after[name]):
                continue
            home_entry = getattr(home, field)[name]
            if not _same_shape(home_entry, before[name]):
                sliced = [v for v, level in wanted] or ["a driven variable"]
                raise ValueError(
                    f"run: the step of {node.name!r} learned {field}[{name!r}], whose shape "
                    f"differs from the home's because the view is sliced at {sliced}; learn it "
                    "through a declared parameter variable, or leave that slot out of learn"
                )
            pairs[name] = after[name]
        if pairs:
            changed[field] = pairs
            keep[field] = {**after, **{n: before[n] for n in pairs}}
    if not changed:
        return learned, None
    restored = dataclasses.replace(learned, **keep)

    def write(home):
        return {field: {**getattr(home, field), **pairs} for field, pairs in changed.items()}

    return restored, write


def _same_tree(a, b) -> bool:
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    if ta != tb or len(la) != len(lb):
        return False
    return all(
        x is y or bool(jnp.array_equal(x, y, equal_nan=True)) for x, y in zip(la, lb, strict=True)
    )


def _same_shape(a, b) -> bool:
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    return ta == tb and all(jnp.shape(x) == jnp.shape(y) for x, y in zip(la, lb, strict=True))


def _kind(system: System, params: SystemParams, control: str) -> str:
    names = system._names(system._params_in(params, "run"))
    return names.kind(names.controls.index(control))


def _level_of(value) -> "int | Array | None":
    """The input a sliced view takes for a variable's value: an ``int`` for a level, the vector
    itself otherwise, ``None`` before the variable has a value."""
    if value is None:
        return None
    array = jnp.asarray(value)
    return int(array) if array.ndim == 0 and jnp.issubdtype(array.dtype, jnp.integer) else array


def _learned_variables(step: Callable, variables) -> set[str]:
    """The parameter variables among ``variables`` that ``step`` is known to learn: a ``plan()``
    step given ``learn`` learns a variable unless its rates leave it at a static zero
    (``learn=True`` learns every learnable slot, a sequence of slot keys only the slots named, a
    mapping the slots its ``lr`` and ``lr_by_slot`` reach). Any other step is not known in advance
    and is checked at the round's boundary."""
    if not isinstance(step, _Plan) or step.learn is None:
        return set()
    keywords = _learn_keywords(step.learn)
    lr = keywords.get("lr", 1.0)
    rates = dict(keywords.get("lr_by_slot") or {})
    out = set()
    for variable in variables:
        key = f"parameter:{variable}"
        if key in rates:
            rate = rates[key]
        else:
            matched = [r for k, r in rates.items() if "*" in k and fnmatch.fnmatchcase(key, k)]
            rate = matched[0] if matched else lr
        if not (isinstance(rate, int | float) and rate == 0):
            out.add(variable)
    return out


def _conjugate(declaration) -> bool:
    """Whether two write-backs of a variable of this form combine exactly by summing their
    deltas: Dirichlet concentrations, which are counts."""
    return isinstance(declaration, Dirichlet | ConditionalDirichlet)


def _check_shared_learning(nodes: list[_Node], views: _Views, home: SystemParams) -> None:
    """Under ``"simultaneous"``: refuse, before the first round, two ``plan()`` steps whose
    ``learn`` both reach one parameter variable whose form is not conjugate counts
    (``_at_the_boundary`` could not combine their write-backs). Other steps are checked there."""
    learners: dict[str, list[str]] = {}
    for node in nodes:
        variables = _reset_view(views, node, home)[1].parameters
        for variable in _learned_variables(node.step, variables):
            learners.setdefault(variable, []).append(node.name)
    for variable, names in learners.items():
        declaration = home.parameters[variable]
        if len(names) > 1 and not _conjugate(declaration):
            raise ValueError(
                f"run: parameter variable {variable!r} ({type(declaration).__name__}) is learned "
                f"by both {names[0]!r} and {names[1]!r} (their plan(learn=...)) under "
                '"simultaneous", where every write-back of a round lands at its boundary and two '
                "write-backs of one variable combine only as counts (Dirichlet concentrations, "
                'their deltas summed); run them under "sequential" or "async", or give one of them '
                f"a learn= that leaves 'parameter:{variable}' at 0"
            )


def _at_the_boundary(
    system: System, start: SystemParams, pending: list, round_: int
) -> SystemParams:
    """The home after a ``"simultaneous"`` round: ``start`` (the home as the round began) with
    every write-back of the round applied. A variable one node wrote takes its value; one that
    several wrote takes the start's concentration plus the sum of their deltas, the posterior
    given every node's data (Dirichlet counts only, refused for any other form, and refused, as
    ``write_back`` refuses one write-back, where the sum is not positive); each node's own learned
    tables are written under their names."""
    held = system._params_in(start, "run")
    homes = dict(held.parameters)
    writers: dict[str, list[tuple[str, Any]]] = {}
    for name, new, _ in pending:
        for variable, declaration in system._params_in(new, "run").parameters.items():
            if not _same_tree(declaration, homes[variable]):
                writers.setdefault(variable, []).append((name, declaration))
    for variable, entries in writers.items():
        if len(entries) == 1:
            homes[variable] = entries[0][1]
            continue
        begun = homes[variable]
        names = [name for name, _ in entries]
        if not _conjugate(begun):
            raise ValueError(
                f"run: parameter variable {variable!r} ({type(begun).__name__}) was written back "
                f"by both {names[0]!r} and {names[1]!r} in round {round_} under "
                '"simultaneous", where every write-back of a round lands at its boundary and two '
                "write-backs of one variable combine only as counts (Dirichlet concentrations, "
                'their deltas summed); run them under "sequential" or "async", or let one node '
                "learn it"
            )
        total = begun.concentration + sum(d.concentration - begun.concentration for _, d in entries)
        if not bool(jnp.all(total > 0.0)):
            raise ValueError(
                f"write_back: parameter variable {variable!r} has a concentration that is not "
                f"positive after the views of {names} learned, their deltas summed at the "
                f"boundary of round {round_} (a decay or rate that overshoots the anchor); it is "
                "not a Dirichlet and is not written"
            )
        homes[variable] = eqx.tree_at(lambda d: d.concentration, begun, total)
    out = dataclasses.replace(held, parameters=homes)
    for slot, i, variable in held.references:
        container = list(getattr(out, slot))
        container[i] = homes[variable]
        out = dataclasses.replace(out, **{slot: tuple(container)})
    out = system._params_out(out)
    for _, _, written in pending:
        if written:
            out = dataclasses.replace(out, **written(out))
    return out


def _declared_durations(params: SystemParams, name: str) -> dict:
    """The durations a view of the stepped node ``name`` declares, keyed by control node, read
    from the home with no view built (a view at no realised level may not exist): with
    ``subsystem``'s precedence, the System's own ``params.durations`` where it holds any (its
    entries on nodes at or beneath ``name``), else the ``parts`` of the leaves at or beneath it."""

    def beneath(key: str) -> bool:
        return key == name or key.startswith(f"{name}{SEPARATOR}")

    if params.durations:
        return {node: d for node, d in params.durations.items() if beneath(node)}
    out: dict = {}
    for leaf, part in (params.parts or {}).items():
        if beneath(leaf):
            out.update(part.durations or {})
    return out


def _reset_view(views: _Views, node: _Node, home: SystemParams):
    """The view of a node's reset call: averaged uniformly over its driven inputs, or, where
    ``subsystem`` refuses that average with a :class:`MixtureRefusal` (a Gaussian target or a
    linear-Gaussian transition conditional on a driven node would be a mixture), sliced at the
    first level of each input. A reset call reads no table, so the level
    is immaterial."""
    try:
        return views.get(node, home, _uniform(node))
    except MixtureRefusal:
        levels = _uniform(node)
        if not levels:
            raise
        return views.get(node, home, tuple((v, 0) for v, _ in levels))


def _uniform(node: _Node) -> tuple:
    """The input levels of a node's first view (nothing realised yet: each averaged uniformly)."""
    if not getattr(node.step, "sliced", False):
        return ()
    return tuple((v, None) for v in node.inputs)


def _check_out(
    node: _Node, out: Mapping[str, Array], types, system: System, reset: bool
) -> dict[str, Array]:
    """A step's return checked: a mapping keyed by out-ports of the node, each action node one
    level (or a float vector of its width), every emission present from the first round."""
    if not isinstance(out, Mapping):
        raise TypeError(
            f"run: the step of {node.name!r} returned a {type(out).__name__}; a step returns a "
            f"mapping from its out-ports {list(node.outs)} to values"
        )
    unknown = [k for k in out if k not in node.outs]
    if unknown:
        raise ValueError(
            f"run: the step of {node.name!r} returned {unknown}, which are no out-port of the "
            f"node; its out-ports are {list(node.outs)}"
            + (
                " (a control node is keyed by its action node)"
                if any(k in system._table.control_names for k in unknown)
                else ""
            )
        )
    if not reset:
        missing = [k for k in node.outs if k not in out]
        if missing:
            emission = [k for k in missing if k in node.emissions]
            what = (
                f"emission {emission[0]!r}, which its owner must realise every round (a plan() "
                "step does; a custom step calls view.emit(...))"
                if emission
                else f"action node {missing[0]!r}, which no step realises, so nothing would "
                "ever emit it"
            )
            raise ValueError(
                f"run: the step of {node.name!r} returned no value for {what}; "
                f"it returned {sorted(out)}"
            )
    checked = {}
    for k, value in out.items():
        kind, size = types[k]
        level = jnp.asarray(value)
        if k in node.emissions:
            soft = dict(system.emissions)[k].realise == "stats"
            floating = jnp.issubdtype(level.dtype, jnp.floating)
            if kind == LEVEL and floating and level.shape == (size,):
                pass  # a probability vector over the port's levels
            elif kind == LEVEL and not (
                level.shape == () and jnp.issubdtype(level.dtype, jnp.integer)
            ):
                raise ValueError(
                    f"run: the step of {node.name!r} returned {k!r} of dtype {level.dtype} and "
                    f"shape {tuple(level.shape)}; the port has {size} levels, so emit one integer "
                    f"level, or a probability vector of shape ({size},)"
                )
            elif kind == LEVEL and not 0 <= int(level) < size:
                raise ValueError(
                    f"run: the step of {node.name!r} returned {k!r} = {int(level)}; the port has "
                    f"{size} levels, so emit a level in [0, {size})"
                )
            elif kind == FLOAT and not soft and level.shape != (size,):
                raise ValueError(
                    f"run: the step of {node.name!r} returned {k!r} of shape "
                    f"{tuple(level.shape)}; the port is a float vector of width {size}, shape "
                    f"({size},)"
                )
            checked[k] = level
            continue
        if kind == FLOAT:
            if level.shape != (size,) or not jnp.issubdtype(level.dtype, jnp.floating):
                raise ValueError(
                    f"run: the step of {node.name!r} returned {k!r} of dtype {level.dtype} and "
                    f"shape {tuple(level.shape)}; a continuous action node takes a float vector "
                    f"of its width, shape ({size},)"
                )
            checked[k] = level
            continue
        if level.shape != () or not jnp.issubdtype(level.dtype, jnp.integer):
            raise ValueError(
                f"run: the step of {node.name!r} returned {k!r} of dtype {level.dtype} and shape "
                f"{tuple(level.shape)}; an action node takes one integer level, shape ()"
            )
        if not 0 <= int(level) < size:
            raise ValueError(
                f"run: the step of {node.name!r} returned {k!r} = {int(level)}; the action node "
                f"has {size} levels, so return a level in [0, {size})"
            )
        checked[k] = level.astype(jnp.int32)
    return checked


# ----------------------------------------------------------------------- the trajectory


def _stack(values: list, null: Array) -> tuple[Array, Array | None]:
    """``values`` (arrays or ``None``) stacked, absent ones as zeros of the first present shape;
    with the flags of which were present."""
    present = jnp.asarray([v is not None for v in values], dtype=bool)
    like = next((v for v in values if v is not None), null)
    filled = [jnp.zeros_like(like) if v is None else jnp.asarray(v) for v in values]
    stacked = (
        jnp.stack(filled) if filled else jnp.zeros((0, *jnp.shape(like)), jnp.asarray(like).dtype)
    )
    return stacked, present


def _assemble_record(
    system,
    params,
    home,
    nodes,
    ticks,
    types,
    ports,
    start,
    received,
    final,
    emitted,
    intended,
    carry_log,
    live_carry,
    decided,
) -> Trajectory:
    table = dict(system.subsystems)
    order = list(table)
    modality_types = system._modality_types(system._params_in(params, "run"))
    control_types = {u: t for u, t in ports.ins.items() if u in system._table.control_names}
    records: dict[str, Trajectory] = {}

    def slot_null(node: _Node, slot: str) -> Array:
        t = modality_types.get(slot) or control_types.get(slot) or types[node.reads[slot]]
        return _null(t)

    for node in nodes:
        r_rounds = received[node.name] + [final[node.name]]
        rec: dict[str, Array] = {}
        pres: dict[str, Array] = {}
        for slot in node.reads:
            rec[slot], pres[slot] = _stack([held[slot] for held in r_rounds], slot_null(node, slot))
        em: dict[str, Array] = {}
        for v in node.outs:
            em[v], _ = _stack([held[v] for held in emitted[node.name]], _null(types[v]))
        it: dict[str, Array] = {}
        for a in node.chosen:
            it[a], _ = _stack([held[a] for held in intended[node.name]], _null(types[a]))
        records[node.name] = Trajectory(
            None,
            {},
            _relative(rec, node.name),
            _relative(pres, node.name),
            _relative(em, node.name),
            _relative(it, node.name),
            _relative(start[node.name], node.name),
            None if carry_log is None else tuple(carry_log[node.name]),
            live_carry[node.name],
            jnp.asarray(decided[node.name], dtype=bool),
        )
    # the tree above the stepped nodes: one node per component
    parents = {
        name: max((p for p in order if name.startswith(f"{p}{SEPARATOR}")), key=len, default=None)
        for name in order
    }

    def union(name: str, kids: dict[str, Trajectory]) -> Trajectory:
        """The node ``name`` over its parts ``kids`` (by full name): the prefixed union of
        their records."""
        prefix = f"{name}{SEPARATOR}" if name else ""
        parts = {k.removeprefix(prefix): v for k, v in kids.items()}

        def merged(field: str) -> dict:
            out = {}
            for rel, part in parts.items():
                out.update({f"{rel}{SEPARATOR}{k}": x for k, x in getattr(part, field).items()})
            return out

        def collected(field: str, keep_none: bool = False) -> dict | None:
            out = {}
            for full, part in kids.items():
                rel, value = full.removeprefix(prefix), getattr(part, field)
                if value is None and not (keep_none and full in records):
                    continue
                if full in records:
                    out[rel] = value
                else:
                    out.update({f"{rel}{SEPARATOR}{k}": x for k, x in value.items()})
            return out or None

        return Trajectory(
            None,
            parts,
            merged("received"),
            merged("present"),
            merged("emitted"),
            merged("intended"),
            merged("start"),
            collected("carries"),
            collected("carry", keep_none=True),
            collected("decided"),
        )

    def build(name: str) -> Trajectory | None:
        if name in records:
            return records[name]
        kids = {k: build(k) for k in order if parents[k] == name}
        kids = {k: v for k, v in kids.items() if v is not None}
        return union(name, kids) if kids else None

    top = {k: build(k) for k in order if parents[k] is None}
    root = union("", {k: v for k, v in top.items() if v is not None})
    return root._replace(params=home)


def _relative(values: Mapping[str, Any], node: str) -> dict[str, Any]:
    prefix = f"{node}{SEPARATOR}"
    return {k.removeprefix(prefix): v for k, v in values.items()}


def run(
    system: System,
    params: SystemParams,
    ticks: int,
    key: Array,
    steps: Mapping[str, Callable],
    clocks: Mapping[str, float] | None = None,
    tick_order: TickOrder = "simultaneous",
    carries: Mapping[str, Any] | None = None,
    record_carries: bool = False,
) -> Trajectory:
    """Step a closed System for ``ticks`` rounds and record a :class:`Trajectory`.

    ``system`` is a merge or a coupling whose every modality is fed by an identification (a
    modality no identification feeds is refused naming it, and the sentence names the fix);
    ``params`` is its parameter home. ``steps`` maps each stepped node to its step: a leaf, or a
    non-leaf node that is then planned as one mind (``joint_planner`` of the node); the keys are
    an antichain over the System's components that covers every leaf. The module docstring
    states the step's contract, the rounds, the order of the nodes and the keys.

    ``tick_order`` is ``"simultaneous"``, ``"sequential"`` or ``"async"``. ``carries`` seeds the
    carry of the named stepped nodes (``None``, the default, lets each step make its own; a
    seeded carry of a ``sample()`` node is a partial point state); ``record_carries=True`` keeps
    every round's carry in ``Trajectory.carries``. ``clocks`` maps a stepped node whose view
    declares durations to the rounds one unit of duration lasts (1.0 for a node it omits), a
    positive finite number. A step that learns has its returned parameter variables written back
    to the home each round (the next view is taken from the new home), and ``Trajectory.params``
    is the home after the last round.

    Refused, each naming the fix: an open System; ``steps`` that is not an antichain covering
    every leaf, that names no component, or that names a non-leaf node and a ``sample()`` leaf
    beneath it; a step that omits an emission or an action node of its node on a round; a cycle
    of identifications of action nodes with control nodes; ``clocks`` or ``carries`` naming no
    stepped node, or ``clocks`` naming a node that declares no durations; a duration whose
    product with its clock is under one round or not a whole number of rounds; an action node
    of a node with durations under a non-identity actuator.
    """
    return _run(system, params, ticks, key, steps, clocks, tick_order, carries, record_carries)


#  the dispatcher every caller reaches shows the signature of the closed-System form
_driver.run.__signature__ = inspect.signature(run)
