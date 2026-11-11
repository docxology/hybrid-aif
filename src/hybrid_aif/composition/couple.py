"""Coupling through blankets: participants, typed ports, and the links between them.

A run's participants are **minds** and **processes**, each named at :func:`couple`. A mind is
``(System, SystemParams, Step)``: its in-ports are its modalities, its out-ports its action
nodes. A process is ``(GenerativeEnv, env_params)``: its in-ports are its action inputs (the
env's action nodes), its out-ports its observations. A port is written ``"participant.port"``
(the participant's name up to the first dot, the port's name after it).

Every port has a type, a kind and a size: a discrete port carries a level, its size the level
count; a continuous port carries a float vector, its size the width. A mind's action nodes are
typed by its control space (``params.control``, its readers' tables), its modalities by their
likelihoods; a process's inputs by its env's names, its observations by ``env_params.likelihood``.

A :class:`~hybrid_aif.composition.merge.Link` between two participants is an identity between an
out-port and an in-port of one type: an action node to a process input, a process observation to
a mind's modality, or an action node to a mind's modality. It carries no kernel and no
``realise``; anything that transforms is a System run as a process. What crosses a link is action
or sensation, never a belief: a mind's belief reaches another mind only as the realised value of
an action node, and a parameter only as a report of counts. (Inside one particle a modality may
read a belief of another component, a belief link declared at
:func:`~hybrid_aif.composition.merge.merge`; between particles nothing does.) An action
node is realised once, when its mind chooses, by a draw (``sample``) from that mind's
``params.actuator`` kernel (a delta where there is no entry); the realised value stands until the
mind chooses again, and every link from the node carries it. :func:`couple` checks the
participants, the ports and the links once and returns a :class:`Wiring`, which
:func:`~hybrid_aif.composition.driver.run` executes.

Design: docs/design/control-states.md C12, C13; docs/design/nesting-systems.md principle 6;
nesting-and-coupling.md N1, N2.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.composition.component_mirror import mirror_components, world_emissions
from hybrid_aif.composition.merge import Clamp, Identify, Link, _assemble, driven_nodes
from hybrid_aif.controls import FLOAT, LEVEL
from hybrid_aif.distributions import Categorical
from hybrid_aif.envs.base import Env
from hybrid_aif.envs.generative import GenerativeEnv, GenerativeEnvParams
from hybrid_aif.system import System, SystemParams
from hybrid_aif.system.core import _observation_type
from hybrid_aif.system.mirror import mirror

if TYPE_CHECKING:
    from hybrid_aif.composition.driver import MindStep

__all__ = ["Wiring", "couple", "with_mirror"]

#: A port's type: its kind (``"level"`` or ``"float"``) and its size (a level count or a width).
PortType = tuple[str, int]


class Mind(NamedTuple):
    """A mind as :func:`couple` holds it: its System, its params, its step function, and its
    action nodes' kernels as ``params.actuator`` declares them, checked."""

    system: System
    params: SystemParams
    step: "MindStep"
    kernels: dict[str, Array]


class Process(NamedTuple):
    """A process as :func:`couple` holds it: a keyed env and its params."""

    env: GenerativeEnv
    params: Any


class Wiring(NamedTuple):
    """A checked coupling: the participants, the links, and what feeds every in-port.

    ``minds`` and ``processes`` are keyed by participant name in the order given to
    :func:`couple` (minds respond in that order under ``"sequential"``); ``links`` as given;
    ``fixed`` holds the constant of each process input held at one; ``types`` gives every port's
    :data:`PortType`, keyed ``"participant.port"``; ``feeds`` maps every in-port to the out-port
    its link comes from, or to ``None`` for a process input in ``fixed``.
    """

    minds: dict[str, Mind]
    processes: dict[str, Process]
    links: tuple[Link, ...]
    fixed: dict[str, Array]
    types: dict[str, PortType]
    feeds: dict[str, str | None]

    def null(self, port: str) -> Array:
        """The value an out-port holds before its participant has emitted: level 0, or zeros
        of the port's width."""
        kind, size = self.types[port]
        if kind == LEVEL:
            return jnp.zeros((), jnp.int32)
        return jnp.zeros((size,), jnp.float32)


def describe_type(kind: str, size: int) -> str:
    """A port type in words, for a sentence."""
    if kind == LEVEL:
        return f"discrete, {size} levels"
    return f"continuous, width {size}"


def _process_observation_types(name: str, env: GenerativeEnv, params: Any) -> dict[str, PortType]:
    """A process's observations, typed from ``env_params.likelihood``; an outside env's own
    params give a continuous observation's width from its shape, and a discrete one no level
    count, which is refused."""
    mods = env.names.modalities
    likelihood = getattr(params, "likelihood", None)
    if isinstance(params, GenerativeEnvParams) and isinstance(likelihood, Mapping):
        return {m: _observation_type(likelihood[m], f"env_params.likelihood[{m!r}]") for m in mods}
    obs, _ = jax.eval_shape(lambda k: env.reset(k, params), jax.random.PRNGKey(0))
    out: dict[str, PortType] = {}
    for m in mods:
        shaped = obs[m]
        if jnp.issubdtype(shaped.dtype, jnp.integer):
            raise ValueError(
                f"couple(): process {name!r} emits {m!r} as a level, and its params give no level "
                "count; pass its params as GenerativeEnvParams keyed by name "
                "(system.from_positional('params', ...)), whose likelihood types the port"
            )
        if shaped.ndim != 1:
            raise ValueError(
                f"couple(): process {name!r} emits {m!r} with shape {tuple(shaped.shape)}; a "
                "continuous port carries a vector, shape (width,), so emit one"
            )
        out[m] = (FLOAT, int(shaped.shape[0]))
    return out


def _check_name(name: object, taken: set[str]) -> str:
    if not isinstance(name, str) or not name or "." in name:
        raise ValueError(
            f"couple(): participant name {name!r} is not a name; give each participant a "
            "non-empty string without a dot, since a port is written 'participant.port'"
        )
    if name in taken:
        raise ValueError(
            f"couple(): two participants are named {name!r}; give every mind and process its "
            "own name"
        )
    taken.add(name)
    return name


def _mind(name: str, entry: object) -> tuple[Mind, dict[str, PortType], dict[str, PortType]]:
    """A mind's entry checked, with its in-port and out-port types keyed by port name."""
    if not (isinstance(entry, tuple | list) and len(entry) == 3):
        raise TypeError(
            f"couple(): minds[{name!r}] is a {type(entry).__name__}; a mind is "
            "(System, SystemParams, Step)"
        )
    system, params, step = entry
    if not isinstance(system, System) or not isinstance(params, SystemParams):
        raise TypeError(
            f"couple(): minds[{name!r}] holds ({type(system).__name__}, "
            f"{type(params).__name__}, ...); a mind is (System, SystemParams, Step)"
        )
    if not callable(step):
        raise TypeError(
            f"couple(): minds[{name!r}]'s third entry is a {type(step).__name__}; pass its "
            "Step (agent_step(...) or ancestral_step(...))"
        )
    ports = system.ports(params)
    modalities = system.modality_names
    ins = {m: t for m, t in ports.ins.items() if m in modalities}
    outs = {a: t for a, t in ports.outs.items() if a in system.action_names}
    kernels = system._actuator(params, f"couple(): minds[{name!r}]")
    return Mind(system, params, step, kernels), ins, outs


def _process(name: str, entry: object) -> tuple[Process, dict[str, PortType], dict[str, PortType]]:
    """A process's entry checked, with its in-port and out-port types keyed by port name."""
    if isinstance(entry, Env) and not isinstance(entry, GenerativeEnv):
        raise TypeError(
            f"couple(): processes[{name!r}] is a bare positional {type(entry).__name__}; a "
            "process speaks names, so pass (GenerativeEnv.wrap(env, system), env_params)"
        )
    if isinstance(entry, Env) or not (isinstance(entry, tuple | list) and len(entry) == 2):
        raise TypeError(
            f"couple(): processes[{name!r}] is a {type(entry).__name__}; a process is "
            "(GenerativeEnv, env_params), as System.as_env returns it"
        )
    env, params = entry
    if isinstance(env, System):
        raise TypeError(
            f"couple(): processes[{name!r}] is a System; run it as a process with "
            "system.as_env(params), which gives (GenerativeEnv, env_params)"
        )
    if not isinstance(env, GenerativeEnv):
        raise TypeError(
            f"couple(): processes[{name!r}] holds a bare positional {type(env).__name__}; a "
            "process speaks names, so pass (GenerativeEnv.wrap(env, system), env_params)"
        )
    kernels = getattr(params, "actuator", None) or {}
    if kernels:
        raise ValueError(
            f"couple(): process {name!r} declares params.actuator {sorted(kernels)}; an actuator "
            "belongs to the mind that emits the action, in its params.actuator, so give the "
            "process its params with actuator={}"
        )
    names = env.names
    if names.actions and not names.sizes:
        raise ValueError(
            f"couple(): process {name!r} gives its inputs {list(names.actions)} no level count or "
            "width; wrap the env with GenerativeEnv.wrap(env, system, params), whose params "
            "type them"
        )
    ins = {a: (names.kind(k), names.sizes[k]) for k, a in enumerate(names.actions)}
    outs = _process_observation_types(name, env, params)
    return Process(env, params), ins, outs


def _couple_components(
    components: Mapping[str, tuple[System, SystemParams]],
    bindings: Sequence[Identify | Clamp],
    fixed: Mapping[str, object] | None,
) -> tuple[System, SystemParams]:
    """The coupling of ``components``: their splice and the bindings, port identifications and
    clamps only."""
    if not isinstance(components, Mapping) or not components:
        raise TypeError(
            "couple() takes components, a non-empty mapping from a name to (System, SystemParams)"
        )
    systems: dict[str, System] = {}
    params: dict[str, SystemParams] = {}
    for name, entry in components.items():
        if not (
            isinstance(entry, tuple | list)
            and len(entry) == 2
            and isinstance(entry[0], System)
            and isinstance(entry[1], SystemParams)
        ):
            raise TypeError(
                f"couple(): components[{name!r}] is not (System, SystemParams); a component is "
                "a System with its params (a Step belongs to run)"
            )
        systems[name], params[name] = entry
    refused = [b for b in bindings if not isinstance(b, Identify | Clamp)]
    if refused:
        raise ValueError(
            f"couple(): {refused[0]!r} is not an Identify or a Clamp, the bindings of a "
            "coupling: a Link is an arc of one model's graph (a structural link, a relay, a belief "
            "read), which merge() makes; a belief read is an emission and an observing slot, "
            "written with Identify of the two ports"
        )
    clamps = [Clamp(node, level) for node, level in dict(fixed or {}).items()]
    try:
        return _assemble(systems, params, [*bindings, *clamps], coupled=True)
    except (ValueError, TypeError, NotImplementedError) as exc:
        message = str(exc).removeprefix("merge: ")
        if message.startswith("couple"):
            raise
        raise type(exc)(f"couple(): {message}") from exc


def _fixed_value(port: str, value: object, kind: str, size: int) -> Array:
    """A constant for a process input, checked against the input's type like a link's source."""
    value = jnp.asarray(value)
    want = describe_type(kind, size)
    if kind == LEVEL:
        ok = value.shape == () and jnp.issubdtype(value.dtype, jnp.integer)
        if ok:
            ok = 0 <= int(value) < size
        if not ok:
            raise ValueError(
                f"couple(): fixed[{port!r}] is {np.asarray(value)!r} (dtype {value.dtype}, shape "
                f"{tuple(value.shape)}); process input {port!r} is {want}, so pass one integer "
                "level in range"
            )
        return value.astype(jnp.int32)
    ok = jnp.issubdtype(value.dtype, jnp.floating) and value.ndim == 1
    ok = ok and value.shape == (size,)
    if not ok:
        raise ValueError(
            f"couple(): fixed[{port!r}] has dtype {value.dtype} and shape {tuple(value.shape)}; "
            f"process input {port!r} is {want}, so pass a float vector of that width"
        )
    return value


def with_mirror(
    system: System,
    params: SystemParams,
    *,
    per_component: bool = False,
    lower: bool = True,
    param_key: Array | None = None,
    names: tuple[str, str] = ("model", "world"),
) -> tuple[System, SystemParams]:
    """``system`` coupled with its own mirror: the closed System of a model and the world it models.

    The coupling of ``names[0]`` (``system`` as given) and ``names[1]`` (``mirror(system,
    params, lower=lower, param_key=param_key)``, the world that generated what the model
    perceives): each modality of ``system`` that ``System.observes`` does not name is identified
    with the world's emission of the same name, and each action node with the world's control node
    it realises, which is then driven by it. Any System is accepted, a merged or coupled one
    included, and nothing is assumed of it; the result is a closed System, which
    :func:`~hybrid_aif.composition.runner.run` steps with ``plan()`` for the model's leaves and
    ``sample()`` for the world. A modality ``system`` feeds from its own emission is no in-port
    and gets no emission, as :func:`~hybrid_aif.system.mirror.mirror` drops it.

    ``lower=True`` (the default) gives the world each learnable declaration at its posterior
    mean, or at one draw under ``param_key``; ``lower=False`` keeps the declarations as the
    model has them. ``per_component=True`` couples the System with the merge of one mirror per
    component (:func:`~hybrid_aif.composition.component_mirror.mirror_components`) in place of the
    mirror of the System whole: a state factor of one component that another reads is then fed to
    the reader as an identity emission of its owner, and each component of the world is a leaf
    that :func:`~hybrid_aif.composition.runner.run` steps on its own.
    """
    if (
        not isinstance(names, tuple | list)
        or len(names) != 2
        or any(not isinstance(n, str) or not n or "." in n for n in names)
        or names[0] == names[1]
    ):
        raise ValueError(
            f"with_mirror: names={names!r} is not two distinct component names; give a pair such "
            "as ('model', 'world'), each a non-empty string without a dot"
        )
    model, world = names
    if per_component:
        mirrored, mirrored_params = mirror_components(
            system,
            params,
            lower=lower,
            param_key=param_key,
            face="with_mirror",
            drive_internal=False,
        )
    else:
        mirrored, mirrored_params = mirror(
            system, params, lower=lower, param_key=param_key, face="with_mirror"
        )
    fed = {slot for slot, _ in system.observes}
    emitted = world_emissions(system) if per_component else {}
    bindings = [
        Identify(f"{world}.{emitted.get(m, m)}", f"{model}.{m}")
        for m in system.modality_names
        if m not in fed
    ]
    names_of = system._names()
    bindings += [
        Identify(f"{model}.{a}", f"{world}.{u}")
        for u, a in zip(names_of.controls, names_of.actions, strict=True)
        if a is not None
    ]
    #  a control node the model drives internally (one of its action nodes identified with it) is
    #  driven in the world by the same action node
    bindings += [
        Identify(f"{model}.{variable}", f"{world}.{node}")
        for node, variable in driven_nodes(system).items()
        if variable in set(names_of.actions)
    ]
    #  the mirror shares no declaration object with the model, whatever ``lower`` says
    mirrored_params = jax.tree.map(jnp.array, mirrored_params)
    return couple({model: (system, params), world: (mirrored, mirrored_params)}, bindings)


def couple(
    components: Mapping[str, tuple[System, SystemParams]] | None = None,
    bindings: Sequence[Identify | Clamp] = (),
    fixed: Mapping[str, "int | Categorical | Array"] | None = None,
    *,
    minds: Mapping[str, tuple] | None = None,
    processes: Mapping[str, tuple] | None = None,
    links: Sequence[Link] = (),
) -> "tuple[System, SystemParams] | Wiring":
    """Components coupled through their ports into one ``(System, SystemParams)``.

    ``components`` maps a name to ``(System, SystemParams)``; the components stay leaves of the
    result, each with its own planner configuration (``Subsystem.config``), its own habit,
    precision, durations and scoring (``SystemParams.parts``) and its own recognition, with
    ``Subsystem.boundary == "coupled"`` (not :func:`~hybrid_aif.composition.merge.nested`). A
    ``bindings`` entry is an :class:`~hybrid_aif.composition.merge.Identify` of an out-port (an
    action node or an emission) with an in-port (a modality, or a control node: a driven
    control node, which carries the out-port's realised value and has no action node of its own),
    between ports of different components, or a :class:`~hybrid_aif.composition.merge.Clamp`;
    ``fixed`` maps a discrete control node (``"world.u_move"``) to the level, or the distribution
    over levels, it is held at, as a ``Clamp`` does. Every other binding is refused, naming
    :func:`~hybrid_aif.composition.merge.merge`: a structural or belief-read ``Link``, an
    ``Identify`` of two parameter variables.

    A coupling is the splice of its components followed by port identifications:
    ``|A| + |B| - k`` variables for components of ``|A|`` and ``|B|`` variables and ``k``
    identifications, a variable one out-port feeds to several in-ports counted once. It plans
    nothing as one mind: a planner field its leaves share it states, and holds
    :data:`~hybrid_aif.system.UNSET` where they differ, a face needing it refusing. A coupling
    may be merged and a merge coupled. A black box
    (:func:`~hybrid_aif.composition.black_box.black_box`, an external process declared by its
    ports alone) is a component like any other: its in-ports are control nodes an ``Identify``
    drives, its out-ports emissions an ``Identify`` feeds a modality or a control node from, and a
    ``Clamp`` on an in-port is refused.

    The keywords ``minds=``, ``processes=``, ``links=`` and ``fixed=`` (a constant per process
    input) give the checked :class:`Wiring` of separate participants that
    :func:`~hybrid_aif.composition.driver.run` executes, as below.

    Wiring form, participants and the links between them, checked into a :class:`Wiring`.

    ``minds`` maps a name to ``(System, SystemParams, Step)``, ``processes`` a name to
    ``(GenerativeEnv, env_params)`` (``System.as_env`` gives one; ``GenerativeEnv.wrap`` keys an
    outside positional env). ``links`` are :class:`~hybrid_aif.composition.merge.Link` s between
    ``"participant.port"``
    spellings; ``fixed`` holds a constant per tick for a process input, keyed the same way.

    Refused, each with its fix named: an unknown participant or port; a link against the
    allowed directions or between ports of two types; an in-port fed twice; a process input
    neither linked nor in ``fixed``; a mind's modality fed by no link (one that
    :func:`~hybrid_aif.composition.driver.particle` supplies from inside is exempt); ``fixed``
    naming anything but a process input, or a value of another type; a process whose
    ``env_params.actuator`` is non-empty (an actuator is the emitting mind's); a port whose
    participant gives it no level count or width. An out-port read by nothing is ignored.

    Scope: a modality is typed by its likelihood, so a mind or a process couples when each of its
    modalities has a tabular, neural (``MLPConditional``) or linear-Gaussian likelihood; an
    amortized mind (a ``PredictorHead`` likelihood) cannot be coupled.
    """
    if components is not None:
        if minds or processes or links:
            raise TypeError(
                "couple() takes components and bindings, or minds, processes and links of the "
                "wiring form, not both"
            )
        return _couple_components(components, bindings, fixed)
    if bindings:
        raise TypeError("couple(bindings=...) needs components: couple(components, bindings)")
    minds = dict(minds or {})
    processes = dict(processes or {})
    fixed = dict(fixed or {})
    taken: set[str] = set()
    mind_table: dict[str, Mind] = {}
    proc_table: dict[str, Process] = {}
    #  per participant: (role, in-port types, out-port types)
    ports: dict[str, tuple[str, dict[str, PortType], dict[str, PortType]]] = {}
    for name, entry in minds.items():
        _check_name(name, taken)
        mind_table[name], ins, outs = _mind(name, entry)
        ports[name] = ("mind", ins, outs)
    for name, entry in processes.items():
        _check_name(name, taken)
        proc_table[name], ins, outs = _process(name, entry)
        ports[name] = ("process", ins, outs)

    types: dict[str, PortType] = {}
    for name, (_, ins, outs) in ports.items():
        types.update({f"{name}.{p}": t for p, t in ins.items()})
        types.update({f"{name}.{p}": t for p, t in outs.items()})

    def resolve(port: object, end: str) -> tuple[str, str, str]:
        """``(participant, role, port)`` for one end of a link, or the sentence that refuses it."""
        what = "an out-port" if end == "source" else "an in-port"
        if not isinstance(port, str) or "." not in port:
            raise ValueError(
                f"couple(): a link's {end} is {port!r}; write a port as 'participant.port', "
                f"participants {sorted(ports)}"
            )
        who, name = port.split(".", 1)
        if who not in ports:
            raise ValueError(
                f"couple(): link {end} {port!r} names participant {who!r}, which is not one; the "
                f"participants are {sorted(ports)}"
            )
        role, ins, outs = ports[who]
        if end == "source" and role == "mind" and name in mind_table[who].system._emission_names:
            raise ValueError(
                f"couple(): link source {port!r} is an emission of mind {who!r}, and couple "
                f"realises action nodes only; link one of its action nodes {sorted(outs)}"
            )
        side, other = (outs, ins) if end == "source" else (ins, outs)
        if name not in side:
            candidates = sorted(f"{who}.{p}" for p in side)
            hint = (
                f" ({port!r} is {'an in-port' if end == 'source' else 'an out-port'}, so it "
                f"is a link's {'target' if end == 'source' else 'source'})"
                if name in other
                else ""
            )
            raise ValueError(
                f"couple(): link {end} {port!r} is not {what} of {role} {who!r}{hint}; its "
                f"{'out' if end == 'source' else 'in'}-ports are {candidates}"
            )
        return who, role, name

    feeds: dict[str, str | None] = {}
    checked: list[Link] = []
    for link in links:
        if not isinstance(link, Link):
            raise TypeError(
                f"couple(): links holds a {type(link).__name__}; pass Link(source, target)"
            )
        if not isinstance(link.source, str) or not isinstance(link.target, str):
            raise ValueError(
                f"couple(): Link({link.source!r}, {link.target!r}) joins ports, so its source "
                "and its target are each one 'participant.port'; a tuple of sources states a "
                "modality's parents, which merge() reads"
            )
        if link.realise is not None or link.kernel is not None:
            raise ValueError(
                f"couple(): Link({link.source!r}, {link.target!r}) carries a realise or a "
                "kernel, which a belief read at merge() declares; a link between participants "
                "carries the out-port's one realised value, so drop it"
            )
        _, s_role, _ = resolve(link.source, "source")
        t_who, t_role, t_port = resolve(link.target, "target")
        if t_role == "mind" and t_port in getattr(mind_table[t_who].step, "supplied", ()):
            raise ValueError(
                f"couple(): Link({link.source!r}, {link.target!r}) feeds modality {t_port!r}, a "
                "read supplied inside the particle from a sibling's belief, so the link would be "
                "overridden; remove the link"
            )
        if s_role == "process" and t_role == "process":
            raise ValueError(
                f"couple(): Link({link.source!r}, {link.target!r}) runs from a process's "
                "observation to a process's input; a link runs from an action node to a process "
                "input or a mind's modality, or from a process observation to a mind's modality"
            )
        s_type, t_type = types[link.source], types[link.target]
        if s_type != t_type:
            raise ValueError(
                f"couple(): Link({link.source!r}, {link.target!r}) joins "
                f"{describe_type(*s_type)} to {describe_type(*t_type)}; a link is an identity, "
                "so both ends must have one type"
            )
        if link.target in feeds:
            raise ValueError(
                f"couple(): in-port {link.target!r} is the target of two links, from "
                f"{feeds[link.target]!r} and {link.source!r}; an in-port takes one link"
            )
        feeds[link.target] = link.source
        checked.append(link)

    inputs = {f"{n}.{p}" for n, (role, ins, _) in ports.items() if role == "process" for p in ins}
    constants: dict[str, Array] = {}
    for port, value in fixed.items():
        if port not in inputs:
            raise ValueError(
                f"couple(): fixed= names {port!r}, which is not a process input; fixed= holds a "
                f"constant for a process input, one of {sorted(inputs)}"
            )
        if port in feeds:
            raise ValueError(
                f"couple(): process input {port!r} is both linked (from {feeds[port]!r}) and in "
                "fixed=; an in-port takes one feed, so drop one"
            )
        constants[port] = _fixed_value(port, value, *types[port])
        feeds[port] = None

    for name, (role, ins, _) in ports.items():
        for p in ins:
            port = f"{name}.{p}"
            if port in feeds:
                continue
            if role == "mind" and p in getattr(mind_table[name].step, "supplied", ()):
                continue  # a particle supplies this modality from inside (a sibling-belief read)
            if role == "process":
                raise ValueError(
                    f"couple(): process input {port!r} is fed by no link; link an action node "
                    f"to it, or hold it at a constant with fixed={{{port!r}: value}}"
                )
            raise ValueError(
                f"couple(): mind modality {port!r} is fed by no link; link a process "
                "observation or an action node to it, or drop the modality from the mind"
            )

    return Wiring(mind_table, proc_table, tuple(checked), constants, types, feeds)
