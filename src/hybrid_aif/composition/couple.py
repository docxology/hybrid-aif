"""Coupling through ports: Systems joined into one closed System by identifications.

A coupling's components are Systems, each named at :func:`couple` and kept as a leaf of the
result. A port is written ``"component.port"``: an out-port is an action node or an emission (a
value the component generates), an in-port a modality or a control node (a slot another component
can feed). An :class:`~hybrid_aif.composition.merge.Identify` joins an out-port of one component
to an in-port of another of the same type; a control node so identified is driven, carrying the
out-port's realised value. What crosses an identification is action or sensation, never a belief:
a mind's belief reaches another component only as the realised value of an action node or an
emission. An action node is realised once, when its node decides, by a draw from the node's
``params.actuator`` kernel (a delta where there is no entry); the realised value stands until the
node decides again, and every identification of it carries that value.
:func:`~hybrid_aif.composition.runner.run` steps the closed System a coupling makes, and
:func:`with_mirror` couples a System to its own mirror.

Design: docs/design/control-states.md C12, C13; docs/design/couple-and-run.md;
nesting-and-coupling.md N1, N2.
"""

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.composition.component_mirror import mirror_components, world_emissions
from hybrid_aif.composition.merge import Clamp, Identify, _assemble, driven_nodes
from hybrid_aif.distributions import Categorical
from hybrid_aif.system import System, SystemParams
from hybrid_aif.system.mirror import mirror

__all__ = ["couple", "with_mirror"]


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
    components: Mapping[str, tuple[System, SystemParams]],
    bindings: Sequence[Identify | Clamp] = (),
    fixed: Mapping[str, "int | Categorical | Array"] | None = None,
) -> tuple[System, SystemParams]:
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

    Refused, each with its fix named: a component that is not ``(System, SystemParams)``; an
    unknown component or port (the sentence lists the candidates); an identification of two ports
    of different types; an in-port identified twice; a binding that would drop an actuator
    kernel. :func:`~hybrid_aif.composition.runner.run` refuses a coupling with an in-port nothing
    feeds.

    Scope: a modality is typed by its likelihood, so a component couples when each of its
    modalities has a tabular, neural (``MLPConditional``) or linear-Gaussian likelihood; an
    amortized component (a ``PredictorHead`` likelihood) cannot be coupled.
    """
    return _couple_components(components, bindings, fixed)
