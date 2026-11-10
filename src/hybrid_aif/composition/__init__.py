"""Compositional active inference: Systems merged into one particle, Systems coupled as separate
particles by links, and the clock that runs them.

- :mod:`~hybrid_aif.composition.merge`: :func:`merge` makes several Systems one particle: a
  namespaced composite whose components share latents through bindings, of three kinds: a
  :class:`Link` (a driver's action node into a control node, a relayed state, a modality's
  parents, a belief read, a terminal target), an :class:`Identify` (two parameter variables made
  one, or an out-port joined to an in-port) and a :class:`Clamp` (a control node held at a
  constant); a component bound only by blanket-shaped bindings, belief reads and action-node
  links, is nested; :func:`subsystem` derives one mind's System from the composite and
  :func:`write_back` returns a view's learned parameter variables to it.
- :mod:`~hybrid_aif.composition.joint`: :func:`joint_planner` builds the one planning problem of
  a merged or coupled System, or of one of its components, as a plain ``(System, SystemParams)``:
  each control node driven by an action node inside the node is collapsed into the driver's, and
  the leaves' configuration, habit, precision, durations and scoring are composed, what cannot be
  joined refused.
- :mod:`~hybrid_aif.composition.couple`: :func:`couple` joins components through their ports
  into one ``(System, SystemParams)``: an :class:`Identify` of an out-port (an action node or an
  emission) with an in-port (a modality or a control node) and a :class:`Clamp`, each component
  kept as a leaf with its own configuration. The keywords ``minds=``, ``processes=`` and
  ``links=`` join separate particles through their blankets, a :class:`Link` from an action node
  to a process input or a mind's modality, or from a process observation to a mind's modality,
  typed and checked into a :class:`Wiring`.
- :mod:`~hybrid_aif.composition.runner`: :func:`run` steps a closed System (a merge or a coupling
  whose every modality is fed by an identification) for a number of rounds and records a
  :class:`Trajectory`. Each leaf, or a non-leaf node planned jointly, takes a :data:`Step`:
  :func:`plan` for a planning mind, :func:`sample` for a process, or a function of the same shape.
  :func:`with_mirror` couples a System to its own mirror, which closes it. A node whose view
  declares durations decides once per option, ``clocks=`` converting a duration into rounds.
- :mod:`~hybrid_aif.composition.component_mirror`: :func:`~hybrid_aif.system.mirror.mirror` with
  ``per_component=True``, the merge of one mirror per component of a merged or coupled System,
  a state factor of another component that a transition or a modality reads fed to the reader by
  an identity emission of its owner.
- :mod:`~hybrid_aif.composition.driver`: the wired form of :func:`run` (a :class:`Wiring`, one
  :class:`MindRecord` or :class:`ProcessRecord` per participant) and its composite form (a merged
  composite, each :data:`MindStep` acting through its own view of the composite's params, which
  learning is written back to, a :class:`CompositeTrajectory`), with :func:`particle`,
  :func:`hold`, :func:`agent_step` and :func:`ancestral_step`.

Design: docs/design/nesting-systems.md; nesting-and-coupling.md N1, N2.
"""

from hybrid_aif.composition.couple import Wiring, couple, with_mirror
from hybrid_aif.composition.driver import (
    CompositeTrajectory,
    HoldCarry,
    MindCarry,
    MindRecord,
    MindStep,
    ParticleCarry,
    ProcessRecord,
    agent_step,
    ancestral_step,
    hold,
    particle,
    run,
)
from hybrid_aif.composition.joint import joint_planner
from hybrid_aif.composition.merge import (
    Clamp,
    Identify,
    Link,
    driven_nodes,
    merge,
    namespaced_key,
    nested,
    resolved_epistemic,
    subsystem,
    subsystem_factors,
    write_back,
)
from hybrid_aif.composition.runner import Step, Trace, Trajectory, plan, sample

__all__ = [
    "Clamp",
    "CompositeTrajectory",
    "HoldCarry",
    "Identify",
    "Link",
    "MindCarry",
    "MindRecord",
    "MindStep",
    "ParticleCarry",
    "ProcessRecord",
    "Step",
    "Trace",
    "Trajectory",
    "Wiring",
    "agent_step",
    "ancestral_step",
    "couple",
    "driven_nodes",
    "hold",
    "joint_planner",
    "merge",
    "namespaced_key",
    "nested",
    "particle",
    "plan",
    "resolved_epistemic",
    "run",
    "sample",
    "subsystem",
    "subsystem_factors",
    "with_mirror",
    "write_back",
]
