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
- :mod:`~hybrid_aif.composition.driver`: :func:`run` closes the loop under a declared tick
  order, over a :class:`Wiring` (one :class:`MindRecord` or :class:`ProcessRecord` per
  participant) or over a merged composite, each :data:`Step` acting through its own view of the
  composite's params, which learning is written back to (a :class:`Trajectory`).

Design: docs/design/nesting-systems.md; nesting-and-coupling.md N1, N2.
"""

from hybrid_aif.composition.couple import Wiring, couple
from hybrid_aif.composition.driver import (
    HoldCarry,
    MindCarry,
    MindRecord,
    ParticleCarry,
    ProcessRecord,
    Step,
    Trajectory,
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

__all__ = [
    "Clamp",
    "HoldCarry",
    "Identify",
    "Link",
    "MindCarry",
    "MindRecord",
    "ParticleCarry",
    "ProcessRecord",
    "Step",
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
    "resolved_epistemic",
    "run",
    "subsystem",
    "subsystem_factors",
    "write_back",
]
