"""Compositional active inference: Systems merged into one particle, Systems coupled as separate
particles by wires, and the clock that runs them.

- :mod:`~hybrid_aif.composition.merge` — :func:`merge` makes several Systems one particle: a
  namespaced composite whose components share latents through bindings (control-node
  identification or relay, a modality's parents, target emission, boundary prior, a shared
  parameter variable, a modality's datum read from another component's state); a component bound
  only by blanket-shaped bindings, sibling-belief reads and control-node acts, is nested;
  :func:`subsystem` derives one mind's System from the composite and :func:`write_back` returns
  a view's learned parameter variables to it.
- :mod:`~hybrid_aif.composition.couple` — :func:`couple` joins separate particles, minds and
  processes, through their blankets: a :class:`Wire` from an action node to a process input or a
  mind's modality, or from a process observation to a mind's modality, typed and checked into a
  :class:`Wiring`.
- :mod:`~hybrid_aif.composition.driver` — :func:`run` closes the loop under a declared tick
  order, over a :class:`Wiring` (one :class:`MindRecord` or :class:`ProcessRecord` per
  participant) or over a merged composite, each mind acting through its own view (a
  :class:`Trajectory`).

Design: docs/design/nesting-systems.md; nesting-and-coupling.md N1, N2.
"""

from hybrid_aif.composition.couple import Wire, Wiring, couple
from hybrid_aif.composition.driver import (
    HoldCarry,
    MindCarry,
    MindRecord,
    ParticleCarry,
    ProcessRecord,
    Trajectory,
    ViewRespond,
    agent_respond,
    ancestral_respond,
    hold,
    particle,
    particle_params,
    run,
    view_respond,
)
from hybrid_aif.composition.merge import (
    ActionControl,
    BoundaryPrior,
    ObservationLink,
    SharedParameter,
    SiblingRead,
    TargetEmission,
    merge,
    namespaced_key,
    nested,
    resolved_epistemic,
    subsystem,
    subsystem_factors,
    write_back,
)

__all__ = [
    "ActionControl",
    "BoundaryPrior",
    "HoldCarry",
    "MindCarry",
    "MindRecord",
    "ObservationLink",
    "ParticleCarry",
    "ProcessRecord",
    "SharedParameter",
    "SiblingRead",
    "TargetEmission",
    "Trajectory",
    "ViewRespond",
    "Wire",
    "Wiring",
    "agent_respond",
    "ancestral_respond",
    "couple",
    "hold",
    "merge",
    "namespaced_key",
    "nested",
    "particle",
    "particle_params",
    "resolved_epistemic",
    "run",
    "subsystem",
    "subsystem_factors",
    "view_respond",
    "write_back",
]
