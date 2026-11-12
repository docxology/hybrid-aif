Preferences
===========

A preference is a target distribution: what the System expects, and so prefers, to observe or to
be. Preferences are entries of ``SystemParams`` and are ordinary distributions.

- ``obs_targets``, by modality: a :class:`~hybrid_aif.Categorical` over a discrete modality's
  levels, or a :class:`~hybrid_aif.Gaussian` or :class:`~hybrid_aif.MixtureGaussian` over a
  continuous reading.
- ``state_targets``, by state factor: a target over states. Under the name of a discrete control
  node, a :class:`~hybrid_aif.Categorical` is a fixed cost per level of that control.
- ``terminal_targets``, by state factor: a target scored once, at the end of the planning horizon.

A per-step target may be conditional on other variables, a control node among them (``C_deps`` on
the ``System``), and a terminal target may be conditional on other state factors (``H_deps``).
A mixture can be used to shape a target, for instance flat over a region and steep at its edge.
How targets enter the quantity a planner minimises is described under :doc:`objectives`.

Targets for steps of several ticks
----------------------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.accumulated_target

