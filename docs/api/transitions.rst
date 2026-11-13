States and transitions
======================

The ``transition`` entry of a state factor gives the distribution of its next value given its
parents: its own previous value, other state factors, and control nodes. It accepts:

- a table over discrete levels: :class:`~hybrid_aif.ConditionalCategorical`, or
  :class:`~hybrid_aif.ConditionalDirichlet` for a table that is learned;
- a linear map with Gaussian noise, optionally one map per level of a discrete parent (a switching
  linear model): :class:`~hybrid_aif.LinearGaussianTransition`;
- a nonlinear map with Gaussian noise, which the map may scale per coordinate by the state and
  the controls: :class:`~hybrid_aif.NonlinearGaussianTransition`;
- a neural conditional: :class:`~hybrid_aif.MLPConditional` for a continuous state,
  :class:`~hybrid_aif.MLPCategorical` for a categorical code;
- a deterministic map that is computed forward and carried:
  :class:`~hybrid_aif.Deterministic`;
- a reference to a shared table: :class:`~hybrid_aif.Parameter` (see :doc:`shared`).

The distribution families themselves are listed under :doc:`distributions`.

Continuous and deterministic transitions
----------------------------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.LinearGaussianTransition
   ~hybrid_aif.NonlinearGaussianTransition
   ~hybrid_aif.Deterministic

