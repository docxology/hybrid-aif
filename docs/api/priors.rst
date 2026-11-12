Initial state priors
====================

The ``initial_state_prior`` entry of a state factor gives its distribution at the start of an
episode. It accepts:

- over discrete levels: :class:`~hybrid_aif.Categorical`, or :class:`~hybrid_aif.Dirichlet` when
  the prior is learned, and their conditional forms :class:`~hybrid_aif.ConditionalCategorical` and
  :class:`~hybrid_aif.ConditionalDirichlet` when the prior has parents;
- over a continuous state: :class:`~hybrid_aif.Gaussian` or :class:`~hybrid_aif.MixtureGaussian`;
- a learned starting value for a carried deterministic node: :class:`~hybrid_aif.LearnedVector`;
- the state factor's own transition evaluated at the start: :class:`~hybrid_aif.DerivedInitial`.

Declarations specific to priors
-------------------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.DerivedInitial
   ~hybrid_aif.LearnedVector

