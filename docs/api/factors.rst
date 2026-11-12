Factors and observation models
==============================

A factor is one state variable of the model with its prior and transition. A modality is one
observation with its likelihood. The tabular forms hold discrete tables; the linear-Gaussian and
nonlinear-Gaussian forms hold continuous maps with a noise model. The helpers at the end build
common continuous pieces, such as an exact discretisation of a continuous-time system.

Discrete tables
---------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.TabularPrior
   ~hybrid_aif.TabularTransition
   ~hybrid_aif.TabularObservation

Continuous maps
---------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.LinearGaussianTransition
   ~hybrid_aif.LinearGaussianObservation
   ~hybrid_aif.NonlinearGaussianTransition
   ~hybrid_aif.Deterministic
   ~hybrid_aif.DerivedInitial

Noise that depends on state
---------------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.NoiseMap
   ~hybrid_aif.GridNoiseMap
   ~hybrid_aif.Grid
   ~hybrid_aif.cell_mass

Builders
--------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.gaussian_emission
   ~hybrid_aif.discretise
   ~hybrid_aif.compose
   ~hybrid_aif.accumulated_target

