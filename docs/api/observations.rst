Observations and likelihoods
============================

The ``likelihood`` entry of a modality gives the distribution of a reading given the state factors
it observes. It accepts:

- a table over discrete readings: :class:`~hybrid_aif.ConditionalCategorical`, or
  :class:`~hybrid_aif.ConditionalDirichlet` for a table that is learned;
- a linear reading with Gaussian noise: :class:`~hybrid_aif.LinearGaussianObservation`, whose
  noise may vary over the observed state through a noise map;
- a neural conditional: :class:`~hybrid_aif.MLPConditional`, or a
  :class:`~hybrid_aif.PredictorHead` with a named loss;
- a reference to a shared table: :class:`~hybrid_aif.Parameter` (see :doc:`shared`).

Linear-Gaussian readings
------------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.LinearGaussianObservation
   ~hybrid_aif.gaussian_emission

Noise that depends on state
---------------------------

These are parametrisations of an observation's noise, not factors of the model.

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.NoiseMap
   ~hybrid_aif.GridNoiseMap
   ~hybrid_aif.Grid
   ~hybrid_aif.cell_mass

