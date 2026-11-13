State inference
===============

Inference turns an observation into a posterior over state. The discrete scheme is mean-field fixed
point iteration. The continuous scheme conditions a Gaussian or a Gaussian mixture on a linear
observation exactly (on readings of independent noise, a ``DiagonalNoise``, in state space at a
cost linear in the number of readings), and rolls mixtures forward through a transition with a
reduction schedule so that the number of components stays fixed. A System calls these for you
through its ``infer_states`` method; the functions are exposed for direct use.

Discrete
--------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.inference.fpi
   ~hybrid_aif.inference.InferenceDetail

Linear-Gaussian and mixture
---------------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.inference.condition
   ~hybrid_aif.inference.DiagonalNoise
   ~hybrid_aif.inference.condition_mixture
   ~hybrid_aif.inference.predict
   ~hybrid_aif.inference.propagate
   ~hybrid_aif.inference.rollout_reduced
   ~hybrid_aif.inference.rollout_gpb1

