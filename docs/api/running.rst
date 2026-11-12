Running a system
================

``run`` steps a closed System for a number of ticks and records a ``Trajectory``. A tick is one
round in which every component steps once. The two step functions, ``plan`` for a mind that plans
and ``sample`` for a process that is simply sampled forward, are what ``run`` calls on each
component; ``env_step`` wraps a positional environment as the second kind. The clock helpers at the
end turn per-tick kernels into kernels for a longer step.

Running
-------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.composition.run
   ~hybrid_aif.composition.plan
   ~hybrid_aif.composition.sample
   ~hybrid_aif.composition.env_step

What a run records
------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.composition.Trajectory
   ~hybrid_aif.composition.Trace
   ~hybrid_aif.composition.MindCarry

Clocks and durations
--------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.discretise
   ~hybrid_aif.compose

