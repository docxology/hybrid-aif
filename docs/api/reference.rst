Reference models
================

Reference content sits below the core library: worked environments and one replication of a
published agent. It reproduces things outside the library and may change without an API break.
The core never imports it.

Reference environments
----------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.reference.envs.MountainCar
   ~hybrid_aif.reference.envs.tmaze

Reference replication
---------------------

The R-AIF replication of a published agent, one module per concern.

.. autosummary::
   :toctree: _generated

   ~hybrid_aif.reference.raif.world
   ~hybrid_aif.reference.raif.agent
   ~hybrid_aif.reference.raif.crspp
   ~hybrid_aif.reference.raif.revise
   ~hybrid_aif.reference.raif.metrics
   ~hybrid_aif.reference.raif.columns
   ~hybrid_aif.reference.raif.seed
   ~hybrid_aif.reference.raif.run
