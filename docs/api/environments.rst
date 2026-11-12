Environments and reference content
==================================


Environments are worlds a System acts in, written as pure functions of state. ``GenerativeEnv`` makes
one from a generative model declared the same way an agent's is.

.. note::

   ``hybrid_aif.reference`` is a second tier. It reproduces things outside the library, such as a
   published architecture or a benchmark environment, and may change without an API break. The
   engine never imports it.

Environments
------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.envs.Env
   ~hybrid_aif.envs.GenerativeEnv
   ~hybrid_aif.envs.GenerativeEnvParams
   ~hybrid_aif.envs.make
   ~hybrid_aif.envs.broadcast_params

Reference environments (second tier)
------------------------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.reference.envs.MountainCar
   ~hybrid_aif.reference.envs.tmaze

Reference replication (second tier)
-----------------------------------

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
