Environments
============


A world a System acts in can be declared three ways. The ``envs`` package holds the functional
form: an ``Env`` has a pure ``reset`` and ``step`` over a separate params pytree, and
``GenerativeEnv`` is one made from a generative model declared the same way an agent's is;
``System.as_env`` returns that pair for any tabular or hybrid System, its tensors run forward.
A System can also be a world as it stands: ``couple`` joins it to an agent through its ports,
``with_mirror`` couples a model with its own mirror, and ``black_box`` stands in for an external
process declared by its ports alone, all on the :doc:`composition` page.

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
