hybrid-aif
==========


**hybrid-aif** is a JAX library for active inference over models that mix continuous and discrete
variables. A model is declared as a ``System`` of state factors, observation modalities and control
nodes, with mixture and switching structure available, and can be inferred, planned, learned and
composed with other Systems. Planning runs as parallel policy search on accelerators, and objectives
are assembled from named terms rather than fixed.

.. code-block:: bash

   uv venv && uv pip install -e "."        # from a checkout

The requirements are ``jax>=0.11``, ``equinox>=0.13.8`` and ``numpy>=2.0``.

Elsewhere: the launch page (link to come), the demo notebook ``examples/demo.ipynb`` (link to
come), and the longer guides ``docs/capabilities.md`` and ``docs/composing-systems.md`` in the
repository.

.. toctree::
   :maxdepth: 1
   :caption: Declaring a model

   system
   factors
   distributions

.. toctree::
   :maxdepth: 1
   :caption: Composing and running

   composition
   running

.. toctree::
   :maxdepth: 1
   :caption: Inference, planning and objectives

   inference
   objectives
   planning

.. toctree::
   :maxdepth: 1
   :caption: Learning and records

   learning
   description

.. toctree::
   :maxdepth: 1
   :caption: Worlds

   environments
