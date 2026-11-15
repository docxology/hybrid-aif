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

Elsewhere: the `project page <../index.html>`_, the `demo notebook
<https://github.com/noumenal-labs/hybrid-aif/blob/main/examples/demo.ipynb>`_, and the longer guides ``docs/capabilities.md`` and ``docs/composing-systems.md`` in the
repository.

.. toctree::
   :maxdepth: 2
   :caption: Declaring a model

   system
   components
   shared
   distributions

.. toctree::
   :maxdepth: 2
   :caption: Composing and running

   composition
   running

.. toctree::
   :maxdepth: 2
   :caption: Inference, learning and planning

   inference
   learning
   planning

.. toctree::
   :maxdepth: 2
   :caption: Objectives

   objectives

.. toctree::
   :maxdepth: 2
   :caption: Records and descriptions

   records
   description
   data

.. toctree::
   :maxdepth: 2
   :caption: Environments

   environments

.. toctree::
   :maxdepth: 2
   :caption: Reference models

   reference

