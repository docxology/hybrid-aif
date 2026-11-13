Composing systems
=================

Several Systems become one in two ways. ``merge`` makes them one particle: a namespaced composite
whose components share a single blanket. ``couple`` keeps each a separate particle and joins them
through their ports. Bindings (``Link``, ``Identify``, ``Clamp``) say which variables of one
component are read by, driven by or identified with those of another. ``black_box`` stands in for
a process declared by its ports alone.

Making a composite
------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.composition.merge
   ~hybrid_aif.composition.couple
   ~hybrid_aif.composition.black_box
   ~hybrid_aif.composition.with_mirror
   ~hybrid_aif.composition.nested
   ~hybrid_aif.composition.subsystem
   ~hybrid_aif.composition.subsystem_states
   ~hybrid_aif.composition.write_back

Bindings
--------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.composition.Link
   ~hybrid_aif.composition.Identify
   ~hybrid_aif.composition.Clamp

Reading a composite
-------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.composition.namespaced_key
   ~hybrid_aif.composition.driven_nodes
   ~hybrid_aif.composition.is_black_box
   ~hybrid_aif.composition.resolved_epistemic
   ~hybrid_aif.composition.joint_planner

