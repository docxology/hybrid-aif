Objectives
==========

An objective is a tree of named terms, one block per factor or modality, each term an ``Interval``
that bounds a quantity such as risk, ambiguity or information gain. ``total`` folds a tree into the
single expected free energy a planner minimises. New terms are added by registering a name and a
sign.

Term trees
----------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.objectives.total
   ~hybrid_aif.objectives.merge
   ~hybrid_aif.objectives.map_leaves
   ~hybrid_aif.objectives.register_term

