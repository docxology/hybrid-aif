Objectives
==========

The quantity a planner minimises is a tree of named terms, one block per state factor or modality.
Each term is an :class:`~hybrid_aif.Interval` that bounds a quantity such as risk, ambiguity or
information gain, and says whether the bound is exact. ``total`` folds a tree into the single
expected free energy. The default objective is the standard expected free energy.

Which variables carry an information-gain term is set by the ``epistemic`` selector of a
``System``, by name: a state factor, the parameters of a transition or of a likelihood. Leaving a
name out removes that term and nothing else. Preferences enter through the targets described under
:doc:`preferences`. New terms are added by registering a name and a sign.

Term trees
----------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.objectives.total
   ~hybrid_aif.objectives.merge
   ~hybrid_aif.objectives.map_leaves
   ~hybrid_aif.objectives.register_term

Intervals
---------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.Interval

