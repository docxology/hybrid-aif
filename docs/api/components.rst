Model components
================

A model is declared by filling named entries of ``SystemParams``, one per variable. A *state
factor* is one state variable of the model, with an initial prior and a transition. A *modality* is
one observation, with a likelihood. A *control node* is a variable the System chooses. In
factor-graph terms each entry is one factor of the generative model: a conditional distribution of
a variable given its parents, or in one case a deterministic map. The pages below are organised by
the role an entry plays, and each says which families the entry accepts.

.. toctree::
   :maxdepth: 1

   transitions
   observations
   priors
   preferences
   controls

