R-AIF
=====

A replication of R-AIF (`arXiv:2409.14216 <https://arxiv.org/abs/2409.14216>`_, "a practical
implementation of active inference with continuous variables and learned preferences"), pinned to
the authors' code and its logged runs as the oracle: their ``tiny`` RSSM world model and CRSPP
preference model as declared Systems, the assembled agent and its training step, self-revision,
their three metrics, and the resumable multi-seed Mountain Car harness. One module per concern.

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
