Shared parameters, emissions and recognition models
===================================================

Three further kinds of entry. A :class:`~hybrid_aif.Parameter` names a table held once and read by
several state factors or modalities, so that it is one variable with several readers. An
:class:`~hybrid_aif.Emission` is an output of the System drawn from its state factors, which other
Systems may observe. A :class:`~hybrid_aif.Recognition` declares an amortised inference model, a
network that maps readings to a posterior.

Declarations
------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.Parameter
   ~hybrid_aif.Emission
   ~hybrid_aif.Recognition

