The System
==========

A ``System`` is the unit of model specification: one declaration of the state factors, the
observation modalities, the control nodes and the objective, together with ``SystemParams``, the
numeric state that goes with it. Everything else in the library takes or returns a System. The
types below are the ones a model declaration names; the factors and distributions they hold have
their own pages.

The declaration
---------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.System
   ~hybrid_aif.SystemParams
   ~hybrid_aif.Box
   ~hybrid_aif.Interval
   ~hybrid_aif.Emission
   ~hybrid_aif.Recognition
   ~hybrid_aif.Parameter

Structure of a built System
---------------------------

These are carried on a System and read back from it, more often than they are written.

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.system.LearnState
   ~hybrid_aif.system.Ports
   ~hybrid_aif.system.Subsystem
   ~hybrid_aif.system.PartParams

