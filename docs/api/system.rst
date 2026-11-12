System and SystemParams
=======================

A ``System`` is the unit of model specification: one declaration of the state factors, the
observation modalities, the control nodes and the objective. ``SystemParams`` is the numeric state
that goes with it: the transition, likelihood and prior of each variable, the preferences, and the
parameters of anything learned. Everything else in the library takes or returns a System. A System
may be discrete, continuous or mixed. The entries a ``SystemParams`` holds are described under
:doc:`components`.

The declaration
---------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.System
   ~hybrid_aif.SystemParams

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

