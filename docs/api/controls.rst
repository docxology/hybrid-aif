Controls and actions
====================

A control node is a variable the System chooses when it plans. A discrete control node has levels;
a continuous one has a :class:`~hybrid_aif.Box` of bounds, held in ``SystemParams.control``. The
control a System decides and the action that is realised are distinct variables: an action node
follows its control node through an actuator kernel (``SystemParams.actuator``), which is the
identity unless one is declared. A discrete control node may carry a duration per level
(``SystemParams.durations``), the length of the step each choice takes.

Declarations
------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.Box

