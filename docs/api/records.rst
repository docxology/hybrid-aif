Logging and replay
==================

A run records a ``Trajectory``: for each tick and each component, what was observed, believed and
chosen, and optionally the full record of the objective, term by term and policy by policy. A
recorded trace can be replayed through a model, which recomputes beliefs and the free energy along
it without acting.

What a run records
------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.composition.Trajectory
   ~hybrid_aif.composition.Trace
   ~hybrid_aif.composition.MindCarry

Replaying a trace
-----------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.replay.replay
   ~hybrid_aif.replay_document
   ~hybrid_aif.Replay
   ~hybrid_aif.stack_traces

