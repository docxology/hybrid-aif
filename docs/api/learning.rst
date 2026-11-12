Learning
========

Parameters are learned either by a conjugate update, where a Dirichlet or matrix-normal posterior
absorbs sufficient statistics in closed form, or by gradient on a trainable module. The conjugate
route leaves behind a closed-form log evidence, which is the quantity description length is
computed from. The training functions fit a whole world model, an actor and critic, or an ensemble
from replayed episodes.

Conjugate updates and evidence
------------------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.learning.update_likelihood_dirichlet
   ~hybrid_aif.learning.log_evidence
   ~hybrid_aif.learning.absorbed_count
   ~hybrid_aif.learning.Evidence

Training loops
--------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.train.fit_world
   ~hybrid_aif.train.world_loss
   ~hybrid_aif.train.WorldBatch
   ~hybrid_aif.train.WorldLosses
   ~hybrid_aif.train.assemble
   ~hybrid_aif.train.posterior_states
   ~hybrid_aif.train.imagine
   ~hybrid_aif.train.roll_batch
   ~hybrid_aif.train.rollout_returns
   ~hybrid_aif.train.lambda_return
   ~hybrid_aif.train.perc_ema
   ~hybrid_aif.train.init_retnorm
   ~hybrid_aif.train.term_value
   ~hybrid_aif.train.ImagineMetrics
   ~hybrid_aif.train.ReturnEstimates
   ~hybrid_aif.train.Rollout
   ~hybrid_aif.train.fit_ensemble
   ~hybrid_aif.train.ensemble_loss
   ~hybrid_aif.train.transitions
   ~hybrid_aif.train.EnsembleBatch
   ~hybrid_aif.train.EnsembleLosses

Episodes and replay buffers
---------------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.train.Episode
   ~hybrid_aif.train.ColumnSpec
   ~hybrid_aif.train.buffers.episode
   ~hybrid_aif.train.as_episode
   ~hybrid_aif.train.episode_columns
   ~hybrid_aif.train.stack_episodes
   ~hybrid_aif.train.fit_length
   ~hybrid_aif.train.ReplayBuffer
   ~hybrid_aif.train.StaticBuffer
   ~hybrid_aif.train.StreamBuffer
   ~hybrid_aif.train.add
   ~hybrid_aif.train.extend
   ~hybrid_aif.train.sample
   ~hybrid_aif.train.sample_windows

Reading data from disk
----------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.data.load_dataset
   ~hybrid_aif.data.detect_format
   ~hybrid_aif.data.Dataset
   ~hybrid_aif.data.Reader
   ~hybrid_aif.data.Writer
   ~hybrid_aif.data.Format

