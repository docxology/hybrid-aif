Model documents and description length
======================================

A System and its parameters can be written as a versioned JSON document, with an arrays file
beside it, and built back from that document. The same description is what description length is
measured on.
Replaying a recorded trace is described under :doc:`records`.

The model as a document
-----------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.describe
   ~hybrid_aif.build
   ~hybrid_aif.schema
   ~hybrid_aif.Description
   ~hybrid_aif.DescriptionError
   ~hybrid_aif.describe_parameters
   ~hybrid_aif.ParameterRecord
   ~hybrid_aif.register_transition
   ~hybrid_aif.registered_transitions
   ~hybrid_aif.description.save
   ~hybrid_aif.description.load
   ~hybrid_aif.description.check_observation_support
   ~hybrid_aif.description.check_joint_support

Description length
------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.DescriptionLength
