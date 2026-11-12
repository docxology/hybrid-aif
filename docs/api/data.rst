Saving runs
===========

The ``hybrid_aif.data`` package reads and writes episodes on disk in two formats, a folder of
arrays and HDF5. It is pure numpy. A run's record can be written with a writer and read back with
``load_dataset``; the arrays it returns are what the replay facility takes. The neural training
buffers read the same formats.

Reading and writing
-------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.data.load_dataset
   ~hybrid_aif.data.detect_format
   ~hybrid_aif.data.Dataset
   ~hybrid_aif.data.Writer
   ~hybrid_aif.data.Format

For authors of a new format
---------------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.data.Reader

