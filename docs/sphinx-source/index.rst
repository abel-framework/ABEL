Introduction
=============
The ABEL simulation framework is a particle-tracking framework for multi-element particle accelerators (such as plasma-accelerator linacs, colliders, experimental test facilities, etc.), implemented at varying levels of complexity, for fast investigations or optimizations. As a systems code, it can be used for physics simulations as well as generating (and optimizing for) cost estimates.

**Documentation is being written and being continuously updated!**

Python modules
===============

...

Config
++++++
.. autoclass:: abel.CONFIG
 :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
    :maxdepth: 2

   :hidden:

   citing_and_acknowledging

Installation
------------
.. toctree::
   :caption: INSTALLATION
   :maxdepth: 1
   :hidden:
   
   installation/local_setup.rst
   installation/hpc_setup.rst

Usage
-----
.. toctree::
   :caption: USAGE
   :maxdepth: 1
   :hidden:

   usage/get_started.rst
   usage/beams_and_trackables.rst
   usage/choosing_implementation.rst
   usage/running_simulations.rst
   usage/parameter_scans.rst
   usage/optimization.rst
   usage/collider_simulations.rst
   usage/experiment_simulations.rst
   usage/examples.rst

API documentation
-----------------
.. toctree::
   :caption: API DOCUMENTATION
   :maxdepth: 2
   :hidden:

   api_external/beam.rst
   api_external/trackables.rst
   api_external/cost_modeled.rst
   api_external/runnable.rst
   api_external/collider.rst
   api_external/wrappers.rst

Development
-----------
.. toctree::
   :caption: DEVELOPMENT
   :maxdepth: 1
   :hidden:

   development/github.rst
   development/developers.rst
   development/join_community.rst
