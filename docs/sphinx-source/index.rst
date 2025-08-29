Introduction
=============

This manual describes ABEL blablah

Python modules
===============

There are four Python modules in the ``CLICopti`` library,
corresponding to the four header files in the C++ code.
Each of them are documented below.

To use the Python interface, first do ``import CLICopti``,
then the submodules are accessible as e.g. ``CLICopti.CellParams``.
The special submodule ``CLICopti.CLICopti`` contains most of the members of all submodules;
it is how the underlying C++ library is loaded into Python.

Basic building blocks
---------------------

These are the basic building blocks of ABEL.

Beam
++++
.. autoclass:: abel.Beam
 :members:

Technical infrastructure
------------------------

Config
++++++
.. autoclass:: abel.CONFIG
 :members:

Different things that can be tracked
------------------------------------

.. autoclass:: abel.Stage
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
-----
.. toctree::
   :caption: API DOCUMENTATION
   :maxdepth: 2
   :hidden:

   api_external/beam.rst
   api_external/trackables.rst
   api_external/cost_modeled.rst
   api_external/runnable.rst
   api_external/collider.rst
   api_external/wrapper_apis.rst

Development
---------
.. toctree::
   :caption: DEVELOPMENT
   :maxdepth: 1
   :hidden:

   development/github.rst
   development/developers.rst
   development/join_community.rst
