Introduction
=============

This manual describes ABEL blablah

For installation of ABEL, see :doc:`installation <installation>`

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
   :caption: Contents:
