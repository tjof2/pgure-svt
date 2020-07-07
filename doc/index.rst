.. pgure-svt documentation master file, created by
   sphinx-quickstart on Tue Apr 28 17:40:25 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PGURE-SVT's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   examples
   api


Installation
------------

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_.
Otherwise, to install ``pgure-svt``, you will first need to install its dependencies:

.. code-block:: bash

   pip install -U numpy

Then install pgure-svt:

.. code-block:: bash

   pip install -U pgure-svt

To upgrade, use the ``--upgrade`` flag provided by `pip`.

To check if everything worked fine, you can do:

.. code-block:: bash

	python -c 'import pgure-svt'

This should not produce any error message.

Quickstart
----------

TODO

.. code-block:: python

   import numpy as np
   from pgure-svt import OnlineRobustPCA

   # Generate toy dataset
   U = np.random.randn(500, 5)
   V = np.random.randn(50, 5)
   X = U @ V.T

   est = OnlineRobustPCA(n_components=5)
   Y = est.fit_transform(X)

Bug reports
-----------

Use the `GitHub issue tracker <https://github.com/tjof2/pgure-svt/issues>`_ to report bugs.

Cite
----

TODO


