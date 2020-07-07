.. pgure-svt documentation master file, created by
   sphinx-quickstart on Tue Apr 28 17:40:25 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PGURE-SVT
=========

PGURE-SVT (Poisson-Gaussian Unbiased Risk Estimator - Singular Value Thresholding)
is an algorithm designed to denoise image sequences acquired in microscopy.
It exploits the correlations between consecutive frames to form low-rank matrices,
which are then recovered using a technique known as nuclear norm minimization.
An unbiased risk estimator for mixed Poisson-Gaussian noise is used to automate
the selection of the regularization parameter, while robust noise and motion
estimation maintain broad applicability to many different types of microscopy.

If you use this code in a publication, please cite our work:

* T. Furnival, R. K. Leary and P. A. Midgley, "Denoising
  time-resolved microscopy sequences with singular value
  thresholding", *Ultramicroscopy*, vol. 178, pp. 112–124,
  2017. DOI: `10.1016/j.ultramic.2016.05.005 <https://doi.org/10.1016/j.ultramic.2016.05.005>`_.

Contents
--------

.. toctree::
   :maxdepth: 1

   install
   examples
   api

Quickstart
----------

To install ``pgure-svt`` in a ``conda`` environment (Linux and MacOS only)
from `conda-forge <https://conda-forge.org/>`_:

.. code-block:: bash

   conda install pgure-svt -c conda-forge

For further details (including building from source),
see the :ref:`installation user guide <install>`.

Once installed, you can use ``pgure-svt`` as below:

.. code-block:: python

   import numpy as np
   from pguresvt import SVT

   # Example dataset has dimensions (nx, ny, nt),
   # in this case a 64x64px video with 25 frames
   X = np.random.randn(64, 64, 25)

   # Initialize the algorithm
   # with default parameters
   svt = SVT()

   # Run the algorithm on the data X
   svt.denoise(X)

   # Get the denoised data Y
   Y = svt.Y_

Contributing
------------

All contributions to PGURE-SVT are welcome!

There are many ways to contribute to PGURE-SVT, with the most common ones being contribution of code,
documentation or examples to the project. You can also help by by answering queries on the issue tracker,
investigating bugs, and reviewing other pull requests, or by reporting any issues you are facing.
Please use the `GitHub issue tracker <https://github.com/tjof2/pgure-svt/issues>`_ to report bugs.

Lastly, you can contribute by helping to spread the word about PGURE-SVT: reference the project
from your blog and articles, link to it from your website, or simply star it in GitHub to say "I use it".

License
-------

PGURE-SVT is released free of charge under the GNU General Public License
(`GPLv3 <http://www.gnu.org/licenses/gpl-3.0.en.html>`_).