.. _examples:

========
Examples
========

.. contents::
   :local:

PGURE-SVT can be used either from Python, or as a standalone executable.

The Python examples require the `HyperSpy <http://hyperspy.org>`_ multi-dimensional
data analysis toolbox to be installed. This provides a number of useful features
including data visualization and data import from a number of microscopy file
formats. To install it in a ``conda`` environment:

.. code-block:: bash

   conda install hyperspy -c conda-forge

An example Jupyter notebook for both of these examples is `provided here
<https://github.com/tjof2/pgure-svt/blob/master/examples/PGURE-SVT-HyperSpy-Demo.ipynb>`_.


Simulated dataset
-----------------

The first step is to load the simulated dataset from the ``examples/`` directory using HyperSpy.

.. code-block:: python

   import numpy as np
   import hyperspy.api as hs

   from pguresvt import hspy, mixed_noise_model

   # Load example dataset
   s = hs.load("examples/example.tif")

   # Truncate to 25 frames
   s = s.inav[:25]

   # Plot the result
   s.plot(navigator='slider')

Now we corrupt the dataset with using a noise generator for mixed Poisson-Gaussian noise,
according to the following equation, where the true, noise-free signal is :math:`\mathbf{X}_{0}`,
and the observed noisy signal is :math:`\mathbf{Y}`.

.. math::

    \mathbf{Y}=\alpha\mathbf{Z}+\mathbf{E}\;\textrm{ with }\;\begin{cases}
    \mathbf{Z}\thicksim\mathcal{P}\left(\frac{\mathbf{X}^{0}}{\alpha}\right)\\
    \mathbf{E}\thicksim\mathcal{N}\left(\mu,\sigma^{2}\right)
    \end{cases}


where :math:`\alpha` is the detector gain, :math:`\mu` is the detector offset, and
:math:`\sigma` is the (Gaussian) detector noise.

.. code-block:: python

   random_seed = 123
   detector_gain = 0.1
   detector_offset = 0.1
   detector_sigma = 0.1

   noisy_data = mixed_noise_model(
       s.data,
       alpha=detector_gain,
       mu=detector_offset,
       sigma=detector_sigma,
       random_seed=random_seed,
   )

   s_noisy = hs.signals.Signal2D(noisy_data)

   # Plot the noisy result
   s_noisy.plot(navigator="slider")

In this example we do not use the noise estimation procedure, and instead provide
the known parameters to the algorithm directly. This information is used by the
PGURE optimizer to calculate the appropriate threshold.

.. code-block:: python

   svt = hspy.HSPYSVT(
       patch_size=4,
       noise_alpha=detector_gain,
       noise_mu=detector_offset,
       noise_sigma=detector_sigma,
       tol=1e-5,
   )

   # Note that the denoising can take a few moments
   s_denoised = svt.denoise(s_noisy)

   # Plot the denoised result
   s_denoised.plot(navigator='slider')

Nanoparticle dataset
--------------------

In this example we apply PGURE-SVT to an experimental dataset of a nanoparticle acquired using ADF-STEM.
This image sequence contains 50 frames at a rate of 4 frames per second. The results of this denoising are
shown in Fig. 11 of the `paper <https://doi.org/10.1016/j.ultramic.2016.05.005>`_.

For larger images, such as the ``256x256`` pixels here, you can use the patch_overlap parameter to control
the trade-off between speed and accuracy of the denoising procedure. This reduces the number of patches
the algorithm works with, at the expense of introducing possible edge artefacts between patches.

For the experimental sequence, the detector offset (``noise_mu``) was known beforehand, so a
noise estimation procedure is used for the other values.

.. code-block:: python

   # Load example dataset and plot
   s_np = hs.load("nanoparticle.tif")
   s_np.plot(navigator="slider")

.. code-block:: python

   # Initialize with suggested parameters, optimized for speed
   expt_svt = hspy.HSPYSVT(patch_size=4, patch_overlap=2, noise_mu=0.075)

   # Run the denoising
   s_np_denoised = expt_svt.denoise(s_np)

   # Plot denoised data
   s_np_denoised.plot(navigator="slider")

Standalone executable
---------------------

The PGURE-SVT standalone executable uses a simple command-line interface
along with a separate parameter file.

.. code-block:: bash

   ./PGURE-SVT param.svt

.. warning::

   The standalone executable currently only supports 8-bit and 16-bit TIFF sequences.

An example file,
`examples/param.svt <https://github.com/tjof2/pgure-svt/blob/master/examples/param.svt>`_
is provided.

.. code-block:: python

   # Example parameter file for PGURE-SVT program

   # Specify the file name of the TIFF stack to be denoised
   filename             : ../test/examplesequence.tif

   # The start and end frames of the sequence to be denoised
   start_image          : 1
   end_image            : 50
