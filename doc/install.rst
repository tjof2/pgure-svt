.. _install:

============
Installation
============

conda-forge
-----------

To install ``pgure-svt`` in a ``conda`` environment (Linux and MacOS only
- Windows coming soon) from `conda-forge <https://conda-forge.org/>`_:

.. code-block:: bash

   conda install pgure-svt -c conda-forge

Building from source
--------------------

Dependencies
^^^^^^^^^^^^

To build from source, PGURE-SVT requires the following packages and libraries to be installed on your machine first.

- `CMake <http://www.cmake.org>`_ 2.8+
- `Armadillo <http://arma.sourceforge.net>`_ 6.400+
- `NLopt <http://ab-initio.mit.edu/wiki/index.php/NLopt>`_ 2.4.2+
- `LibTIFF <http://www.remotesensing.org/libtiff/>`_ - for the standalone executable only

When installing the Armadillo linear algebra library, it is recommended that you also install
a high-speed BLAS replacement such as OpenBLAS or MKL; more information can be found in the
`Armadillo <http://arma.sourceforge.net/faq.html#blas_lapack_replacements>`_ documentation.

Python
^^^^^^

Once you have installed the dependencies listed above, you can build the Python package from source:

.. code-block:: bash

   git clone https://github.com/tjof2/pgure-svt.git
   cd pgure-svt
   pip install -e .

Standalone executable
^^^^^^^^^^^^^^^^^^^^^

The standalone PGURE-SVT executable has been tested on Ubuntu 12.04+. You can
use CMake to compile and install PGURE-SVT.

To install the PGURE-SVT executable into ``/usr/bin``, use:

.. code-block:: bash

   git clone https://github.com/tjof2/pgure-svt.git
   cd pgure-svt
   mkdir build
   cd build
   cmake ..
   make
   sudo make install

To change the install location, replace the last three lines with:

.. code-block:: bash

   cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ..
   make
   sudo make install

.. note::

   For OSX users, you may need to use the GCC compiler rather than the default.