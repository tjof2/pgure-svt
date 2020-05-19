# PGURE-SVT

[![Build Status](https://travis-ci.org/tjof2/pgure-svt.svg?branch=master)](https://travis-ci.org/tjof2/pgure-svt)
[![DOI](https://zenodo.org/badge/48366354.svg)](https://zenodo.org/badge/latestdoi/48366354)

PGURE-SVT (Poisson-Gaussian Unbiased Risk Estimator - Singular Value Thresholding) is an algorithm designed to denoise image sequences acquired in microscopy. It exploits the correlations between consecutive frames to form low-rank matrices, which are then recovered using a technique known as nuclear norm minimization. An unbiased risk estimator for mixed Poisson-Gaussian noise is used to automate the selection of the regularization parameter, while robust noise and motion estimation maintain broad applicability to many different types of microscopy.

More information on the algorithm can be found [in our paper](http://dx.doi.org/10.1016/j.ultramic.2016.05.005) in *Ultramicroscopy*, and on the associated website: [http://tjof2.github.io/pgure-svt](http://tjof2.github.io/pgure-svt).

PGURE-SVT is released free of charge under the GNU General Public License ([GPLv3](http://tjof2.github.io/pgure-svt/www.gnu.org/licenses/gpl-3.0.en.html)).

## Citing

If you use this code in a publication, please cite our work:

> T. Furnival, R. K. Leary and P. A. Midgley, "Denoising time-resolved microscopy sequences with singular value thresholding", *Ultramicroscopy*, vol. 178, pp. 112– 124, 2017. DOI:[10.1016/j.ultramic.2016.05.005](http://dx.doi.org/10.1016/j.ultramic.2016.05.005)

```
@article{PGURESVT2017,
    title   = {Denoising time-resolved microscopy sequences with
               singular value thresholding.},
    author  = {Furnival, Tom and Leary, Rowan K. and Midgley, Paul A.},
    journal = {Ultramicroscopy},
    doi     = {10.1016/j.ultramic.2016.05.005},
    url     = {https://doi.org/10.1016/j.ultramic.2016.05.005}
    year    = {2017}
    volume  = {178},
    pages   = {112--124},
}
```

## Installation

#### Linux/Mac

PGURE-SVT has been tested on Ubuntu 12.04, 13.04 and 14.04. For Mac users, you may need to use the GCC compiler (v4.9+) rather than the default.

##### Dependencies
To successfully compile the C++ code, PGURE-SVT requires the following packages and libraries to be installed on your machine first.

- [CMake](http://www.cmake.org) 2.8+
- [Armadillo](http://arma.sourceforge.net) 6.400+
- [NLopt](http://ab-initio.mit.edu/wiki/index.php/NLopt) 2.4.2+
- [LibTIFF](http://www.remotesensing.org/libtiff/) - for the standalone executable only

When installing the Armadillo linear algebra library, it is recommended that you also install a high-speed BLAS replacement such as OpenBLAS or MKL; more information can be found in the [Armadillo](http://arma.sourceforge.net/faq.html#blas_lapack_replacements) documentation.

When installing NLopt, make sure you specify to build the shared library:
```bash
cd nlopt-2.4.2
./configure --enable-shared
make
sudo make install
```

##### Compilation

By default, the system will build a C++ library for linking with Python. Installation of the standalone executable must be specified by the user. To build PGURE-SVT, first unpack the source:

```bash
tar -xzf pgure-svt-0.4.2.tar.gz
cd pgure-svt-0.4.2
```

Next create a `build` directory for CMake to use when compiling and installing PGURE-SVT:

```bash
mkdir build
cd build
```

Use CMake to compile and install PGURE-SVT. By default, this will generate a shared library in the `/usr/lib` directory. It will also install the Python wrapper by running `python setup.py install`.

```bash
cd build
cmake ..
make
sudo make install
```

To change the install location, type `cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ..`, and recompile with `make && sudo make install`.

The standalone executable is **not** built by default, as it requires LibTIFF to be installed. To install the PGURE-SVT executable into `/usr/bin`, use:

```bash
cmake -DBUILD_EXECUTABLE=ON ..
make
sudo make install
```

#### Windows

The Windows binaries have been tested on 64-bit versions of Windows 7 and Windows 10. You may need to install the [Microsoft Visual C++ 2015 redistributable package](https://www.microsoft.com/en-gb/download/details.aspx?id=48145) before running PGURE-SVT.

**[Download 64-bit Windows release](https://github.com/tjof2/pgure-svt/releases/download/v0.3.3/PGURE-SVT_Win64.zip)**

## Usage

TODO

Copyright (C) 2015-2020 Tom Furnival.
