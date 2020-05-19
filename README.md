# PGURE-SVT

[![Build Status](https://travis-ci.org/tjof2/pgure-svt.svg?branch=master)](https://travis-ci.org/tjof2/pgure-svt)
[![DOI](https://zenodo.org/badge/48366354.svg)](https://zenodo.org/badge/latestdoi/48366354)

PGURE-SVT (Poisson-Gaussian Unbiased Risk Estimator - Singular Value Thresholding) is an algorithm designed to denoise image sequences acquired in microscopy. It exploits the correlations between consecutive frames to form low-rank matrices, which are then recovered using a technique known as nuclear norm minimization. An unbiased risk estimator for mixed Poisson-Gaussian noise is used to automate the selection of the regularization parameter, while robust noise and motion estimation maintain broad applicability to many different types of microscopy.

More information on the algorithm can be found [in our paper](http://dx.doi.org/10.1016/j.ultramic.2016.05.005) in *Ultramicroscopy*.

PGURE-SVT is released free of charge under the GNU General Public License ([GPLv3](http://tjof2.github.io/pgure-svt/www.gnu.org/licenses/gpl-3.0.en.html)).

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

## Contents

- [PGURE-SVT](#pgure-svt)
  - [Contents](#contents)
  - [Installation](#installation)
      - [Building from source (Linux/OSX)](#building-from-source-linuxosx)
        - [Dependencies](#dependencies)
        - [Compilation](#compilation)
      - [Windows](#windows)
  - [Usage](#usage)
      - [Python](#python)
        - [Options](#options)
        - [Integration with HyperSpy](#integration-with-hyperspy)
      - [Standalone executable](#standalone-executable)

## Installation

#### Building from source (Linux/OSX)

PGURE-SVT has been tested on Ubuntu 12.04+. For OSX users, you may need to use the GCC compiler rather than the default.

##### Dependencies
To successfully compile the C++ code, PGURE-SVT requires the following packages and libraries to be installed on your machine first.

- [CMake](http://www.cmake.org) 2.8+
- [Armadillo](http://arma.sourceforge.net) 6.400+
- [NLopt](http://ab-initio.mit.edu/wiki/index.php/NLopt) 2.4.2+
- [LibTIFF](http://www.remotesensing.org/libtiff/) - for the standalone executable only

When installing the Armadillo linear algebra library, it is recommended that you also install a high-speed BLAS replacement such as OpenBLAS or MKL; more information can be found in the [Armadillo](http://arma.sourceforge.net/faq.html#blas_lapack_replacements) documentation.

When installing NLopt, make sure you specify to build the shared library with
`./configure --enable-shared`.

##### Compilation

By default, the system will build a C++ library for linking with Python. Installation of the standalone executable must be specified by the user. To build PGURE-SVT:

```bash
tar -xzf pgure-svt-0.4.2.tar.gz
cd pgure-svt-0.4.2
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

PGURE-SVT can be used either from Python, or as a standalone executable.

#### Python

PGURE-SVT comes with a simple Python wrapper that accepts any NumPy array with dimensions `(nx, ny, T)` for denoising.

```python
import numpy as np
from pguresvt import pguresvt

# Example dataset has dimensions (nx, ny, T),
# in this case a 128x128px video with 25 frames
X = np.random.randn(128, 128, 25)

# Initialize the algorithm
# with default parameters
svt = pguresvt.SVT()

# Run the algorithm on the data X
svt.denoise(X)

# Get the denoised data Y
Y = svt.Y
```

##### Options

For more information on the effects of the parameters, users are referred to the publication.

- `patchsize` - Dimensions of spatial patches in pixels
- `patchoverlap` - Step size in pixels by which successive spatial patches are displaced with respect to one another
- `length` - Length in frames of a temporal block (must be **odd** integer)
- `optimize` - Turn on (`True`) or off (`False`) optimization of lambda with PGURE
(if `False`, threshold must be specified)
- `threshold` - Option to specify lambda when PGURE is not used
(ignored when optimize=True)
- `estimatenoise` - Turn automated noise estimation on (`True`) or off (`False`)
(if `False`, `alpha`, `mu` and `sigma` must be specified)
- `alpha` - Option to specify detector gain
(ignored when `estimatenoise=True`)
- `mu` - Option to specify detector offset
(ignored when `estimatenoise=True`)
- `sigma` - Option to specify Gaussian noise component
(ignored when `estimatenoise=True`)
- `arpssize` - Size in pixels of ARPS motion estimation neighbourhood
- `tol` - PGURE optimization terminates after relative change is below tolerance
- `median` - Size of median filter window in pixels
- `hotpixelthreshold` - `n` * median absolute deviation above the image median, where `n` is the user value. Any hot pixels are replaced by the median of their immediate neighbours.
- `numthreads` - Number of threads to use on a multicore computer

##### Integration with HyperSpy

PGURE-SVT can be integrated with the HyperSpy multi-dimensional data analysis toolbox, which provides a number of useful features including data visualization and data import from a number of microscopy file formats. An example iPython notebook is [provided here](https://github.com/tjof2/pgure-svt/blob/master/examples/PGURE-SVT-HyperSpy-Demo.ipynb).

#### Standalone executable


The PGURE-SVT standalone executable uses a simple command-line interface along with a separate parameter file. It is convenient to use on remote computing clusters. The parameter file allows the user to customize various options of the PGURE-SVT algorithm. Note that the standalone executable currently only supports TIFF sequences.

```bash
./PGURE-SVT param.svt
```

An example file, [param.svt](https://github.com/tjof2/pgure-svt/blob/master/examples/param.svt) is provided.

```
##### Example parameter file for PGURE-SVT program #####

##### REQUIRED #####
# Specify the file name of the TIFF stack to be denoised
filename             : ../test/examplesequence.tif

# The start and end frames of the sequence to be denoised
start_image          : 1
end_image            : 50
```

Copyright (C) 2015-2020 Tom Furnival.
