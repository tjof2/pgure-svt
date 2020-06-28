[![Build Status](https://travis-ci.org/tjof2/pgure-svt.svg?branch=master)](https://travis-ci.org/tjof2/pgure-svt)
[![Coverage Status](https://coveralls.io/repos/github/tjof2/pgure-svt/badge.svg?branch=master)](https://coveralls.io/github/tjof2/pgure-svt?branch=master)
[![DOI](https://zenodo.org/badge/48366354.svg)](https://zenodo.org/badge/latestdoi/48366354)

# PGURE-SVT
PGURE-SVT (Poisson-Gaussian Unbiased Risk Estimator - Singular Value Thresholding) is an algorithm designed to denoise image sequences acquired in microscopy. It exploits the correlations between consecutive frames to form low-rank matrices, which are then recovered using a technique known as nuclear norm minimization. An unbiased risk estimator for mixed Poisson-Gaussian noise is used to automate the selection of the regularization parameter, while robust noise and motion estimation maintain broad applicability to many different types of microscopy.

More information on the algorithm can be found [in our paper](http://dx.doi.org/10.1016/j.ultramic.2016.05.005) in *Ultramicroscopy*.

If you use this code in a publication, please cite our work:

> T. Furnival, R. K. Leary and P. A. Midgley, "Denoising time-resolved microscopy sequences with singular value thresholding", *Ultramicroscopy*, vol. 178, pp. 112–124, 2017. DOI:[10.1016/j.ultramic.2016.05.005](http://dx.doi.org/10.1016/j.ultramic.2016.05.005)

PGURE-SVT is released free of charge under the GNU General Public License ([GPLv3](http://www.gnu.org/licenses/gpl-3.0.en.html)).

## Installation

<!-- The easiest way to install the package is with `pip`:

```bash
$ pip install -U ctrwfractal
$ pip install git+https://github.com/tjof2/ctrwfractal.git
``` -->

#### Dependencies
To successfully compile the C++ code, PGURE-SVT requires the following packages and libraries to be installed on your machine first.

- [CMake](http://www.cmake.org) 2.8+
- [Armadillo](http://arma.sourceforge.net) 6.400+
- [NLopt](http://ab-initio.mit.edu/wiki/index.php/NLopt) 2.4.2+
- [LibTIFF](http://www.remotesensing.org/libtiff/) - for the standalone executable only

When installing the Armadillo linear algebra library, it is recommended that you also install a high-speed BLAS replacement such as OpenBLAS or MKL; more information can be found in the [Armadillo](http://arma.sourceforge.net/faq.html#blas_lapack_replacements) documentation.

#### Python
To build the Python package from source:

```bash
$ git clone https://github.com/tjof2/pgure-svt.git
$ cd pgure-svt
$ pip install -e .
```

#### Standalone executable
PGURE-SVT has been tested on Ubuntu 12.04+. For OSX users, you may need to use the GCC compiler rather than the default.

Use CMake to compile and install PGURE-SVT. Note that it requires LibTIFF to be available on your system. To install the PGURE-SVT executable into `/usr/bin`, use:

```bash
$ git clone https://github.com/tjof2/pgure-svt.git
$ cd pgure-svt
$ mkdir build
$ cd build
$ cmake ..
$ make
$ sudo make install
```

To change the install location, type `cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ..`, and recompile with `make && sudo make install`.

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
PGURE-SVT can be integrated with the [HyperSpy](http://hyperspy.org) multi-dimensional data analysis toolbox, which provides a number of useful features including data visualization and data import from a number of microscopy file formats. An example Jupyter notebook is [provided here](https://github.com/tjof2/pgure-svt/blob/master/examples/PGURE-SVT-HyperSpy-Demo.ipynb).

#### Standalone executable
The PGURE-SVT standalone executable uses a simple command-line interface along with a separate parameter file. It is convenient to use on remote computing clusters. The parameter file allows the user to customize various options of the PGURE-SVT algorithm. Note that the standalone executable currently only supports 8-bit and 16-bit TIFF sequences.

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
