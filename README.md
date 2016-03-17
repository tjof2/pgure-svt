# PGURE-SVT

PGURE-SVT is an algorithm designed to denoise image sequences acquired in microscopy. It exploits the correlations between
consecutive frames to form low-rank matrices, which are then recovered using a technique known as nuclear norm minimization.
An unbiased risk estimator for mixed Poisson-Gaussian noise is used to automate the selection of the regularization parameter, while
robust noise and motion estimation maintain broad applicability to many different types of microscopy. The algorithm is
described in detail in: 

> Furnival T, Leary R, Midgley PA. (2016). Denoising time-resolved  microscopy sequences with singular 
> value thresholding. *Manuscript in preparation.*

[![Build Status](https://travis-ci.org/tjof2/pgure-svt.svg?branch=master)](https://travis-ci.org/tjof2/pgure-svt)

---

## Contents

+ [Installation](#installation)
+ [Using PGURE-SVT](#using-pgure-svt)

## Installation

#### Dependencies

PGURE-SVT makes use of several 3rd-party libraries, which need to be installed first.

+ **[CMake](http://www.cmake.org)** (>=2.8)
+ **[LibTIFF](http://www.remotesensing.org/libtiff/)** 
+ **[Armadillo](http://arma.sourceforge.net)** (>=6.400)
+ **[NLopt](http://ab-initio.mit.edu/wiki/index.php/NLopt)** (>=2.4.2)

#### Compiling from source

To build PGURE-SVT, unpack the source and use the following commands

```bash
$ tar -xzf pgure-svt.tar.gz
$ cd pgure-svt
$ cmake .
$ make
```

This will generate an executable called `PGURE-SVT` in the `bin` directory.

## Using PGURE-SVT

For an in-depth explanation of the options, users are referred to the
paper describing the algorithm.

### Python

An example file, `demo.py`, is provided to show users how the algorithm can
be linked with the [HyperSpy](http://hyperspy.org) multi-dimensional data analysis toolbox,
which provides a number of useful features including:
- Data visualization
- Import from a number of microscopy formats

The basic workflow is:

```python
import pguresvt

# Initialize with default parameters
svt = pguresvt.SVT()

# Run the algorithm on the data X
# and get the denoised output Y
Y = SVT.denoise(X)
```

### Standalone

PGURE-SVT uses a simple command-line interface along with a separate parameter file.

```bash
$ ./PGURE-SVT param.svt
```

The parameter file allows the user to customize various options of the PGURE-SVT
algorithm. An example file, `param.svt` is provided, with short comments to explain 
the effects of each option.














