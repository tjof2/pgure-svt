# PGURE-SVT

PGURE-SVT is an algorithm designed to denoise image sequences acquired in microscopy. It exploits the correlations between
consecutive frames to form low-rank matrices, which are then recovered using a technique known as nuclear norm minimization.
An unbiased risk estimator for mixed Poisson-Gaussian noise is used to automate the selection of the regularization parameter, while
robust noise and motion estimation maintain broad applicability to many different types of microscopy. The algorithm is
described in detail in: 

> Furnival T, Leary R, Midgley PA. (2015). Denoising time-resolved  microscopy sequences with singular 
> value thresholding. *Manuscript in preparation.*

[![Build Status](https://travis-ci.org/tjof2/pgure-svt.svg?branch=master)](https://travis-ci.org/tjof2/pgure-svt)

---

## Contents

+ [Installation](#installation)
+ [Using PGURE-SVT](#using-pgure-svt)

## Installation

#### Dependencies

PGURE-SVT makes use of several 3rd-party libraries, which need to be installed first.

+ **[CMake](http://www.cmake.org)** 

   CMake is used to configure the build before compilation.

+ **[LibTIFF](http://www.remotesensing.org/libtiff/)** 

   PGURE-SVT currently only supports TIFF image stacks.

+ **[Armadillo](http://arma.sourceforge.net)**

   Armadillo is a C++ linear algebra library, and is used extensively in PGURE-SVT. The latest version (6.500) is recommended.
   
+ **[NLopt](http://ab-initio.mit.edu/wiki/index.php/NLopt)** 

   NLopt is a non-linear optimization library, implementing several different optimization algorithms.
   The latest version (2.4.2) is recommended.

#### Compiling from source

To build PGURE-SVT, unpack the source and `cd` into the unpacked directory:

```bash
$ tar -xzf pgure-svt.tar.gz
$ cd pgure-svt
```

The next step is to configure the build, and then finally compile it. This will generate 
an executable file in the `bin/` directory. 

```bash
$ cmake .
$ make
```

To run PGURE-SVT from any directory, add the following line to your `.bashrc` file including
the full path to the `bin/` directory:

```bash
$ echo alias PGURE-SVT='/path/to/directory/PGURE-SVT' >> ~/.bashrc
$ source ~/.bashrc
```

You can now run the program by typing `PGURE-SVT` in the terminal.

## Using PGURE-SVT

PGURE-SVT uses a simple command-line interface along with a separate parameter file.

```bash
$ PGURE-SVT param.svt
```

The parameter file allows the user to customize various options of the PGURE-SVT
algorithm. An example is provided, with short comments to explain the effects of
each option. For an in-depth explanation of the options, users are referred to the
paper describing the algorithm.














