# PGURE-SVT

### [http://tjof2.github.io/pgure-svt](http://tjof2.github.io/pgure-svt)

PGURE-SVT is an algorithm designed to denoise image sequences acquired in microscopy. It exploits the correlations between
consecutive frames to form low-rank matrices, which are then recovered using a technique known as nuclear norm minimization.
An unbiased risk estimator for mixed Poisson-Gaussian noise is used to automate the selection of the regularization parameter,
while robust noise and motion estimation maintain broad applicability to many different types of microscopy.

More information on the algorithm can be found [in our paper](http://dx.doi.org/10.1016/j.ultramic.2016.05.005) in *Ultramicroscopy*.

### Citing

If you use this code in a publication, please cite our work:

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

### Download

#### Linux/Mac

PGURE-SVT has been tested on Ubuntu 12.04, 13.04 and 14.04.
For Mac users, you may need to use the GCC compiler (v4.9+) rather than the default.
Compilation instructions can be found [here](http://tjof2.github.io/pgure-svt/install.html).

**[Download source](https://github.com/tjof2/pgure-svt/archive/v0.3.2.tar.gz)**

#### Windows

The Windows binaries have been tested on 64-bit versions of Windows 7 and Windows 10. You may need
to install the [Microsoft Visual C++ 2015 redistributable package](https://www.microsoft.com/en-gb/download/details.aspx?id=48145)
before running PGURE-SVT.

**[Download 64-bit Windows release](https://github.com/tjof2/pgure-svt/releases/download/v0.3.3/PGURE-SVT_Win64.zip)**

[![Build Status](https://travis-ci.org/tjof2/pgure-svt.svg?branch=master)](https://travis-ci.org/tjof2/pgure-svt)

Copyright (C) 2015-2019 Tom Furnival.
PGURE-SVT is released free of charge under the GNU General Public License ([GPLv3](http://tjof2.github.io/pgure-svt/www.gnu.org/licenses/gpl-3.0.en.html)).
