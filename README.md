# PGURE-SVT

### [http://tjof2.github.io/pgure-svt](http://tjof2.github.io/pgure-svt)

PGURE-SVT is an algorithm designed to denoise image sequences acquired in microscopy. It exploits the correlations between
consecutive frames to form low-rank matrices, which are then recovered using a technique known as nuclear norm minimization.
An unbiased risk estimator for mixed Poisson-Gaussian noise is used to automate the selection of the regularization parameter,
while robust noise and motion estimation maintain broad applicability to many different types of microscopy.

More information on the algorithm can be found in our paper, DOI:[10.1016/j.ultramic.2016.05.005](http://dx.doi.org/10.1016/j.ultramic.2016.05.005).

### Citing

If you use this code in a publication, please cite our work:

```
@article{PGURESVT2016,
    title   = {Denoising time-resolved microscopy sequences with
               singular value thresholding.},
    author  = {Furnival, Tom and Leary, Rowan K. and Midgley, Paul A.},
    journal = {Ultramicroscopy},
    doi     = {10.1016/j.ultramic.2016.05.005},
    year    = {Article in press}
}
```

### Download

PGURE-SVT has been tested on Ubuntu 12.04, 13.04 and 14.04. The Windows binaries
have been tested on 64-bit versions of Windows 7 and Windows 10.

+ **[Download source](https://github.com/tjof2/pgure-svt/archive/v0.3.1.tar.gz)**
+ **[Download 64-bit Windows release](https://github.com/tjof2/pgure-svt/releases/download/v0.3.1/PGURE-SVT_Win64.zip)**

_Note_: You may need to install the [Microsoft Visual C++ 2015 redistributable package](https://www.microsoft.com/en-gb/download/details.aspx?id=48145) before running PGURE-SVT.

[![Build Status](https://travis-ci.org/tjof2/pgure-svt.svg?branch=master)](https://travis-ci.org/tjof2/pgure-svt)

<small>
Copyright (C) 2015-2016 Tom Furnival.

PGURE-SVT is released free of charge under the GNU General Public License ([GPLv3](http://tjof2.github.io/pgure-svt/www.gnu.org/licenses/gpl-3.0.en.html)).
</small>
