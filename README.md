# PGURE-SVT

[http://tjof2.github.io/pgure-svt](http://tjof2.github.io/pgure-svt)

PGURE-SVT is an algorithm designed to denoise image sequences acquired in microscopy. It exploits the correlations between
consecutive frames to form low-rank matrices, which are then recovered using a technique known as nuclear norm minimization.
An unbiased risk estimator for mixed Poisson-Gaussian noise is used to automate the selection of the regularization parameter,
while robust noise and motion estimation maintain broad applicability to many different types of microscopy. 

**[Download stable version (0.2.3)](https://github.com/tjof2/pgure-svt/archive/v0.2.3.tar.gz)**


PGURE-SVT is released free of charge under the GNU General Public License ([GPLv3](http://tjof2.github.io/pgure-svt/www.gnu.org/licenses/gpl-3.0.en.html)). If you use this code in a publication, please cite our work using this citation: 

```
@article{PGURESVT2016,
    title   = {Denoising time-resolved microscopy sequences with 
               singular value thresholding.},
    author  = {Furnival, Tom and Leary, Rowan K. and Midgley, Paul A.},
    year    = {Manuscript submitted}
}
```

[![Build Status](https://travis-ci.org/tjof2/pgure-svt.svg?branch=master)](https://travis-ci.org/tjof2/pgure-svt)
