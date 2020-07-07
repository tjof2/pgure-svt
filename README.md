[![Build Status](https://travis-ci.org/tjof2/pgure-svt.svg?branch=master)](https://travis-ci.org/tjof2/pgure-svt)
[![Coverage Status](https://coveralls.io/repos/github/tjof2/pgure-svt/badge.svg?branch=master)](https://coveralls.io/github/tjof2/pgure-svt?branch=master)
[![DOI](https://zenodo.org/badge/48366354.svg)](https://zenodo.org/badge/latestdoi/48366354)

# PGURE-SVT

**Documentation: https://tjof2.github.io/pgure-svt/**

PGURE-SVT (Poisson-Gaussian Unbiased Risk Estimator - Singular Value Thresholding) is an algorithm designed to denoise image sequences acquired in microscopy. It exploits the correlations between consecutive frames to form low-rank matrices, which are then recovered using a technique known as nuclear norm minimization. An unbiased risk estimator for mixed Poisson-Gaussian noise is used to automate the selection of the regularization parameter, while robust noise and motion estimation maintain broad applicability to many different types of microscopy.

More information on the algorithm can be found [in our paper](http://dx.doi.org/10.1016/j.ultramic.2016.05.005) in *Ultramicroscopy*.

If you use this code in a publication, please cite our work:

> T. Furnival, R. K. Leary and P. A. Midgley, "Denoising time-resolved microscopy sequences with singular value thresholding", *Ultramicroscopy*, vol. 178, pp. 112–124, 2017. DOI:[10.1016/j.ultramic.2016.05.005](http://dx.doi.org/10.1016/j.ultramic.2016.05.005)

PGURE-SVT is released free of charge under the GNU General Public License ([GPLv3](http://www.gnu.org/licenses/gpl-3.0.en.html)).

Copyright (C) 2015-2020 Tom Furnival.
