# PGURE-SVT

[http://tjof2.github.io/pgure-svt](http://tjof2.github.io/pgure-svt)

PGURE-SVT is an algorithm designed to denoise image sequences acquired in microscopy. It exploits the correlations between
consecutive frames to form low-rank matrices, which are then recovered using a technique known as nuclear norm minimization.
An unbiased risk estimator for mixed Poisson-Gaussian noise is used to automate the selection of the regularization parameter,
while robust noise and motion estimation maintain broad applicability to many different types of microscopy. The algorithm is
described in detail in:

> T Furnival, R Leary, PA Midgley. (2016). Denoising time-resolved  microscopy sequences with singular 
> value thresholding. *Manuscript submitted.*
