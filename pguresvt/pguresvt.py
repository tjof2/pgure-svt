# Copyright 2015-2020 Tom Furnival
#
# This file is part of PGURE-SVT.
#
# PGURE-SVT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PGURE-SVT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PGURE-SVT.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from ._pguresvt import pguresvt_16


class SVT:
    """
    Parameters
    ----------
    patchsize : integer
        The dimensions of the patch in pixels
        to form a Casorati matrix (default = 4)
    length : integer
        Length in frames of the block to form
        a Casorati matrix. Must be odd (default = 15)
    optimize : bool
        Whether to optimize PGURE or just denoise
        according to given threshold (default = True)
    threshold : float
        Threshold to use if not optimizing PGURE
        (default = 0.5)
    alpha : float
        Level of noise gain, if negative then
        estimated online (default = -1)
    mu : float
        Level of noise offset, if negative then
        estimated online (default = -1)
    sigma : float
        Level of Gaussian noise, if negative then
        estimated online (default = -1)
    arpssize : integer
        Size of neighbourhood for ARPS search
        Must be odd
        (default = 7 pixels)
    tol : float
        Tolerance of PGURE optimizers
        (default = 1E-7)

    """

    def __init__(
        self,
        trajLength=15,
        blockSize=4,
        blockOverlap=2,
        motionWindow=7,
        noiseMethod=4,
        maxIter=1000,
        nJobs=-1,
        randomSeed=-1,
        optPGURE=True,
        expWeighting=True,
        motionEstimation=True,
        lambdaEst=0.15,
        alphaEst=-1.0,
        muEst=-1.0,
        sigmaEs=-1.0,
        tol=1e-7,
    ):
        self.trajLength = trajLength
        self.blockSize = blockSize
        self.blockOverlap = blockOverlap
        self.motionWindow = motionWindow
        self.noiseMethod = noiseMethod
        self.maxIter = maxIter
        self.nJobs = nJobs
        self.randomSeed = randomSeed
        self.optPGURE = optPGURE
        self.expWeighting = expWeighting
        self.motionEstimation = motionEstimation
        self.lambdaEst = lambdaEst
        self.alphaEst = alphaEst
        self.muEst = muEst
        self.sigmaEst = sigmaEst
        self.tol = tol

        self.Y_ = None

    def denoise(self, X):
        """Denoise the data X

        Parameters
        ----------
        X : array [nx, ny, time]
            The image sequence to be denoised

        Returns
        -------
        Y : array [nx, ny, time]
            Returns the denoised sequence

        """
        # if self.overlap > self.patchsize:
        #     raise ValueError("Patch overlap should not be greater than patch size")
        # if self.arpssize % 2 == 0:
        #     raise ValueError("ARPS motion estimation window size should be odd")
        # if self.threshold < 0.0 or self.threshold > 1.0:
        #     raise ValueError("Threshold should be in range [0,1]")
        # if self.median % 2 == 0:
        #     raise ValueError("Median filter size should be odd")

        # if self.estimation:
        #     if X.shape[0] != X.shape[1]:
        #         raise ValueError(
        #             f"Quadtree noise estimation requires square images, got {X.shape}"
        #         )

        #     if not self._is_power_of_two(dims[0]):
        #         raise ValueError(
        #             "Quadtree noise estimation requires image dimensions 2^N"
        #         )

        res = pguresvt_16(
            input_images=X.astype(np.uint16),
            filtered_images=X,
            trajLength=self.trajLength,
            blockSize=self.blockSize,
            blockOverlap=self.blockOverlap,
            motionWindow=self.motionWindow,
            noiseMethod=self.noiseMethod,
            maxIter=self.maxIter,
            nJobs=self.nJobs,
            randomSeed=self.randomSeed,
            optPGURE=self.optPGURE,
            expWeighting=self.expWeighting,
            motionEstimation=self.motionEstimation,
            lambdaEst=self.lambdaEst,
            alphaEst=self.alphaEst,
            muEst=self.muEst,
            sigmaEst=self.sigmaEst,
            tol=self.tol,
        )
        self.Y_ = res[0]

        return self

    def _is_power_of_two(self, n):
        n = n / 2
        if n == 2:
            return True
        elif n > 2:
            return self._is_power_of_two(n)
        else:
            return False


def _addnoise(x, alpha, mu, sigma):
    """Add Poisson-Gaussian noise to the data x

    Parameters
    ----------
    x : float
        The original data

    alpha : float
        Level of noise gain

    mu : float
        Level of noise offset

    sigma : float
        Level of Gaussian noise

    Returns
    -------
    y : float
        The corrupted data

    """
    y = alpha * np.random.poisson(x / alpha) + mu + sigma * np.random.randn()
    return y


def PoissonGaussianNoiseGenerator(X, alpha=0.1, mu=0.1, sigma=0.1):
    """Add Poisson-Gaussian noise to the data X

    Parameters
    ----------
    X : array
        The data to be corrupted

    alpha : float
        Level of noise gain

    mu : float
        Level of noise offset

    sigma : float
        Level of Gaussian noise

    Returns
    -------
    Y : array
        The corrupted data

    """
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha should be in range [0,1]")

    addnoise = np.vectorize(_addnoise, otypes=[np.float])

    Xmax = np.amax(X)
    X = X / Xmax

    Y = addnoise(X, alpha, mu, sigma)
    Y = Y + np.abs(np.amin(Y))
    Y = Xmax * Y / np.amax(Y)

    return Y
