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
        trajectory_length=15,
        patch_size=4,
        patch_overlap=2,
        arps_window=7,
        median_filter=5,
        noise_method=4,
        max_iter=1000,
        n_jobs=-1,
        random_seed=-1,
        optimize_pgure=True,
        exponential_weighting=True,
        motion_estimation=True,
        lambda1=0.0,
        noise_alpha=-1.0,
        noise_mu=-1.0,
        noise_sigma=-1.0,
        tol=1e-7,
    ):
        self.trajectory_length = trajectory_length
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.arps_window = arps_window
        self.median_filter = median_filter
        self.noise_method = noise_method
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.optimize_pgure = optimize_pgure
        self.exponential_weighting = exponential_weighting
        self.motion_estimation = motion_estimation
        self.lambda1 = lambda1
        self.noise_alpha = noise_alpha
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.tol = tol

        self.Y_ = None

    def _is_power_of_two(self, n):
        n = n / 2
        if n == 2:
            return True
        elif n > 2:
            return self._is_power_of_two(n)
        else:
            return False

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

        if not X.flags.f_contiguous:
            X = np.asfortranarray(X, dtype=np.uint16)

        res = pguresvt_16(
            input_images=X,
            trajectory_length=self.trajectory_length,
            patch_size=self.patch_size,
            patch_overlap=self.patch_overlap,
            arps_window=self.arps_window,
            median_filter=self.median_filter,
            noise_method=self.noise_method,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            random_seed=self.random_seed,
            optimize_pgure=self.optimize_pgure,
            exponential_weighting=self.exponential_weighting,
            motion_estimation=self.motion_estimation,
            lambda1=self.lambda1,
            noise_alpha=self.noise_alpha,
            noise_mu=self.noise_mu,
            noise_sigma=self.noise_sigma,
            tol=self.tol,
        )
        self.Y_ = res[0]

        return self
