# Author: Tom Furnival
# License: GPLv3

import numpy as np

from ._pguresvt import pguresvt_16


def _is_power_of_two(n):
    """Checks if n is a power of 2"""
    return (n & (n - 1) == 0) and n != 0


class SVT:
    """Singular value thresholding for denoising image sequences.

    PGURE-SVT is an algorithm designed to denoise image sequences
    acquired in microscopy. It exploits the correlations between
    consecutive frames to form low-rank matrices, which are then
    recovered using a technique known as nuclear norm minimization.
    An unbiased risk estimator for mixed Poisson-Gaussian noise is
    used to automate the selection of the regularization parameter,
    while robust noise and motion estimation maintain broad applicability
    to many different types of microscopy. See [Furnival2017]_.

    Parameters
    ----------
    trajectory_length : int, default=15
        Length in frames of the block to form
        a Casorati matrix. Must be odd.
    patch_size : int, default=4
        The dimensions of the patch in pixels
        to form a Casorati matrix.
    patch_overlap : int, default=1
        The dimensions of the patch in pixels
        to form a Casorati matrix.
    motion_window : int, default=7
        Size of neighbourhood in pixels for ARPS
        motion estimation search. Must be odd.
    motion_filter : int or None, default=5
        Size of median filter in pixels used to
        improve motion estimation search. If ``None``,
        no median filtering is performed.
    optimize_pgure : bool, default=True
        Whether to optimize PGURE or just denoise
        according to given threshold.
    threshold : float or None, default=None
        Threshold to use if not optimizing PGURE.
        Ignored if ``optimize_pgure=True``.
    noise_alpha : float or None, default=None
        Level of noise gain. If None, then parameter is
        estimated online. Ignored if ``optimize_pgure=False``.
    noise_mu : float or None, default=None
        Level of noise offset. If None, then parameter is
        estimated online. Ignored if ``optimize_pgure=False``.
    noise_sigma : float or None, default=None
        Level of Gaussian noise. If None, then parameter is
        estimated online. Ignored if ``optimize_pgure=False``.
    tol : float, default=1e-7
        Stopping tolerance of PGURE optimization algorithm.
        Ignored if ``optimize_pgure=False``.
    max_iter : int, default=1000
        Maximum iterations of PGURE optimization algorithm.
        Ignored if ``optimize_pgure=False``.
    n_jobs : int or None, default=None
        The number of threads to use. A value of ``None`` means
        using a single thread, while -1 means using all
        threads dependent on the available hardware.
    random_seed : int or None, default=None
        Random seed used when optimizing PGURE.
        Ignored if ``optimize_pgure=False``.

    Attributes
    ----------
    Y_ : np.ndarray
        Denoised image sequence.

    References
    ----------
    .. [Furnival2017] T. Furnival, R. K. Leary and P. A. Midgley, "Denoising
                      time-resolved microscopy sequences with singular value
                      thresholding", Ultramicroscopy, vol. 178, pp. 112–124,
                      2017. DOI:10.1016/j.ultramic.2016.05.005

    """

    def __init__(
        self,
        trajectory_length=15,
        patch_size=4,
        patch_overlap=1,
        motion_estimation=True,
        motion_window=7,
        motion_filter=5,
        optimize_pgure=True,
        lambda1=0.0,
        exponential_weighting=True,
        noise_method=4,
        noise_alpha=-1.0,
        noise_mu=-1.0,
        noise_sigma=-1.0,
        tol=1e-7,
        max_iter=1000,
        n_jobs=None,
        random_seed=None,
    ):
        self.trajectory_length = trajectory_length
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.motion_estimation = motion_estimation
        self.motion_window = motion_window
        self.motion_filter = motion_filter
        self.optimize_pgure = optimize_pgure
        self.lambda1 = lambda1
        self.exponential_weighting = exponential_weighting
        self.noise_method = noise_method
        self.noise_alpha = noise_alpha
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.tol = tol
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_seed = random_seed

        self.Y_ = None

    def _check_arguments(self, X_shape):
        """Sanity-checking of arguments before calling C++ code."""
        # C++ uses numerical values instead of None for defaults
        self.motion_filter_ = -1 if self.motion_filter is None else self.motion_filter
        self.n_jobs_ = 0 if self.n_jobs is None else self.n_jobs
        self.random_seed_ = -1 if self.random_seed is None else self.random_seed

        # Check arguments
        if self.patch_overlap > self.patch_size:
            raise ValueError(
                f"Invalid patch_overlap parameter: got {self.patch_overlap}, "
                f"should not be greater than patch_size ({self.patch_size})"
            )

        if self.motion_estimation:
            if self.motion_window < 2:
                raise ValueError(
                    f"Invalid motion_window parameter: got {self.motion_window}, "
                    "should be greater than 1 pixel"
                )

            if not isinstance(self.motion_filter_, int):
                raise ValueError(
                    f"Invalid motion_filter parameter: got {type(self.motion_filter)}, "
                    "should be an integer number of pixels or None."
                )

        if not self.optimize_pgure and (self.lambda1 < 0.0 or self.lambda1 > 1.0):
            raise ValueError(
                f"Invalid lambda1 parameter: got {self.lambda1}, "
                "should be a float in range [0, 1]"
            )

        if any(v < 0.0 for v in [self.noise_alpha, self.noise_mu, self.noise_sigma]):
            if X_shape[0] != X_shape[1]:
                raise ValueError(
                    f"Quadtree noise estimation requires square images, got {X_shape}"
                )

            if not _is_power_of_two(X_shape[0]):
                raise ValueError(
                    "Quadtree noise estimation requires image dimensions 2^N"
                )

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
        self._check_arguments(X.shape)

        X_dtype = getattr(X, "dtype", None)

        if not X.flags.f_contiguous:
            X = np.asfortranarray(X, dtype=np.uint16)
        elif X_dtype != "uint16":
            X = X.astype(np.uint16)

        res = pguresvt_16(
            input_images=X,
            trajectory_length=self.trajectory_length,
            patch_size=self.patch_size,
            patch_overlap=self.patch_overlap,
            motion_estimation=self.motion_estimation,
            motion_window=self.motion_window,
            motion_filter=self.motion_filter_,
            optimize_pgure=self.optimize_pgure,
            lambda1=self.lambda1,
            exponential_weighting=self.exponential_weighting,
            noise_method=self.noise_method,
            noise_alpha=self.noise_alpha,
            noise_mu=self.noise_mu,
            noise_sigma=self.noise_sigma,
            tol=self.tol,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs_,
            random_seed=self.random_seed_,
        )
        self.Y_ = res[0]

        return self
