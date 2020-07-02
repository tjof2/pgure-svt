# Author: Tom Furnival
# License: GPLv3

import os
import numpy as np
import pytest

from pguresvt import SVT


def _nsed(A, B):
    """Calculate the normalized Euclidean distance between two arrays."""
    A_m = A - A.mean()
    B_m = B - B.mean()

    return (
        0.5
        * np.linalg.norm(A_m - B_m) ** 2
        / (np.linalg.norm(A_m) ** 2 + np.linalg.norm(B_m) ** 2)
    )


class TestGaussianNoise:
    def setup_method(self, method):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.X = np.load(f"{cur_path}/data.npz")["a"]

        self.seed = 123
        self.rng = np.random.RandomState(self.seed)
        self.mu = 100.0
        self.sigma = 100.0

        self.Y = self.X + self.mu + self.sigma * self.rng.randn(*self.X.shape)
        self.Y[self.Y < 0.0] = 0.0

    def test_default_single_threaded(self):
        s = SVT(n_jobs=1, random_seed=self.seed)
        s.denoise(self.Y)
        assert _nsed(self.X, s.Y_) < 0.025

    def test_default_multi_threaded(self):
        s = SVT(n_jobs=-1, random_seed=self.seed)
        s.denoise(self.Y)
        assert _nsed(self.X, s.Y_) < 0.025

    def test_default_known_noise(self):
        s = SVT(
            noise_mu=self.mu, noise_sigma=self.sigma, n_jobs=-1, random_seed=self.seed
        )
        s.denoise(self.Y)
        assert _nsed(self.X, s.Y_) < 0.3


class TestErrors:
    def setup_method(self, method):
        self.X = np.ones((32, 32, 16))

    def test_negative_data(self):
        with pytest.raises(ValueError, match="Negative values found in data"):
            s = SVT()
            self.X[0, 0, 0] = -1
            s.denoise(self.X)

    def test_error_patch_overlap(self):
        with pytest.raises(ValueError, match="Invalid patch_overlap parameter"):
            s = SVT(patch_size=10, patch_overlap=11)
            s.denoise(self.X)

    def test_error_motion_window(self):
        with pytest.raises(ValueError, match="Invalid motion_window parameter"):
            s = SVT(motion_estimation=True, motion_window=1)
            s.denoise(self.X)

    def test_error_motion_filter(self):
        with pytest.raises(ValueError, match="Invalid motion_filter parameter"):
            s = SVT(motion_estimation=True, motion_filter=0.5)
            s.denoise(self.X)

    def test_error_lambda1(self):
        with pytest.raises(ValueError, match="Invalid lambda1 parameter"):
            s = SVT(optimize_pgure=False, lambda1=-1.0)
            s.denoise(self.X)

    def test_error_non_square(self):
        m, _, _ = self.X.shape
        with pytest.raises(ValueError, match="requires square images"):
            s = SVT()
            s.denoise(self.X[: m // 2, :, :])

    def test_error_non_power_of_two(self):
        m, n, _ = self.X.shape  # Should be 32x32x16
        assert m == n
        with pytest.raises(ValueError, match="requires image dimensions 2\\^N"):
            s = SVT()
            s.denoise(self.X[: m - 1, : n - 1, :])
