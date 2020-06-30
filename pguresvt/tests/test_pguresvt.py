# Author: Tom Furnival
# License: GPLv3

import hashlib

import os
import numpy as np
import pytest

from pguresvt import SVT


def _hash_ndarray(arr, n_char=-1):
    """Simple function to hash a np.ndarray object."""
    return hashlib.sha256(arr.data.tobytes()).hexdigest()[:n_char]


class TestSVT:
    def setup_method(self, method):
        cur_path = os.path.dirname(os.path.realpath(__file__))

        self.seed = 123
        self.X = np.load(f"{cur_path}/data.npz")["a"]

    def test_default(self):
        s = SVT(
            patch_size=16,
            trajectory_length=15,
            patch_overlap=3,
            optimize_pgure=False,
            lambda1=0.15,
            noise_alpha=0.1,
            noise_mu=0.1,
            noise_sigma=0.1,
            motion_estimation=True,
            motion_window=7,
            motion_filter=3,
            n_jobs=1,
            random_seed=self.seed,
        )
        s.denoise(self.X)

        assert _hash_ndarray(s.Y_, 8) == "5345bc0a"

    def test_no_motion(self):
        s = SVT(
            patch_size=16,
            trajectory_length=15,
            patch_overlap=3,
            optimize_pgure=False,
            lambda1=0.15,
            noise_alpha=0.1,
            noise_mu=0.1,
            noise_sigma=0.1,
            motion_estimation=False,
            motion_window=7,
            motion_filter=1,
            n_jobs=1,
            random_seed=self.seed,
        )
        s.denoise(self.X)

        assert _hash_ndarray(s.Y_, 8) == "0115ba23"

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
