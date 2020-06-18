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
            motion_filter=1,
            n_jobs=1,
            random_seed=self.seed,
        )
        s.denoise(self.X.copy())

        assert _hash_ndarray(s.Y_, 8) == "5c3a8c72"
