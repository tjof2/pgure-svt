# Author: Tom Furnival
# License: GPLv3

import os
import numpy as np
import pytest

from pguresvt import SVT, mixed_noise_model


def _nsed(A, B):
    """Calculate the normalized Euclidean distance between two arrays."""
    A_m = A - A.mean()
    B_m = B - B.mean()

    return (
        0.5
        * np.linalg.norm(A_m - B_m) ** 2
        / (np.linalg.norm(A_m) ** 2 + np.linalg.norm(B_m) ** 2)
    )


class TestMixedNoiseModel:
    def setup_method(self, method):
        self.rng = np.random.RandomState(101)
        self.X = self.rng.uniform(low=0, high=255, size=(64, 64, 32))

    def test_default(self):
        Y = mixed_noise_model(self.X, random_state=self.rng)
        np.testing.assert_allclose(_nsed(self.X, Y), 0.3753506, rtol=1e-6)

    def test_alpha(self):
        Y = mixed_noise_model(self.X, alpha=1e-5, random_state=self.rng)
        np.testing.assert_allclose(_nsed(self.X, Y), 1.4986839425e-05, rtol=1e-6)

    def test_mu(self):
        Y = mixed_noise_model(self.X, alpha=1e-5, mu=0.1, random_state=self.rng)
        np.testing.assert_allclose(_nsed(self.X, Y), 1.4986839425e-05, rtol=1e-6)
        np.testing.assert_allclose(
            self.X.mean(), Y.mean() - 0.1 * self.X.max(), rtol=5e-5
        )

    def test_sigma(self):
        Y = mixed_noise_model(self.X, alpha=1e-5, sigma=0.1, random_state=self.rng)
        np.testing.assert_allclose(_nsed(self.X, Y), 0.02836968, rtol=1e-6)

    def test_error_alpha(self):
        with pytest.raises(ValueError, match="alpha should be in range"):
            _ = mixed_noise_model(self.X, alpha=-1.0)

    def test_error_sigma(self):
        with pytest.raises(ValueError, match="sigma should be"):
            _ = mixed_noise_model(self.X, sigma=-1.0)

    @pytest.mark.parametrize("rng", [None, 101, np.random.RandomState(101)])
    def test_random_state(self, rng):
        # Should not throw any errors
        _ = mixed_noise_model(self.X, random_state=rng)


class TestGaussianNoise:
    def setup_method(self, method):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.X = np.load(f"{cur_path}/data.npz")["a"]

        self.seed = 101
        self.rng = np.random.RandomState(self.seed)
        self.mu = 100.0
        self.sigma = 100.0

        self.Y = self.X + self.mu + self.sigma * self.rng.randn(*self.X.shape)
        self.Y[self.Y < 0.0] = 0.0
        self.Y = self.Y.astype(np.uint16)  # Temporary for now

    def test_single_threaded(self):
        s = SVT(n_jobs=1, random_seed=self.seed)
        s.denoise(self.Y)

        assert hasattr(s, "Y_")
        assert hasattr(s, "lambda1s_")
        assert hasattr(s, "noise_alphas_")
        assert hasattr(s, "noise_mus_")
        assert hasattr(s, "noise_sigmas_")

        assert _nsed(self.X, s.Y_) < 0.025

    def test_multi_threaded(self):
        s = SVT(random_seed=self.seed)
        s.denoise(self.Y)
        assert _nsed(self.X, s.Y_) < 0.025

    @pytest.mark.parametrize("opt", [True, False])
    def test_opt_lambda1(self, opt):
        s = SVT(lambda1=5.0, optimize_pgure=opt, random_seed=self.seed)
        s.denoise(np.asfortranarray(self.Y))
        assert _nsed(self.X, s.Y_) < 0.025

    def test_fortran_array(self):
        s = SVT(random_seed=self.seed)
        s.denoise(np.asfortranarray(self.Y))
        assert _nsed(self.X, s.Y_) < 0.025

    def test_known_noise(self):
        s = SVT(noise_mu=self.mu, noise_sigma=self.sigma, random_seed=self.seed)
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

    def test_error_dtype(self):
        with pytest.raises(TypeError, match="Invalid dtype"):
            s = SVT()
            s.denoise((self.X == 0))
