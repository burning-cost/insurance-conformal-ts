"""
Tests for nonconformity.py

Verifies each score:
- Returns correct shapes
- score and inverse are consistent (round-trip)
- Handles edge cases (zeros, large values)
- Protocol compliance
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_conformal_ts.nonconformity import (
    AbsoluteResidualScore,
    ExposureAdjustedScore,
    LocallyWeightedScore,
    NegBinomPearsonScore,
    NonConformityScore,
    PoissonPearsonScore,
)


RNG = np.random.default_rng(42)


class TestProtocolCompliance:
    """All scores must satisfy the NonConformityScore Protocol."""

    @pytest.mark.parametrize("cls,kwargs", [
        (AbsoluteResidualScore, {}),
        (PoissonPearsonScore, {}),
        (NegBinomPearsonScore, {"phi": 5.0}),
        (ExposureAdjustedScore, {}),
        (LocallyWeightedScore, {}),
    ])
    def test_is_conformity_score(self, cls, kwargs):
        instance = cls(**kwargs)
        assert isinstance(instance, NonConformityScore)

    @pytest.mark.parametrize("cls,kwargs", [
        (AbsoluteResidualScore, {}),
        (PoissonPearsonScore, {}),
        (NegBinomPearsonScore, {"phi": 5.0}),
        (ExposureAdjustedScore, {}),
        (LocallyWeightedScore, {}),
    ])
    def test_has_score_method(self, cls, kwargs):
        instance = cls(**kwargs)
        assert callable(getattr(instance, "score", None))

    @pytest.mark.parametrize("cls,kwargs", [
        (AbsoluteResidualScore, {}),
        (PoissonPearsonScore, {}),
        (NegBinomPearsonScore, {"phi": 5.0}),
        (ExposureAdjustedScore, {}),
        (LocallyWeightedScore, {}),
    ])
    def test_has_inverse_method(self, cls, kwargs):
        instance = cls(**kwargs)
        assert callable(getattr(instance, "inverse", None))


class TestAbsoluteResidualScore:
    def setup_method(self):
        self.score = AbsoluteResidualScore()
        self.y = np.array([5.0, 10.0, 15.0, 8.0])
        self.y_hat = np.array([4.0, 12.0, 14.0, 9.0])

    def test_score_shape(self):
        s = self.score.score(self.y, self.y_hat)
        assert s.shape == (4,)

    def test_score_nonnegative(self):
        s = self.score.score(self.y, self.y_hat)
        assert np.all(s >= 0)

    def test_score_values(self):
        s = self.score.score(self.y, self.y_hat)
        expected = np.array([1.0, 2.0, 1.0, 1.0])
        np.testing.assert_allclose(s, expected)

    def test_inverse_upper(self):
        s = np.array([1.5])
        y_hat = np.array([5.0])
        upper = self.score.inverse(s, y_hat, upper=True)
        np.testing.assert_allclose(upper, np.array([6.5]))

    def test_inverse_lower(self):
        s = np.array([1.5])
        y_hat = np.array([5.0])
        lower = self.score.inverse(s, y_hat, upper=False)
        np.testing.assert_allclose(lower, np.array([3.5]))

    def test_inverse_lower_clipped_at_zero(self):
        """Default clip_lower=True: lower bound cannot go negative."""
        s = np.array([10.0])
        y_hat = np.array([3.0])
        lower = self.score.inverse(s, y_hat, upper=False)
        assert lower[0] == 0.0

    def test_inverse_lower_no_clip(self):
        """clip_lower=False: lower bound can be negative."""
        score = AbsoluteResidualScore(clip_lower=False)
        s = np.array([10.0])
        y_hat = np.array([3.0])
        lower = score.inverse(s, y_hat, upper=False)
        assert lower[0] == -7.0

    def test_round_trip(self):
        """score(y, y_hat) -> s; inverse(s, y_hat) == y for upper bound."""
        y = np.array([7.0])
        y_hat = np.array([5.0])
        s = self.score.score(y, y_hat)
        recovered = self.score.inverse(s, y_hat, upper=True)
        np.testing.assert_allclose(recovered, y)

    def test_zero_residual(self):
        s = self.score.score(np.array([5.0]), np.array([5.0]))
        assert s[0] == 0.0

    def test_large_arrays(self):
        n = 10_000
        y = RNG.poisson(10, size=n).astype(float)
        y_hat = np.full(n, 10.0)
        s = self.score.score(y, y_hat)
        assert s.shape == (n,)
        assert np.all(s >= 0)


class TestPoissonPearsonScore:
    def setup_method(self):
        self.score = PoissonPearsonScore()

    def test_score_formula(self):
        y = np.array([12.0])
        mu = np.array([10.0])
        s = self.score.score(y, mu)
        expected = (12.0 - 10.0) / np.sqrt(10.0)
        np.testing.assert_allclose(s, [expected])

    def test_score_shape(self):
        y = np.arange(1, 11, dtype=float)
        mu = np.full(10, 5.0)
        s = self.score.score(y, mu)
        assert s.shape == (10,)

    def test_zero_mean_clipped(self):
        """Near-zero mu should not cause division by zero."""
        y = np.array([0.0])
        mu = np.array([0.0])
        s = self.score.score(y, mu)
        assert np.isfinite(s[0])

    def test_inverse_round_trip(self):
        mu = np.array([8.0])
        y = np.array([11.0])
        s = self.score.score(y, mu)
        recovered = self.score.inverse(s, mu)
        np.testing.assert_allclose(recovered, y, rtol=1e-10)

    def test_inverse_always_above_mu(self):
        """Upper bound from positive score should exceed mu."""
        mu = np.array([5.0, 10.0, 20.0])
        s = np.array([1.5, 2.0, 1.0])
        upper = self.score.inverse(s, mu)
        assert np.all(upper > mu)

    def test_min_mu_validation(self):
        with pytest.raises(ValueError):
            PoissonPearsonScore(min_mu=0)

    def test_min_mu_validation_negative(self):
        with pytest.raises(ValueError):
            PoissonPearsonScore(min_mu=-1.0)


class TestNegBinomPearsonScore:
    def setup_method(self):
        self.score_nb2 = NegBinomPearsonScore(phi=5.0, parameterisation="NB2")
        self.score_nb1 = NegBinomPearsonScore(phi=5.0, parameterisation="NB1")

    def test_nb2_variance_formula(self):
        y = np.array([15.0])
        mu = np.array([10.0])
        phi = 5.0
        var = mu + mu**2 / phi  # = 10 + 20 = 30
        expected = (15.0 - 10.0) / np.sqrt(30.0)
        s = self.score_nb2.score(y, mu)
        np.testing.assert_allclose(s, [expected], rtol=1e-10)

    def test_nb1_variance_formula(self):
        y = np.array([15.0])
        mu = np.array([10.0])
        phi = 5.0
        var = mu + mu / phi  # = 10 + 2 = 12
        expected = (15.0 - 10.0) / np.sqrt(12.0)
        s = self.score_nb1.score(y, mu)
        np.testing.assert_allclose(s, [expected], rtol=1e-10)

    def test_invalid_parameterisation(self):
        with pytest.raises(ValueError):
            NegBinomPearsonScore(phi=1.0, parameterisation="NB3")

    def test_phi_required_at_score_time(self):
        score = NegBinomPearsonScore()  # no phi at construction
        with pytest.raises(ValueError):
            score.score(np.array([5.0]), np.array([5.0]))

    def test_phi_override_at_score_time(self):
        score = NegBinomPearsonScore(phi=1.0)
        y = np.array([12.0])
        mu = np.array([10.0])
        s1 = score.score(y, mu, phi=1.0)
        s2 = score.score(y, mu, phi=10.0)
        assert s1[0] != s2[0]

    def test_inverse_round_trip_nb2(self):
        mu = np.array([10.0])
        y = np.array([14.0])
        s = self.score_nb2.score(y, mu)
        recovered = self.score_nb2.inverse(s, mu)
        np.testing.assert_allclose(recovered, y, rtol=1e-10)

    def test_nb2_wider_than_poisson(self):
        """NB2 has larger variance than Poisson for same mu, so upper bound is higher."""
        mu = np.array([10.0])
        q = np.array([1.96])
        nb_upper = self.score_nb2.inverse(q, mu)
        poisson_score = PoissonPearsonScore()
        p_upper = poisson_score.inverse(q, mu)
        assert nb_upper[0] > p_upper[0]


class TestExposureAdjustedScore:
    def setup_method(self):
        self.score = ExposureAdjustedScore()

    def test_score_formula(self):
        y = np.array([30.0])
        rate = np.array([0.15])
        exposure = np.array([200.0])
        s = self.score.score(y, rate, exposure=exposure)
        expected = 30.0 / 200.0 - 0.15  # = 0.15 - 0.15 = 0
        np.testing.assert_allclose(s, [expected], atol=1e-10)

    def test_no_exposure_uses_ones(self):
        y = np.array([5.0])
        rate = np.array([3.0])
        s = self.score.score(y, rate)
        expected = 5.0 - 3.0  # = 2.0
        np.testing.assert_allclose(s, [expected])

    def test_inverse_upper(self):
        rate = np.array([0.15])
        exposure = np.array([200.0])
        s = np.array([0.05])
        upper = self.score.inverse(s, rate, exposure=exposure, upper=True)
        expected = 200.0 * (0.15 + 0.05)  # = 40.0
        np.testing.assert_allclose(upper, [expected])

    def test_inverse_lower_clipped(self):
        rate = np.array([0.01])
        exposure = np.array([10.0])
        s = np.array([1.0])  # rate - s would be negative
        lower = self.score.inverse(s, rate, exposure=exposure, upper=False)
        assert lower[0] == 0.0

    def test_min_exposure_floor(self):
        """Exposure below min_exposure should be floored."""
        score = ExposureAdjustedScore(min_exposure=10.0)
        y = np.array([1.0])
        rate = np.array([0.1])
        exposure = np.array([0.0])  # below floor
        # Should not raise and should use floor
        s = score.score(y, rate, exposure=exposure)
        expected = 1.0 / 10.0 - 0.1  # = 0.0
        np.testing.assert_allclose(s, [expected], atol=1e-10)

    def test_shape_preservation(self):
        n = 50
        y = np.ones(n) * 10
        rate = np.ones(n) * 0.1
        exposure = np.ones(n) * 100
        s = self.score.score(y, rate, exposure=exposure)
        assert s.shape == (n,)


class TestLocallyWeightedScore:
    def setup_method(self):
        self.score = LocallyWeightedScore()

    def test_score_formula(self):
        y = np.array([12.0])
        mu = np.array([10.0])
        sigma = np.array([2.0])
        s = self.score.score(y, mu, sigma_hat=sigma)
        np.testing.assert_allclose(s, [1.0])

    def test_no_sigma_reduces_to_absolute(self):
        y = np.array([12.0])
        mu = np.array([10.0])
        s_lw = self.score.score(y, mu, sigma_hat=None)
        abs_score = AbsoluteResidualScore(clip_lower=False)
        s_abs = abs_score.score(y, mu)
        np.testing.assert_allclose(s_lw, s_abs)

    def test_inverse_upper(self):
        mu = np.array([10.0])
        sigma = np.array([2.0])
        q = np.array([1.96])
        upper = self.score.inverse(q, mu, sigma_hat=sigma, upper=True)
        expected = 10.0 + 1.96 * 2.0
        np.testing.assert_allclose(upper, [expected], rtol=1e-10)

    def test_inverse_round_trip(self):
        mu = np.array([50.0])
        sigma = np.array([5.0])
        y = np.array([57.0])
        s = self.score.score(y, mu, sigma_hat=sigma)
        recovered = self.score.inverse(s, mu, sigma_hat=sigma, upper=True)
        np.testing.assert_allclose(recovered, y, rtol=1e-10)

    def test_min_sigma_floor(self):
        """sigma=0 should be handled gracefully."""
        y = np.array([5.0])
        mu = np.array([5.0])
        sigma = np.array([0.0])
        s = self.score.score(y, mu, sigma_hat=sigma)
        assert np.isfinite(s[0])
