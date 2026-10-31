import pytest

from octoqai.stats import paired_bootstrap_diff, pass_at_k, pass_hat_k, wilson_interval

pytestmark = [pytest.mark.unit, pytest.mark.regression]


@pytest.mark.smoke
def test_wilson_known_value():
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)


def test_wilson_all_pass_upper_is_one():
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)


def test_wilson_zero_samples_is_uninformative():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(5, 3)


def test_bootstrap_is_deterministic_and_detects_clear_win():
    a, b = [1.0] * 30, [0.2] * 30
    assert paired_bootstrap_diff(a, b) == paired_bootstrap_diff(a, b)
    mean, lo, hi = paired_bootstrap_diff(a, b)
    assert mean == pytest.approx(0.8) and lo > 0


def test_bootstrap_no_difference_straddles_zero():
    a = [1, 0] * 15
    b = [0, 1] * 15
    _, lo, hi = paired_bootstrap_diff(a, b)
    assert lo < 0 < hi


def test_bootstrap_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0], [1.0, 0.0])


def test_bootstrap_empty_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([], [])


def test_pass_k():
    assert pass_at_k([False, True]) and not pass_hat_k([False, True])
    assert pass_hat_k([True, True])
