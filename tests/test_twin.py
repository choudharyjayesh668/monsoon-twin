import datetime as dt

import numpy as np
import pytest

import twin

kit = pytest.importorskip("monsoon_r1")
from monsoon_r1 import contract, stats  # noqa: E402


def _days(year):
    d = dt.date(year, 1, 1)
    while d.year == year:
        yield d
        d += dt.timedelta(days=1)


def test_clim_slot_matches_the_kit_calendar_in_a_normal_year():
    for d in _days(2023):
        s, c = contract.doy_sincos(d.timetuple().tm_yday)
        assert twin.clim_slot(s, c) == stats._doy_index(d), d


def test_clim_slot_is_at_most_one_day_off_in_a_leap_year():
    for d in _days(2024):
        s, c = contract.doy_sincos(d.timetuple().tm_yday)
        gap = abs(twin.clim_slot(s, c) - stats._doy_index(d))
        assert min(gap, 366 - gap) <= 1, d


def test_clim_slot_counts_forecast_steps_forward_across_new_year():
    s, c = contract.doy_sincos(365)                        # 31 Dec of a normal year
    assert twin.clim_slot(s, c, 1) == stats._doy_index(dt.date(2024, 1, 1))
    assert twin.clim_slot(s, c, 8) == stats._doy_index(dt.date(2024, 1, 8))


# ---- the rollout seam, on an untrained twin with random weights (fixture in conftest.py) -----------------
from conftest import SCALE, make_init as _init  # noqa: E402

SC = [0.42, -0.91]


@pytest.mark.parametrize("steps", [1, 10, 14])
def test_rollout_is_valid_for_any_step_count(model, steps):
    out = model.rollout(_init(), SC, steps)
    assert out.shape == (steps, 6, 32, 36) and out.dtype == np.float32
    assert np.isfinite(out).all() and out[:, 1].min() >= 0
    assert 233.15 <= out[:, 0].min() and out[:, 0].max() <= 333.15


def test_perturbation_is_added_once_to_the_last_input_day(model):
    x = _init(); p = np.random.default_rng(2).normal(0, 1, (6, 32, 36)).astype("f4")
    shifted = x.copy(); shifted[-1] += p
    np.testing.assert_array_equal(model.rollout(x, SC, 5, p), model.rollout(shifted, SC, 5))
    np.testing.assert_array_equal(x, _init())                   # the caller's array is untouched


def test_zero_perturbation_changes_nothing(model):
    x = _init()
    np.testing.assert_array_equal(model.rollout(x, SC, 10, np.zeros((6, 32, 36), "f4")), model.rollout(x, SC, 10))


def test_rollout_is_deterministic(model):
    np.testing.assert_array_equal(model.rollout(_init(), SC, 10), model.rollout(_init(), SC, 10))


@pytest.mark.parametrize("steps", [1, 2])
def test_a_point_shock_travels_at_most_six_cells_a_day(model, steps):
    x = _init(); p = np.zeros((6, 32, 36), "f4"); p[:, 16, 18] = 2 * SCALE
    d = np.abs(model.rollout(x, SC, steps, p) - model.rollout(x, SC, steps))[-1].max(axis=0)
    ii, jj = np.indices(d.shape); far = np.maximum(abs(ii - 16), abs(jj - 18)) > 6 * steps
    assert d[~far].max() > 0 and d[far].max() == 0


def test_rollout_rejects_malformed_input(model):
    with pytest.raises(ValueError):
        model.rollout(_init()[1:], SC, 10)
    with pytest.raises(ValueError):
        model.rollout(_init(), SC, 15)


def test_heavy_rain_lift_raises_only_rain_past_the_cutoff():
    x = np.zeros((1, 1, 6, 32, 36), "f4"); x[..., 0, :, :] = 300.0
    x[0, 0, 1, 0, :5] = [7.0, 7.75, 8.0, 9.0, 12.0]
    out = twin.emit(x, np.array([8.0], "f4"))[0, 0, 1, 0, :5]
    np.testing.assert_allclose(out, [7.0, 7.75 + 0.5 * (10.01 - 7.75), 10.01, 10.01, 12.0], rtol=1e-6)
    np.testing.assert_array_equal(twin.emit(x, np.array([np.inf], "f4")), x)   # no cutoff, no change


def test_saved_weights_load_back_to_the_same_forecast(model, tmp_path):
    twin.save(model, tmp_path / "t.pt")
    again = twin.load(tmp_path / "t.pt")
    np.testing.assert_array_equal(again.rollout(_init(), SC, 3), model.rollout(_init(), SC, 3))
