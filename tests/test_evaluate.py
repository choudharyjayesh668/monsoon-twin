import numpy as np
import pytest

import train

pytest.importorskip("monsoon_r1")
import evaluate  # noqa: E402

needs_data = pytest.mark.skipif(not train.DATA.exists(), reason="training data not found")


@pytest.fixture(scope="module")
def data():
    X, days, _ = train.load()
    return X, days, train.climatology(X, days)


@needs_data
def test_harness_reproduces_the_damped_persistence_baseline(data):
    # Oracle computed separately with the kit's own climatology and true calendar dates, on 2021. A normal year, so
    # the doy_sincos slot equals the calendar slot (in leap years it is a day late from March, see twin.clim_slot).
    X, days, clim = data
    f = evaluate.damped_persistence(X, days, clim, evaluate.starts(days, range(2005, 2020)))
    r = evaluate.forecast_scores(f, X, days, evaluate.starts(days, (2021,)), clim)
    np.testing.assert_allclose(r["C"]["by_lead"], [0.0697, 0.0314, 0.0102], atol=5e-4)
    assert r["B"]["mean"] == pytest.approx(-0.0023, abs=5e-4)


@needs_data
def test_persistence_scores_zero_against_itself_at_day_one(data):
    X, days, clim = data
    r = evaluate.forecast_scores(evaluate.persistence, X, days, evaluate.starts(days, (2020,))[:50], clim)
    assert r["B"]["by_lead"][0] <= 1e-6                    # never better than the better reference, which is itself


@needs_data
def test_dynamics_known_answer_for_persistence(data):
    X, days, _ = data
    r = evaluate.dynamics_scores(evaluate.persistence, X, days, evaluate.starts(days, (2021,))[::60], evaluate.stats_std())
    assert r["valid_14"] == 1.0 and r["null_ok"] == 1.0
    for name in ("warm_ocean", "monsoon_stall", "arabian_heat"):
        s = r["shocks"][name]
        assert s["nonzero"] == 1.0 and s["localized"] == 1.0 and s["stable"] == 1.0
        assert s["propagates"] == 0.0                         # a frozen shock never spreads
    assert r["causality_points"] == pytest.approx(15 * 3 / 4)  # 3 of 4 checks on both kit shocks
