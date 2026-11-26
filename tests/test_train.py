import numpy as np
import pytest

import train

kit = pytest.importorskip("monsoon_r1")
from monsoon_r1 import stats  # noqa: E402

needs_data = pytest.mark.skipif(not train.DATA.exists(), reason="training data not found")


@needs_data
def test_climatology_equals_the_graders_recipe():
    X, days, _ = train.load()
    keep = slice(0, 3 * 365 + 366)                         # 2005-2008, including 29 Feb 2008; keeps the kit's slow loop quick
    yyyymmdd = np.array([d.year * 10000 + d.month * 100 + d.day for d in days[keep]])
    ours = train.climatology(X[keep], days[keep])
    np.testing.assert_allclose(ours, stats.climatology(X[keep], yyyymmdd), rtol=0, atol=1e-3)


@needs_data
def test_load_gives_the_contract_grid_in_physical_units():
    X, days, static = train.load()
    assert X.shape == (len(days), 6, 32, 36) and static.shape == (2, 32, 36)
    assert 250 < X[:, 0].mean() < 310 and X[:, 1].min() >= 0       # t2m in K, rain never negative
    assert (days[1] - days[0]).days == 1


def test_best_cutoff_finds_the_threshold_that_separates_rain():
    # Truth is heavy exactly where the raw forecast reaches 7, so the cutoff 7 scores a perfect CSI.
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 9.5, (40, 32, 36)).astype("f4")
    truth = np.where(raw >= 7, 15.0, 1.0)
    zeros = np.zeros_like(truth)
    assert train.best_cutoff(raw, truth, zeros, zeros, grid=np.arange(4.0, 10.01, 0.25)) == 7.0


def test_best_cutoff_keeps_the_lift_off_when_it_only_adds_false_alarms():
    # Heavy rain falls only where the forecast is driest: lifting wet cells can only add false alarms.
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 9.5, (40, 32, 36)).astype("f4")
    truth = np.where(raw < 1, 15.0, 1.0)
    zeros = np.zeros_like(truth)
    assert train.best_cutoff(raw, truth, zeros, zeros, grid=np.arange(4.0, 10.01, 0.25)) == np.inf
