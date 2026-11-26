import numpy as np
import pytest

pytest.importorskip("mlx")
import mlx_twin  # noqa: E402
from conftest import make_init, make_twin  # noqa: E402

SC = [0.42, -0.91]


def test_mlx_forecast_matches_the_served_pytorch_model():
    t = make_twin(seed=5)
    f = mlx_twin.forecaster(mlx_twin.from_torch(t), mlx_twin.Fields(t))
    x = np.stack([make_init(1), make_init(2)]); sc = np.array([SC, [0.9, 0.1]], "f4")
    p = np.zeros((2, 6, 32, 36), "f4"); p[0, 0, 10:14, 20:24] = 3.0
    np.testing.assert_allclose(f(x, sc, 10, p), t.forecast(x, sc, 10, p), rtol=1e-5, atol=1e-3)


def test_weights_survive_the_round_trip_to_pytorch():
    t = make_twin(seed=6)
    back = mlx_twin.to_torch(mlx_twin.from_torch(t), make_twin(seed=7))
    for a, b in zip(t.nets.parameters(), back.nets.parameters()):
        np.testing.assert_array_equal(a.detach().numpy(), b.detach().numpy())
