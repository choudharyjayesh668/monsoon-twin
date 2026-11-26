import base64
import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

import twin
from conftest import make_init, make_twin

SC = [0.42, -0.91]


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """The real serve module, loading a small untrained twin saved where it expects weights."""
    path = tmp_path_factory.mktemp("w") / "twin.pt"
    model = make_twin(seed=3)
    twin.save(model, path)
    old, twin.WEIGHTS = twin.WEIGHTS, path
    try:
        serve = importlib.reload(importlib.import_module("serve"))
    finally:
        twin.WEIGHTS = old
    return TestClient(serve.app), model


def _b64(a):
    a = np.ascontiguousarray(a, dtype="<f4")
    return {"b64": base64.b64encode(a.tobytes()).decode(), "shape": list(a.shape)}


def test_health_says_ready(served):
    client, _ = served
    r = client.post("/health")
    assert r.status_code == 200 and r.json()["status"] == "ready" and r.json()["model_name"]


def test_simulate_returns_the_models_rollout_as_plain_lists(served):
    client, model = served
    x = make_init()
    r = client.post("/simulate", json={"init_states": x.tolist(), "doy_sincos": SC, "steps": 10, "perturbation": None})
    body = r.json()
    assert r.status_code == 200 and isinstance(body["rollout"], list) and isinstance(body["inference_ms"], float)
    np.testing.assert_array_equal(np.asarray(body["rollout"], "f4"), model.rollout(x, SC, 10))


def test_simulate_accepts_base64_and_a_perturbation(served):
    client, model = served
    x = make_init(); p = np.full((6, 32, 36), 0.5, "f4")
    r = client.post("/simulate", json={"init_states": _b64(x), "doy_sincos": SC, "steps": 3, "perturbation": _b64(p)})
    np.testing.assert_array_equal(np.asarray(r.json()["rollout"], "f4"), model.rollout(x, SC, 3, p))


@pytest.mark.parametrize("bad", [{"init_states": [[0.0]], "doy_sincos": SC, "steps": 10},
                                 {"init_states": None, "doy_sincos": SC, "steps": 10}, {"steps": 10}])
def test_malformed_requests_get_a_clear_422(served, bad):
    client, _ = served
    assert client.post("/simulate", json=bad).status_code == 422
