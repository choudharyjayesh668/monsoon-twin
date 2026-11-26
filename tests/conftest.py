import numpy as np
import pytest

import twin

MEAN = np.array([292.4, 3.6, 0.9, 0.2, 101122.0, 30.0], "f4")
STD = np.array([13.5, 7.6, 2.9, 2.4, 619.0, 20.5], "f4")
SCALE = np.array([1.7, 6.2, 1.5, 1.4, 281.0, 6.1], "f4")


def make_twin(seed: int = 0) -> twin.Twin:
    """An untrained twin with random weights on synthetic fields: enough to test every contract property."""
    import torch
    rng = np.random.default_rng(seed)
    clim = MEAN[None, :, None, None] + rng.normal(0, 0.3, (366, 6, 32, 36)).astype("f4") * SCALE[None, :, None, None]
    clim[:, 1] = np.abs(clim[:, 1])
    static = np.stack([rng.uniform(0, 1, (32, 36)), rng.uniform(0, 3000, (32, 36))]).astype("f4")
    m = twin.Twin(width=16, blocks=5)
    m.set_fields(clim, SCALE, static, MEAN, STD)
    torch.manual_seed(seed)
    for p in m.nets.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return m.eval()


def make_init(seed: int = 1) -> np.ndarray:
    r = np.random.default_rng(seed)
    x = MEAN[None, :, None, None] + r.normal(0, 1, (3, 6, 32, 36)).astype("f4") * SCALE[None, :, None, None]
    x[:, 1] = np.abs(x[:, 1])
    return x


@pytest.fixture(scope="module")
def model():
    return make_twin()
