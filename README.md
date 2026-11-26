# Monsoon Twin

A fast digital twin of India's daily weather for Track 2, "Multiverse of Monsoon". From the last 3 days of India's state, it rolls temperature, rain, wind, pressure and moisture forward one day at a time on the 32 × 36 grid. A shock added to the last day, such as a warmer Bay of Bengal, spreads outward step by step.

- **Model**: a small convolutional network that predicts each day's change of the anomaly from climatology. It only sees 6 cells around each point per day, so a local shock spreads at most 6 cells a day. See [ADR 1](docs/adr/0001-local-one-day-step.md).
- **Server**: `serve.py`, offline, CPU only, the API from the organisers' contract.
- **Dashboard**: `dashboard/index.html`, which calls the live server.

## Run the server

```bash
pip install torch numpy fastapi uvicorn
python serve.py --port 8000
```

It listens on `0.0.0.0`, loads `weights/twin.pt` before answering, and needs nothing from the network. It writes nothing next to the code.

- `POST /health` returns `{"status": "ready", "model_name": "monsoon-twin"}`.
- `POST /simulate` takes `{"init_states": [3,6,32,36], "doy_sincos": [sin, cos], "steps": 1..14, "perturbation": null | [6,32,36]}` and returns `{"rollout": [steps,6,32,36], "inference_ms": ...}`. Everything is in physical units: K, mm/day, m/s, m/s, Pa, kg/m².

## Train

Training reads the organisers' data from the kit folder next to this repo (`../yug-monsoon-kit-main/release/india_climate_train.nc`), or from `--data`. The data file is never copied into this repo.

```bash
python -m venv .venv && source .venv/bin/activate
pip install mlx torch numpy xarray netCDF4 fastapi uvicorn httpx pytest
pip install ../yug-monsoon-kit-main/release/monsoon_r1_kit-1.0.0-py3-none-any.whl   # scoring, for evaluate.py and tests

# development run: train on 2005-2017, watch the loss on 2018-2019
python train.py --years 2005-2017 --val-years 2018-2019 --out /tmp/dev.pt
python train.py --tune-rain /tmp/dev.pt --val-years 2018-2019 --out /tmp/dev.pt     # heavy-rain cutoffs
# the shipped weights: the same recipe on all years, keeping the dev run's heavy-rain cutoffs
python train.py --years 2005-2021 --rain-cut-from /tmp/dev.pt
```

Training runs on the Apple GPU through [MLX](https://github.com/ml-explore/mlx) (open source, MIT), in `mlx_twin.py`, a layer-for-layer copy of the served network. The weights are saved as the PyTorch checkpoint `serve.py` loads, and a test checks that both give the same forecast. `netCDF4` (MIT licence) is needed only to read the data file.

## Evaluate

```bash
python evaluate.py /tmp/dev.pt --years 2018 2019 --baseline --dynamics --events
```

This scores a model with the organisers' scoring functions:
- forecast skill and heavy-rain skill at days 1, 4 and 8, against the better of persistence and climatology;
- day-8 blur;
- 14-step validity;
- the null shock;
- three shocks: the grader's warm Bay of Bengal and weaker Arabian Sea wind, plus a +2.5 °C Arabian Sea;
- windows like the five hidden ones.

The grader's averaging across windows is not published, so skill is reported three ways (see [GLOSSARY.md](GLOSSARY.md)).

## Test

```bash
pytest
python serve.py --port 8000 &  python -m monsoon_r1.schema_validator http://localhost:8000
```

## Dashboard

Start the server, then open `dashboard/index.html` in a browser. It sends a sample 3-day window to `/simulate`, with or without a shock, and maps any field on any day. It can also map the difference between the shocked and the unshocked world.

## Results

Measured once on **2020–2021**, held out from training and tuning: the development model, trained on 2005–2017 and tuned on 2018–2019. The shipped weights use the same recipe retrained on 2005–2021 ([ADR 2](docs/adr/0002-ship-weights-retrained-on-all-years.md)). Scored with the organisers' functions on 721 windows. The grader's way of averaging across windows is not published, so forecast points are shown three ways.

| | Damped persistence | **Monsoon twin** |
|---|---|---|
| Forecast + heavy-rain points (of 45), readings A / B / C | 17.5 / 0.0 / 9.1 | **35.4 / 20.7 / 40.8** |
| Skill at days 1 / 4 / 8 (reading B) | +0.034 / −0.019 / −0.024 | **+0.250 / +0.004 / −0.112** |
| Day 4 vs persistence (qualifying gate) | +0.202 | **+0.225** |
| No-blur points (of 10) | 10.0 | **10.0** |
| Shock points (of 15) | 11.2 | **13.4** |
| Valid for 14 steps; null shock unchanged | 100%; 100% | **100%; 100%** |

Shock pass rates on 61 windows:

| Shock | Local on day 1 | Spreads by days 4–8 | Stays within 5 std |
|---|---|---|---|
| Warm Bay of Bengal | 100% | 98% | 74% |
| Weaker Arabian Sea wind | 100% | 54% | 90% |
| +2.5 °C Arabian Sea | 100% | 77% | 92% |

Known weak spots:
- Extra moisture can move or strengthen heavy-rain cells by more than 5 std (38 mm/day) in single cells.
- The weaker-wind shock fades before it has spread far enough in about half the windows.
- In leap years the climatology is one day late from March on, because the year is never sent (about −0.004 skill at days 4–8).

Server (`python serve.py --port 8000`, read-only copy, open-source CPU PyTorch on OpenBLAS): ready in 3.1 s; a 10-step request takes 0.09–0.14 s. The organisers' schema validator reports all checks passed, and the repo scan scores 15/15 with no flags.

## Layout

| Path | What |
|---|---|
| `twin.py` | The model: network, climatology, rollout, output guards |
| `serve.py` | The HTTP server (imports only `twin.py`) |
| `train.py` | Training (MLX on the GPU) |
| `mlx_twin.py` | The MLX copy of the network, for training and local evaluation |
| `evaluate.py` | Local scoring with the organisers' functions |
| `weights/twin.pt` | The shipped weights, climatology and scales |
| `dashboard/` | The map page |
| `tests/` | Tests |
| `docs/adr/` | Design decisions |
