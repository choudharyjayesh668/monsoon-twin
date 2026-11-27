"""HTTP server for the monsoon twin. The organisers run it offline with:  python serve.py --port 8000

POST /health    -> {"status": "ready", "model_name": ...}; the weights are loaded before the server starts listening
POST /simulate  -> {"rollout": [steps,6,32,36], "inference_ms": ...}; physical units, plain nested JSON lists
"""
import sys

sys.dont_write_bytecode = True        # the repo is read-only at evaluation: never write __pycache__ next to the code

import argparse  # noqa: E402
import base64  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

import twin  # noqa: E402

torch.set_num_threads(4)              # the sandbox has 4 cores
MODEL = twin.load(twin.WEIGHTS)
MODEL_NAME = "monsoon-twin"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST"], allow_headers=["*"])   # the dashboard


def _array(x) -> np.ndarray:
    """Nested lists (the contract) or {"b64", "shape"} (tolerated): little-endian float32."""
    if isinstance(x, dict):
        shape = [int(s) for s in x["shape"]]
        raw = base64.b64decode(x["b64"], validate=True)
        if len(raw) != 4 * int(np.prod(shape)):
            raise ValueError("b64 length does not match shape")
        return np.frombuffer(raw, dtype="<f4").reshape(shape)
    return np.asarray(x, dtype="f4")


@app.post("/health")
def health():
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate(request: Request):
    t0 = time.perf_counter()
    try:
        body = await request.json()
        pert = body.get("perturbation")
        out = MODEL.rollout(_array(body["init_states"]), body["doy_sincos"], int(body["steps"]),
                            None if pert is None else _array(pert))
    except (AttributeError, KeyError, TypeError, ValueError) as e:     # AttributeError: a body that is not an object
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=422)
    # A ready-made response skips FastAPI's per-element encoder, which is slow on 69,120 numbers.
    return JSONResponse({"rollout": out.tolist(), "inference_ms": (time.perf_counter() - t0) * 1000.0})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, host="0.0.0.0", port=ap.parse_args().port, log_level="warning")
