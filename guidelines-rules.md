# Guidelines & Rules: Multiverse of Monsoon, Stage 1

**These rules are binding. Every change to this repo must comply. No exceptions, no "just for now", no workarounds.**
If a rule seems to block you, stop and ask. Do not bend it.

Sources: the Track 2 problem statement (`../Yugaantar DevForge Tracks.pdf`), plus `docs/R1_CONTRACT.md`, `monsoon_r1/scoring.py` and `monsoon_r1/repo_scan.py` in the kit.
If this file and the contract ever disagree, the contract wins. Fix this file right away.

Scope: **Stage 1 only (the automated benchmark).** Stage 2 (live demo, dashboard polish, impact module) is out of scope until someone says otherwise. Still, a minimal dashboard is required for repo points (R-4.7).

---

## 0. Folder layout (fixed)

```
yugaantar/
├── Yugaantar DevForge Tracks.pdf   READ-ONLY reference: the Track 2 problem statement. Stays here, outside the repo.
├── yug-monsoon-kit-main/   READ-ONLY reference: the organisers' kit (github.com/OfficialAbhinavSingh/yug-monsoon-kit)
│   ├── docs/R1_CONTRACT.md    the API contract
│   ├── monsoon_r1/            validator, scoring, repo scan, starter serve.py
│   └── release/               v1.0.0 Release assets: india_climate_train.nc, stats.json, kit wheel, SHA256SUMS.txt
└── monsoon-twin/           OUR SUBMISSION: the only folder that becomes the public GitHub repo
```

- **R-0.1** `monsoon-twin/` is the submission repo and is the only folder ever committed or pushed. Every rule below about "the repo" means this folder.
- **R-0.2** `yug-monsoon-kit-main/` is reference material only. Never edit it, commit it, or copy its code into the submission. Exception: `stats.json` may be copied in, because the model must bundle it (R-1.7).
- **R-0.3** Data comes from the kit's `release/` folder, which holds the v1.0.0 Release assets: https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/tag/v1.0.0. Check them against `SHA256SUMS.txt`. Training reads the data by path (default `../yug-monsoon-kit-main/release/india_climate_train.nc`, overridable with a CLI flag). Never copy it into `monsoon-twin/`.
- **R-0.4** Local tooling runs from the `.venv` inside `monsoon-twin/` (git-ignored). Install the kit there from its wheel, `pip install ../yug-monsoon-kit-main/release/monsoon_r1_kit-1.0.0-py3-none-any.whl`, for validating, scanning and evaluating only.
- **R-0.5** **The user makes every `git commit` and `git push`.** The agent prepares the working tree, runs the checks and reports what is ready. A hook in `.claude/hooks/` blocks both commands for the agent.

## 1. Runtime environment (hard limits)

- **R-1.1** The server starts with exactly `python serve.py --port 8000`, run from the repo root. No other flags or setup steps.
- **R-1.2** It listens on `0.0.0.0`, never on `127.0.0.1`.
- **R-1.3** The grader runs **offline**: no internet, no DNS, no downloads at startup or at request time.
- **R-1.4** Limits: **6 GB RAM, 4 CPU cores, no GPU.** Never assume CUDA or MPS at inference. Use `map_location="cpu"`.
- **R-1.5** The repo is **read-only** at evaluation. Write only to `/tmp`. Never write caches, logs or compiled artefacts next to the code.
- **R-1.6** Only these libraries are available at evaluation: `torch` (CPU), `numpy`, `scipy`, `pandas`, `xarray`, `scikit-learn`, `lightgbm`, `onnxruntime`, `fastapi`, `flask`, `uvicorn`, `gunicorn`. Any other library must be vendored inside the repo. Code that is committed and runs at grading time (`serve.py` and everything it imports) must never import `monsoon_r1`, because the kit is not installed in the grading sandbox (R-4.10).
- **R-1.6a** Graded code (`serve.py` and everything it imports) imports only the R-1.6 list. Any other library is local-only (training, tests, evaluation) and must be open source. **Training and local evaluation run only on an open-source GPU stack: MLX (MIT)** (user decision, 2026-11-26). No PyTorch GPU backend (it calls Apple's closed MPS/MPSGraph) and no Apple Accelerate for maths. PyTorch appears in training only to write the checkpoint `serve.py` loads. Other local-only libraries: `netCDF4` (MIT, reads the `.nc`), `pytest` (MIT), `httpx` (BSD-3), `requests` (Apache-2.0), the kit wheel (MIT).
- **R-1.7** All weights and static fields (land_sea_mask, orography, climatology, normalisation stats) live **inside the repo** and load from paths relative to `serve.py` (`Path(__file__).parent`). A Hugging Face link alone is not enough.
- **R-1.8** Total repo size stays **under 2 GB**. Commit weights as plain files, not through Git LFS: an LFS pointer is under 1 KB, so the scan flags it (R-4.6) and the server cannot load it.

## 2. API contract (any violation scores 0 for that request)

- **R-2.1** `POST /health` returns HTTP 200 `{"status": "ready", "model_name": "<str>"}`, but **only once the model is fully loaded**. Load everything at import time or startup, within **120 s**.
- **R-2.2** `POST /simulate` takes `{"init_states": [3,6,32,36], "doy_sincos": [sin, cos], "steps": int, "perturbation": null | [6,32,36]}`.
- **R-2.3** It returns `{"rollout": [steps,6,32,36], "inference_ms": float}` as **plain nested JSON lists of finite floats**. Return exactly `steps` days. `rollout[k]` is day k+1 after the last input day.
- **R-2.4** Channel order is fixed: `0 t2m (K), 1 tp (mm/day), 2 u10 (m/s), 3 v10 (m/s), 4 msl (Pa), 5 tcwv (kg/m²)`.
- **R-2.5** **Physical units on the wire, both ways, always.** `stats.json` normalisation is used only inside the model.
- **R-2.6** `init_states` are oldest first. `doy_sincos = [sin(2π·doy/365), cos(2π·doy/365)]` refers to the **last input day**, so forecast step k has day-of-year `doy + k`.
- **R-2.7** `perturbation` is a physical delta added to `init_states[-1]` **before** simulating. Apply it exactly once, to the last input day only.
- **R-2.8** Grid: 32 lat × 36 lon, lat 6.5→37.5 (south→north), lon 65.5→100.5 (west→east). Never transpose or flip it.
- **R-2.9** Accept the tolerated `{"b64","shape"}` input form, but always respond with lists.
- **R-2.10** Every 10-step request finishes in **under 5.0 s** of wall time, measured by the grader, including JSON parsing and serialising. Target **< 1 s** on 4 cores. No retries happen.
- **R-2.11** Handle any `steps` from 1 to 14 (`MAX_STEPS`). Never hard-code 10.

## 3. Validity gate (failing it on any of the 8 requests disqualifies)

- **R-3.1** No NaN or Inf anywhere in the output. Sanitise with `np.nan_to_num` as a last line of defence, but the model itself must not produce them.
- **R-3.2** Rain (`tp`) is never negative. Clamp it to `>= 0` on output.
- **R-3.3** `t2m` stays within **[233.15 K, 333.15 K]** (−40 °C to +60 °C). Clamp it on output.
- **R-3.4** Rollouts stay stable for **at least 14 steps**: no explosion, no drift beyond physical ranges. Test 14, not 10.
- **R-3.5** Output shape is exactly `(steps, 6, 32, 36)`.

## 4. Repo integrity (15 points, scanned statically; critical flags cost all 8 anti-cheat points)

- **R-4.1** **No external weather or data sources** in inference code (`serve.py` and every local module it imports, transitively). That includes ERA5/CDS, WeatherBench2, ARCO, Open-Meteo, IMD, NOAA, any URL or any network call.
- **R-4.2** **No hardcoded dates in inference code.** No `YYYY-MM-DD` / `YYYYMMDD` literals and no `date(20xx…)` / `datetime(20xx…)` / `Timestamp(20xx…)` calls. Day-of-year comes only from `doy_sincos`. Calendar years are never sent and must never be inferred.
- **R-4.3** **No lookup tables or archives of past weather.** Never ship any array shaped `(n > 400, …, 32, 36)`. A one-year climatology (≤ 366 days) or harmonic coefficients are allowed. Never match an input against stored history.
- **R-4.4** **Never commit climate data files** (`.nc`, `.nc4`, `.grib`, `.grib2`, `.zarr`). Data stays in the kit folder (R-0.3), and `.gitignore` blocks these extensions as a safety net.
- **R-4.5** Commit **real training code** (a genuine `loss.backward()` / `optimizer.step()` loop) in a file separate from the server. It must reproduce the shipped weights.
- **R-4.6** Weights ship in the repo with a standard extension (`.pt`, `.pth`, `.onnx`, `.safetensors`, …) and are > 1 KB.
- **R-4.7** Keep the **server separate from the UI.** No streamlit, gradio, dash, plotly or matplotlib imports in `serve.py` or anything it imports. Dashboard code lives in its own directory (e.g. `dashboard/`) and calls `/simulate` over HTTP.
- **R-4.8** A `README.md` at the root explains how to train and how to run.
- **R-4.9** Whenever inference code changes or files are added, the agent runs the scan, gets **0 flags**, and reports the result so the user can commit (R-0.5):
  `python -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('.'), indent=1))"` (run inside `monsoon-twin/`)
- **R-4.10** **No kit code in the submission repo.** No `monsoon_r1/` package, no copied kit files (`build_dataset.py`, `repo_scan.py`, `client.py`, the starter `serve.py`), and no `release/` folder. The kit is used only through the pip-installed wheel, by local tooling (eval scripts, the validator, the scan). If an eval script imports `monsoon_r1`, keep it out of the server's import graph. Measured 2026-11-26: with the kit's files present, the scan treats them as inference code (they contain the string `/simulate`) and raises **12 critical flags**, because `repo_scan.py` lists the banned API names. That costs all 8 anti-cheat points. Run R-4.9 on a fresh clone before submitting.
- **R-4.11** **The route string marks a file as inference code.** The scan treats any `.py` file containing `/simulate`, or `simulate` in quotes, as inference code, together with every local module it imports. This holds even inside a comment or docstring. So:
  - Only `serve.py` and `tests/test_serve.py` contain that string, and they import only `twin.py`.
  - `train.py`, `evaluate.py` and their tests call the model in-process as `twin.rollout` and never contain the string. If `train.py` joined the inference set, the 3 training-pipeline points would be lost.
  - Inference files hold no plotting imports, no names of weather services, and no 8-digit numbers that look like dates.

## 5. Forecast skill (45 points)

- **R-5.1** The baselines are persistence (repeat the last input day) and **the grader's climatology: a 366-day calendar-day mean over 2005–2021 with a 15-day circular running mean** (`monsoon_r1.stats.climatology`). Evaluate against exactly these baselines, nothing weaker.
- **R-5.2** Score metric: lat-weighted RMSE skill `1 − RMSE / min(RMSE_persist, RMSE_clim)` at **lead days 1, 4 and 8**, all 6 channels. Full points at mean skill ≥ **0.12**.
- **R-5.3** Heavy-rain CSI at **10 mm/day**, same leads, compared with the better reference. Full points at CSI skill ≥ **0.10**. Never trade away heavy-rain hits for RMSE without measuring both.
- **R-5.4** **Hard gate:** beat persistence at day 4. Never ship a model that fails this on the validation years.
- **R-5.5** Three-way split: train on 2005–2017, tune on 2018–2019, report **once** on 2020–2021 after the recipe is frozen. Then retrain the frozen recipe (same settings, epochs and seeds) on 2005–2021 and ship those weights (user decision, 2026-11-26). The reported numbers come from the 2020–2021 hold-out run, and the brief says so. The hidden test is 2023–24 and covers monsoon onset, heavy rain, heatwave, cyclone and western disturbance; evaluation reports event-like windows of each kind separately.
- **R-5.6** **Score three ways, keep what lifts the lowest.** The grading script is not in the kit, so every evaluation reports forecast skill under three readings: (A) per window with leads 1, 4, 8 pooled into one RMSE, (B) per window with each lead scored alone then averaged, (C) all windows pooled. A model change is kept only if it raises the lowest of the three and lowers none by more than 0.005 skill.

## 6. Dynamics (25 points)

- **R-6.1** **Anti-blur:** at day 8, the per-channel spatial variance must stay between **40% and 250%** of the real field's. Check it per channel, `tp` included. Never fix blur by injecting random noise (it breaks R-6.3).
- **R-6.2** **Inference is deterministic.** No dropout, no unseeded sampling, no ensemble randomness at inference.
- **R-6.3** **A null shock changes nothing.** `perturbation = zeros` gives output identical to `perturbation = null`, within `1e-3 × channel std` (target: bit-identical).
- **R-6.4** **Shocks are causal and local.** On day 1, ≥ **60%** of the response energy must sit within **3 cells** of the shocked box. The model's per-step receptive field must stay small. No global pooling, attention or deep U-Net bottleneck that teleports a local shock across India in one step.
- **R-6.5** **Shocks propagate.** Energy outside the neighbourhood must grow from day 1 to days 4–8 (> 1.5× day 1 and > 2% of peak), at no more than **8 cells/day** (far-field share < 5% beyond the reachable radius).
- **R-6.6** **Shocks stay stable:** the response never exceeds **5 channel stds** anywhere.
- **R-6.7** **Shocks must get a response.** A non-zero response (≥ 1e-3 std) is required, otherwise the 15 causality points are 0 and the R2 gate fails. Test with the exact shocks from `monsoon_r1.scoring.make_shock`: `warm_ocean` (Bay of Bengal, +3 K t2m, +15% tcwv, lat 8–20, lon 82–94) and `monsoon_stall` (Arabian Sea, −40% u10/v10, lat 8–20, lon 65–75).
- **R-6.8** **Inject a perturbation: ripples must propagate.** A **+2.5 °C Arabian Sea heat anomaly** (t2m +2.5 K over lat 8–20, lon 65–75) dropped into the model must produce realistic atmospheric ripples that travel outward: the response is still visible at day 8, with no smoothing and no averaging out. Evaluation runs this shock beside the two in R-6.7 and applies R-6.4 to R-6.6 to all three. A model change that lowers any shock's pass rate is rejected.

## 7. Verification (required before calling anything done)

- **R-7.1** The schema validator passes: `python -m monsoon_r1.schema_validator http://localhost:8000` must print `all checks passed`.
- **R-7.2** A local evaluation script mirrors the grader using `monsoon_r1.scoring` functions (skill, CSI, blur ratio, causality, null consistency) on held-out windows across all seasons. Report numbers, not impressions.
- **R-7.3** Test offline and resource-limited before submitting:
  `docker run --rm --network none --cpus 4 --memory 6g -v $PWD:/sub:ro -w /sub -p 8000:8000 <python image with allowed libs> python serve.py --port 8000`
- **R-7.4** Time the cold start (< 120 s) and a 10-step request (< 5 s; target < 1 s) inside that container.
- **R-7.5** Never claim a score, a pass or a speed that was not measured. Write down any failure honestly.

## 8. Ground rules (disqualification if broken)

- **R-8.1** **Offline only. Own every output.** Everything the system predicts comes from our model alone, computed from the request inputs and the weights in the repo.
- **R-8.1a** **No external weather APIs, anywhere in the repo.** That covers training, inference, dashboard, scripts and notebooks. No Open-Meteo, OpenWeatherMap, WeatherAPI, Visual Crossing, Tomorrow.io, met.no, Meteomatics, NASA POWER, CDS/`cdsapi`, ERA5/ARCO/WeatherBench2 buckets, IMD/`imdlib`/MOSDAC, NOAA/GFS/NOMADS, or any similar service, and no HTTP calls to fetch weather data. The **only** data source is the provided `india_climate_train.nc` and `stats.json`. Never call `monsoon_r1.build_dataset` or copy it in, because it downloads from cloud archives (see R-4.10).
- **R-8.2** No lookup tricks, no hardcoded dates, no shipped archives of past weather (see R-4.2 and R-4.3).
- **R-8.3** Original work: repo, training, weights and dashboard are built during the hackathon. Open-source libraries are fine.
- **R-8.4** The user commits and pushes before the deadline (R-0.5). The repo is **frozen** at the deadline and later changes are ignored.
- **R-8.5** The dashboard must run on live `/simulate` output, never on pre-rendered results.

## 9. Submission checklist (every box must be ticked)

- [ ] Public GitHub repo pushed from `monsoon-twin/` only, with `serve.py` at the root
- [ ] Training code at the root or in `train/`, reproducible from the README
- [ ] Weights in the repo as plain files (R-1.8), total < 2 GB
- [ ] Dashboard code separate from the server code
- [ ] `README.md` explains how to train and run
- [ ] 2-page PDF brief: p1 covers model design, how drift and blur are prevented, and failure modes; p2 covers the visualisation and the impact use case
- [ ] Repo scan shows 0 flags. Schema validator passes. Offline Docker run passes. Local eval beats persistence at day 4 and responds locally to both shocks.

## 10. Engineering discipline

- **R-10.1** Make the smallest change that fully works (ponytail). No speculative abstractions, configs or features beyond Stage 1.
- **R-10.2** Every non-trivial piece of logic (the doy-to-lead mapping, normalisation round-trips, perturbation application, clamps) has a small test or assert-based self-check.
- **R-10.3** A known shortcut gets a code comment: `shortcut: <the limit>, <when to upgrade>`.
- **R-10.4** Never commit secrets, data files or `.venv`.
