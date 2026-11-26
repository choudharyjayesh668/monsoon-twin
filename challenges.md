# Challenges

## Training speed, and keeping the whole maths stack open source

**The constraint.** We want every library that does our maths to be open source, not just the framework we install. On a Mac that rules out two Apple libraries that frameworks often call without saying so:

- **Accelerate**: Apple's closed maths library for the CPU (BLAS, LAPACK, vector maths).
- **Metal Performance Shaders / MPSGraph**: Apple's closed library of GPU kernels (convolutions and the like).

**Metal** itself, the interface every program uses to reach the GPU on macOS, can't be avoided. We accept it, as long as the kernels running on it are open source.

**What we did, in order.**

1. **PyTorch on the GPU (the `mps` backend).** Training was too slow for our iteration loop: about 0.06 s per step for a one-step rollout and 0.45 s for an eight-step one. A full run took over 40 minutes. Profiling showed the GPU was already compute-bound, so larger batches didn't help.
2. **Speeding PyTorch up.** bfloat16 convolutions in a channels-last layout made each step 3.4× faster in isolation and 2.1× faster in a real run, with identical day-1 skill (0.270 vs 0.270). The problem: PyTorch is open source (BSD-3), but its GPU backend hands the convolutions to Apple's closed MPS/MPSGraph. That is the same situation as PyTorch on the CPU calling Accelerate. We stopped using it.
3. **Other open-source GPU stacks.** We looked at MLX (MIT), tinygrad (MIT), IREE (Apache-2.0) and Vulkan through MoltenVK (Apache-2.0). They all reach the GPU through Metal, which is unavoidable. **MLX** is the most mature of them, and it runs its own open-source kernels, so we chose it. We also briefly considered a conda-forge PyTorch build on OpenBLAS (BSD-3) for CPU maths. We dropped that in favour of doing all training on the GPU.
4. **Porting to MLX.** `mlx_twin.py` is a layer-for-layer copy of the served network in MLX. Training (`train.py`) and local evaluation (`evaluate.py`) both run on it.
   - The weights are converted to the PyTorch checkpoint that `serve.py` loads. The grader's sandbox has CPU PyTorch, so the server stays on PyTorch.
   - A test checks that the MLX and PyTorch versions give the same 10-step forecast, shock included.
   - **Cost:** on this network MLX is about 1.7× slower than PyTorch's GPU backend: 54 vs about 32 ms per one-step training step at batch 32. One full training run took 18–19 minutes (1,088 s for the development run, 1,162 s for the final one).

**Checking the server locally.** `serve.py` has to use CPU PyTorch, because that is what the grader's sandbox provides. To check it without Accelerate, we run it from a conda-forge environment (outside the repo): PyTorch built against **OpenBLAS** (BSD-3) and LLVM OpenMP (Apache-2.0). There, `BLAS_INFO=open` and `libblas` points to `libopenblas`. This build is compiled with GPU support and weak-links Apple's GPU frameworks, but they are loaded only if code asks for the GPU, which `serve.py` never does.

**Still open.** The numpy in our local environment is linked against Accelerate. Our code uses numpy only for element-wise maths and averages, which don't call Accelerate's BLAS/LAPACK routines. Still, a strictly open stack would use a numpy built on OpenBLAS, for example from conda-forge.

## Population data on the flood page is from 2011

**What we use.** The flood page counts the people who live in the alert area. The numbers come from the WorldPop 2011 estimate for India (1 km grid, CC BY 4.0), which spreads census counts over the map. `demo/make_population.py` adds it up onto the twin's 1° cells.

**Why 2011: recent population data was hard to find.** We looked for newer counts and found none that fit:
- The Census of India 2011 is still the latest census. The 2021 census was postponed, and no results of a newer census are published.
- Newer numbers exist only as estimates or projections: WorldPop's yearly grids after 2011, GHSL's population grids, and the official state-level projections. They extend the 2011 counts with models, and the official projections stop at state level, so they cannot show where in a state people live.
- An open, gridded, census-based map that works offline and that we may bundle in the repo (CC BY 4.0) exists for 2011. So 2011 is the most recent year that rests directly on a census.

**What this means.**
- The counts are about 15 years old. India's population grew by roughly 15–20% since 2011, and cities grew faster, so the page **undercounts** the people at risk, most of all around fast-growing cities.
- Each 1° cell is about 110 km wide. Everybody in a cell counts as "in the alert area" when the cell has an alert, so the count is a rough size, not a list of affected towns.
- The population is a fixed map. The alerts come live from the twin; only the people count uses the fixed map.
