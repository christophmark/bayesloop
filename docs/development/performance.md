# bayesloop v2 Performance Analysis

Date: 2026-04-27

Environment used for measurements:

- macOS 26.2 on Apple Silicon
- CPython 3.12.4
- NumPy 2.4.4
- SciPy 1.17.1
- Current branch: `v2`

The benchmark harness lives in `benchmarks/performance_analysis.py`.

## Executive Summary

The best first speedup is not Numba or a C extension. It is algorithmic reuse of observation likelihoods.

`Study.fit` currently evaluates the same likelihood once in the forward pass and again in the backward pass. `HyperStudy.fit` amplifies that cost by calling `Study.fit` once per hyperparameter value even though the observation likelihoods are identical across those fits. A cached-likelihood prototype reproduced the baseline results exactly and delivered 1.45x to 1.58x speedups for larger `Study` cases and 1.86x for the larger `HyperStudy` case.

The second-best speedup is model-specific NumPy optimization in built-in observation models. In microbenchmarks, a Gaussian likelihood implementation that precomputes parameter-grid invariants was 2.35x faster than the current likelihood loop, and caching repeated Poisson observation values was 25.84x faster than recomputing the same likelihood for every time step.

Numba is useful as an optional tool for custom loop-heavy likelihood kernels, but it should not be the default route for the core built-ins. On the measured Gaussian likelihood kernel, Numba was 1.92x faster than the current code but slower than optimized NumPy. Most transition-model time is already inside SciPy C kernels, so Numba has little to compile there.

CPython/Cython extensions should be deferred. They would add packaging complexity and are unlikely to beat the cheaper wins above until bayesloop has a smaller, explicitly typed internal kernel API.

## Benchmark Commands

Full representative pass:

```bash
uv run --with-editable . python benchmarks/performance_analysis.py --repeats 3 --json /tmp/bayesloop_perf_full.json
```

Quick pass with likelihood microbenchmarks and optional Numba:

```bash
uv run --with-editable . --with numba python benchmarks/performance_analysis.py --quick --case poisson_static_1d --micro --numba-micro
```

Profile HyperStudy:

```bash
uv run --with-editable . python benchmarks/performance_analysis.py --case hyper_gaussian_random_walk_2d --profile-case hyper_gaussian_random_walk_2d --repeats 1
```

## Fit Benchmark Results

These timings include likelihood precomputation in the cached-likelihood prototype.

| case | baseline median s | cached median s | speedup | likelihood cache MiB | validation |
| --- | ---: | ---: | ---: | ---: | --- |
| `poisson_static_1d` | 0.1688 | 0.1069 | 1.58x | 47.7 | exact |
| `gaussian_random_walk_2d` | 0.2251 | 0.1553 | 1.45x | 44.9 | exact |
| `hyper_gaussian_random_walk_2d` | 0.8682 | 0.4668 | 1.86x | 12.4 | exact |
| `ar1_static_2d` | 0.3797 | 0.2407 | 1.58x | 89.4 | exact |
| `bivariate_random_walk_2d` quick | 0.0236 | 0.0212 | 1.11x | 1.1 | exact |

Validation means zero measured difference in log evidence, posterior mean checksum, posterior final normalization, and HyperStudy entropy where applicable.

## Profile Findings

The larger `HyperStudy` baseline profile shows the main issue clearly:

- `HyperStudy.fit`: 0.941 s total in the profiled run.
- `Study.fit`: 10 calls, 0.737 s cumulative.
- `ObservationModel.processedPdf`: 3600 calls, 0.417 s cumulative.
- `Gaussian.pdf`: 0.408 s cumulative.
- `GaussianRandomWalk.computeForwardPrior`: 3600 calls, 0.190 s cumulative.
- SciPy `gaussian_filter1d`/`correlate1d`: 0.187 s cumulative.

The cached prototype changes the shape of the remaining work:

- Total cached run: 0.569 s in the profiled run.
- Cached `fit_study_with_cached_likelihoods`: 10 calls, 0.322 s cumulative.
- `precompute_likelihoods`: 180 calls to `processedPdf`, 0.022 s cumulative.
- Transition filtering is now the largest remaining kernel: 0.203 s cumulative.

For `BivariateRandomWalk`, SciPy `convolve2d` dominates the profile. Likelihood caching only helped 1.11x in the quick case because the transition convolution is already the bottleneck.

## Likelihood Microbenchmarks

Quick-mode likelihood generation benchmarks:

| kernel | median s | speedup vs group baseline |
| --- | ---: | ---: |
| `gaussian_current_processedPdf_loop` | 0.0174 | 1.00x |
| `gaussian_numpy_invariant_loop` | 0.0074 | 2.35x |
| `gaussian_numpy_vectorized_all_time` | 0.0067 | 2.58x |
| `numba_gaussian_loop_compile_excluded` | 0.0090 | 1.92x |
| `poisson_current_processedPdf_loop` | 0.0446 | 1.00x |
| `poisson_unique_observation_cache` | 0.0017 | 25.84x |

Interpretation:

- Gaussian, Laplace, AR1, and ScaledAR1 should precompute grid-only terms such as variance, inverse variance, and log-normalization once per grid.
- Discrete observation models should cache likelihoods by unique data segment value. This is especially valuable for Poisson count data, where the number of unique observations is usually tiny compared with the number of time steps.
- Full vectorization over time can be fastest, but it materializes the same large likelihood cube as the general cache. It should be gated by a memory budget.

## Recommended Implementation Plan

Phase 1: Adaptive likelihood cache

- Add an internal likelihood cache path to `Study.fit`.
- API shape: `cacheLikelihoods="auto"`, `True`, or `False`, plus a maximum cache size.
- For `Study.fit`, use the cache for full forward-backward fits when it fits the memory budget. Skip it for plain `evidenceOnly` or `forwardOnly` unless a caller explicitly asks for it.
- For `HyperStudy.fit`, precompute likelihoods once and reuse them across hyperparameter values. This should also help `evidenceOnly` HyperStudy fits because observation likelihoods still repeat across hyperparameter settings.
- In multiprocessing HyperStudy, start with one cache per worker shard. Shared-memory caches can be investigated later if large hypergrids make memory pressure visible.
- Avoid repeating `movingWindow` inside every HyperStudy sub-fit.
- Keep the current streaming implementation as the fallback for large grids.

Phase 2: Observation-model grid preparation

- Add a private hook such as `ObservationModel.prepareGrid(grid)` or a small bound-kernel object built in `Study.setObservationModel`.
- Implement Gaussian-family invariant caches first.
- Implement unique-value likelihood caches for Poisson and Bernoulli. The cache key should be the formatted data segment and must handle missing data cleanly.
- Keep public observation model behavior unchanged.

Phase 3: Transition-model cleanup

- Bind transition models to the study once and cache target axis indices instead of resolving names on every time step.
- For `GaussianRandomWalk`, benchmark replacing repeated `gaussian_filter1d` calls with a cached 1D kernel plus `scipy.ndimage.correlate1d`. The profile shows kernel construction overhead exists, but the actual SciPy C correlation remains the main cost.
- For `BivariateRandomWalk`, focus on algorithm choice and SciPy kernel options rather than Python JIT. `convolve2d` dominates.

Phase 4: Optional compiler path

- Keep Numba optional, not required. The `speed` extra now requires `numba>=0.65`, matching the current Numba compatibility table for Python 3.10-3.14 and NumPy 2.0-2.4.
- Use Numba only for kernels that are naturally loop-based and not already faster as vectorized NumPy/SciPy.
- Do not start with CPython/Cython extensions. Cython typed memoryviews are useful for reducing Python indexing overhead in array loops, and the CPython C API supports native extension modules, but bayesloop's measured hot paths are mostly NumPy/SciPy kernels plus avoidable repeated work.

## Decision

The optimal route is:

1. Implement adaptive likelihood reuse in core `Study` and `HyperStudy`.
2. Add built-in observation-model fast paths using NumPy and small per-grid/per-data caches.
3. Re-profile transition models after those changes.
4. Use Numba selectively for custom/user-defined likelihood kernels after the internal kernel API is cleaner.
5. Defer Cython/CPython until profiling shows a remaining pure-Python inner loop that cannot be expressed well in NumPy/SciPy/Numba.

## External References

- Numba version support table: https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information
- Cython NumPy memoryview performance discussion: https://docs.cython.org/en/stable/src/userguide/numpy_tutorial.html#efficient-indexing-with-memoryviews
- Python extension module documentation: https://docs.python.org/3.12/extending/
