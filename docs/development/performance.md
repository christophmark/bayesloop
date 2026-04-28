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

`Study.fit` previously evaluated the same likelihood once in the forward pass and again in the backward pass. `HyperStudy.fit` amplified that cost by running the same observation likelihoods once per hyperparameter value. The implemented cached-likelihood path reproduces baseline results exactly and still delivers 1.21x to 1.50x speedups for larger continuous-model cases after the observation and transition models themselves were optimized.

This first optimization is now implemented in core via `fit(..., cache_likelihoods="auto", max_cache_size=512)`.
`cache_likelihoods="auto"` is the default and caches full forward-backward fits only when the estimated likelihood
cache is below the memory budget. Passing `cache_likelihoods=True` forces the cache, while `False` preserves the
previous on-demand behavior.

The second-best speedup is model-specific NumPy optimization in built-in observation models. This is now implemented for the Gaussian-family built-ins via prepared grid invariants and for Bernoulli/Poisson via small repeated-value caches. Poisson and Bernoulli opt out of the full time-by-grid sequence cache in `auto` mode, because their own cache is smaller and faster for repeated discrete observations.

Numba is useful as an optional tool for custom loop-heavy likelihood kernels, but it should not be the default route for the core built-ins. After the Gaussian grid-invariant cache, the measured Numba Gaussian loop was slower than the built-in NumPy path. Most transition-model time is already inside SciPy C kernels, so Numba has little to compile there.

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

These timings compare `cache_likelihoods=False` with the implemented default `cache_likelihoods="auto"` after the observation-model fast paths. The Poisson case reports `0.0` MiB because auto mode uses the model's repeated-value cache instead of a full time-by-grid likelihood cube.

| case | baseline median s | auto median s | speedup | sequence cache MiB | validation |
| --- | ---: | ---: | ---: | ---: | --- |
| `poisson_static_1d` | 0.1201 | 0.0979 | 1.23x | 0.0 | exact |
| `gaussian_random_walk_2d` | 0.3046 | 0.2527 | 1.21x | 44.9 | exact |
| `hyper_gaussian_random_walk_2d` | 1.2834 | 0.8847 | 1.45x | 12.4 | exact |
| `ar1_static_2d` | 0.4092 | 0.2732 | 1.50x | 89.4 | exact |
| `bivariate_random_walk_2d` quick | 0.0236 | 0.0212 | 1.11x | 1.1 | exact |

Validation means zero measured difference in log evidence, posterior mean checksum, posterior final normalization, and HyperStudy entropy where applicable.

## Profile Findings

The larger `HyperStudy` baseline profile shows the main issue clearly:

- `HyperStudy.fit`: 0.894 s total in the profiled run.
- `_fit_formatted_data`: 10 calls, 0.695 s cumulative.
- `ObservationModel.processed_pdf`: 3600 calls, 0.407 s cumulative.
- `Gaussian.pdf`: 0.399 s cumulative.
- `GaussianRandomWalk.compute_forward_prior`: 3600 calls, 0.180 s cumulative.
- SciPy `gaussian_filter1d`/`correlate1d`: 0.177 s cumulative.

The cached implementation changes the shape of the remaining work:

- Total cached run: 0.533 s in the profiled run.
- `_fit_formatted_data`: 10 calls, 0.297 s cumulative.
- `_compute_likelihood_sequence`: 180 calls to `processed_pdf`, 0.021 s cumulative.
- Transition filtering is now the largest remaining kernel: 0.186 s cumulative.

For `BivariateRandomWalk`, SciPy `convolve2d` dominates the profile. Likelihood caching only helped 1.11x in the quick case because the transition convolution is already the bottleneck.

## Likelihood Microbenchmarks

Quick-mode likelihood generation benchmarks after built-in observation-model caches:

| kernel | median s | speedup vs group baseline |
| --- | ---: | ---: |
| `gaussian_current_processed_pdf_loop` | 0.0120 | 1.00x |
| `gaussian_numpy_invariant_loop` | 0.0111 | 1.08x |
| `gaussian_numpy_vectorized_all_time` | 0.0097 | 1.24x |
| `numba_gaussian_loop_compile_excluded` | 0.0159 | 0.75x |
| `poisson_current_processed_pdf_loop` | 0.0098 | 1.00x |
| `poisson_unique_observation_cache` | 0.0018 | 5.37x |

Interpretation:

- Gaussian, Laplace, WhiteNoise, AR1, and ScaledAR1 now precompute grid-only terms such as variance, inverse variance, and log-normalization once per grid.
- Bernoulli and Poisson now cache likelihoods by unique data segment value. This is especially valuable for Poisson count data, where the number of unique observations is usually tiny compared with the number of time steps.
- Full vectorization over time can be fastest, but it materializes the same large likelihood cube as the general cache. It should be gated by a memory budget.

## Recommended Implementation Plan

Phase 1: Adaptive likelihood cache (implemented)

- `Study.fit` has an internal likelihood cache path.
- API shape: `cache_likelihoods="auto"`, `True`, or `False`, plus `max_cache_size` in MiB.
- `Study.fit` uses the cache for full forward-backward fits when it fits the memory budget.
- `HyperStudy.fit` precomputes likelihoods once and reuses them across hyperparameter values.
- Multiprocessing HyperStudy builds one cache per worker shard. Shared-memory caches can be investigated later if large hypergrids make memory pressure visible.
- `HyperStudy.fit` avoids repeating `moving_window` inside every sub-fit.
- The previous streaming implementation remains available through `cache_likelihoods=False` and as the automatic fallback for large grids.

Phase 2: Observation-model grid preparation (implemented)

- `ObservationModel.prepare_grid(grid)` prepares model-specific grid caches from `Study.set_observation_model`.
- Gaussian-family invariant caches are implemented for `Gaussian`, `Laplace`, `WhiteNoise`, `AR1`, and `ScaledAR1`.
- Unique-value likelihood caches are implemented for `Poisson` and `Bernoulli`.
- Public observation model behavior is unchanged.

Phase 3: Transition-model cleanup (partially implemented)

- Bind transition models to the study once and cache target axis indices instead of resolving names on every time step.
- `GaussianRandomWalk` now replaces repeated `gaussian_filter1d` kernel construction with cached 1D kernels plus `scipy.ndimage.correlate1d`. Kernels are cached per normalized sigma and target axis, which also avoids rebuilding kernels during `OnlineStudy` hyperparameter scans.
- For `BivariateRandomWalk`, focus on algorithm choice and SciPy kernel options rather than Python JIT. `convolve2d` dominates.

Phase 4: Optional compiler path

- Keep Numba optional, not required. The `speed` extra now requires `numba>=0.65`, matching the current Numba compatibility table for Python 3.10-3.14 and NumPy 2.0-2.4.
- Use Numba only for kernels that are naturally loop-based and not already faster as vectorized NumPy/SciPy.
- Do not start with CPython/Cython extensions. Cython typed memoryviews are useful for reducing Python indexing overhead in array loops, and the CPython C API supports native extension modules, but bayesloop's measured hot paths are mostly NumPy/SciPy kernels plus avoidable repeated work.

## Decision

The remaining optimal route is:

1. Re-profile `OnlineStudy` with large transition-model hypergrids after the per-sigma `GaussianRandomWalk` kernel cache.
2. Investigate `BivariateRandomWalk` kernel choices.
3. Use Numba selectively for custom/user-defined likelihood kernels after the internal kernel API is cleaner.
4. Defer Cython/CPython until profiling shows a remaining pure-Python inner loop that cannot be expressed well in NumPy/SciPy/Numba.

## External References

- Numba version support table: https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information
- Cython NumPy memoryview performance discussion: https://docs.cython.org/en/stable/src/userguide/numpy_tutorial.html#efficient-indexing-with-memoryviews
- Python extension module documentation: https://docs.python.org/3.12/extending/
