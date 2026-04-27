#!/usr/bin/env python
"""
Performance harness for bayesloop v2 modernization work.

The goal of this script is not to replace a proper benchmark suite. It is a
focused analysis tool that measures representative bayesloop workloads and
isolates candidate speedups before changing package internals.

Examples:
    uv run --with-editable . python benchmarks/performance_analysis.py --quick --micro
    uv run --with-editable . python benchmarks/performance_analysis.py --quick --profile-case hyper_gaussian_random_walk_2d
    uv run --with-editable . --with numba python benchmarks/performance_analysis.py --quick --micro --numba-micro
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import math
import platform
import pstats
import statistics
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
from scipy.special import logsumexp

import bayesloop as bl
from bayesloop.core import HyperStudy
from bayesloop.preprocessing import movingWindow


StudyFactory = Callable[[], object]


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    description: str
    factory: StudyFactory
    default: bool = True


@dataclass
class FitSnapshot:
    log_evidence: float
    posterior_mean_checksum: float | None
    posterior_last_norm: float | None
    hyper_entropy: float | None = None


def _quiet(callable_: Callable[[], object]) -> object:
    with redirect_stdout(io.StringIO()):
        return callable_()


def _duration_stats(durations: list[float]) -> dict[str, float]:
    return {
        "min_s": min(durations),
        "median_s": statistics.median(durations),
        "max_s": max(durations),
        "mean_s": statistics.mean(durations),
    }


def _snapshot(study: object) -> FitSnapshot:
    posterior_mean_values = getattr(study, "posteriorMeanValues", None)
    if isinstance(posterior_mean_values, np.ndarray) and posterior_mean_values.size:
        posterior_mean_checksum = float(np.mean(posterior_mean_values))
    else:
        posterior_mean_checksum = None

    posterior_sequence = getattr(study, "posteriorSequence", None)
    if isinstance(posterior_sequence, np.ndarray) and posterior_sequence.size:
        posterior_last_norm = float(np.sum(posterior_sequence[-1]))
    else:
        posterior_last_norm = None

    hyper_distribution = getattr(study, "hyperParameterDistribution", None)
    if isinstance(hyper_distribution, np.ndarray) and hyper_distribution.size:
        positive = hyper_distribution[hyper_distribution > 0]
        hyper_entropy = float(-np.sum(positive * np.log(positive)))
    else:
        hyper_entropy = None

    return FitSnapshot(
        log_evidence=float(getattr(study, "logEvidence")),
        posterior_mean_checksum=posterior_mean_checksum,
        posterior_last_norm=posterior_last_norm,
        hyper_entropy=hyper_entropy,
    )


def _format_study_data(study: object) -> tuple[np.ndarray, np.ndarray]:
    formatted_data = movingWindow(study.rawData, study.observationModel.segmentLength)
    formatted_timestamps = study.rawTimestamps[study.observationModel.segmentLength - 1 :]
    return formatted_data, formatted_timestamps


def precompute_likelihoods(
    study: object,
    formatted_data: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Compute all observation likelihoods once for a fixed observation grid."""
    if formatted_data is None:
        formatted_data, _ = _format_study_data(study)

    likelihoods = np.empty([len(formatted_data)] + study.gridSize, dtype=float)
    for i, data_segment in enumerate(formatted_data):
        likelihood = study.observationModel.processedPdf(study.grid, data_segment)
        if likelihood.dtype == object:
            likelihood = likelihood.astype(float)
        likelihoods[i] = likelihood

    return likelihoods, likelihoods.nbytes / 2**20


def fit_study_with_cached_likelihoods(
    study: object,
    likelihoods: np.ndarray,
    formatted_data: np.ndarray,
    formatted_timestamps: np.ndarray,
    *,
    forward_only: bool = False,
    evidence_only: bool = False,
) -> None:
    """Prototype Study.fit equivalent that consumes a precomputed likelihood cube."""
    study._checkConsistency()
    study.formattedData = formatted_data
    study.formattedTimestamps = formatted_timestamps

    n_time = len(formatted_data)
    if not evidence_only:
        study.posteriorSequence = np.empty([n_time] + study.gridSize)

    study.logEvidence = 0
    study.localEvidence = np.empty(n_time)
    lattice_product = float(np.prod(study.latticeConstant))

    alpha = study._computePrior(silent=True)
    for i in range(n_time):
        alpha *= likelihoods[i]
        norm = np.sum(alpha)
        if norm <= 0.0:
            study.logEvidence = -np.inf
            return

        alpha /= norm
        study.logEvidence += np.log(norm)
        study.localEvidence[i] = norm * lattice_product

        if not evidence_only:
            study.posteriorSequence[i] = alpha

        alpha = study.transitionModel.computeForwardPrior(alpha, formatted_timestamps[i])

    study.logEvidence += np.log(lattice_product)

    if not (forward_only or evidence_only):
        beta = np.ones(study.gridSize)
        beta /= np.sum(beta)

        for i in range(n_time - 1, -1, -1):
            study.posteriorSequence[i] *= beta
            norm = np.sum(study.posteriorSequence[i])
            if norm <= 0.0:
                study.logEvidence = -np.inf
                return

            study.posteriorSequence[i] /= norm

            likelihood = likelihoods[i]
            with np.errstate(invalid="ignore", divide="ignore"):
                study.localEvidence[i] = 1.0 / (
                    np.sum(study.posteriorSequence[i] / likelihood) * lattice_product
                )

            beta = study.transitionModel.computeBackwardPrior(
                beta * likelihood,
                formatted_timestamps[i],
            )
            beta /= np.sum(beta)

    if evidence_only:
        study.posteriorMeanValues = []
    else:
        study.posteriorMeanValues = np.empty([len(study.grid), len(study.posteriorSequence)])
        for i in range(len(study.grid)):
            study.posteriorMeanValues[i] = np.array(
                [np.sum(posterior * study.grid[i]) for posterior in study.posteriorSequence]
            )


def fit_hyperstudy_with_cached_likelihoods(
    study: HyperStudy,
    *,
    forward_only: bool = False,
    evidence_only: bool = False,
) -> float:
    """Prototype single-process HyperStudy.fit with observation likelihood reuse."""
    study.fitWarningCounter = 0

    study.formattedData, study.formattedTimestamps = _format_study_data(study)
    study._createHyperGrid(silent=True)
    study._checkConsistency()

    likelihoods, cache_mib = precompute_likelihoods(study, study.formattedData)

    if not evidence_only:
        study.averagePosteriorSequence = np.zeros([len(study.formattedData)] + study.gridSize) - np.inf

    study.logEvidenceList = []
    study.localEvidenceList = []

    if len(study.hyperGridValues) <= 1:
        fit_study_with_cached_likelihoods(
            study,
            likelihoods,
            study.formattedData,
            study.formattedTimestamps,
            forward_only=forward_only,
            evidence_only=evidence_only,
        )
        return cache_mib

    for i, hyper_param_values in enumerate(study.hyperGridValues):
        study._setSelectedHyperParameters(hyper_param_values)
        fit_study_with_cached_likelihoods(
            study,
            likelihoods,
            study.formattedData,
            study.formattedTimestamps,
            forward_only=forward_only,
            evidence_only=evidence_only,
        )

        study.logEvidenceList.append(study.logEvidence)
        study.localEvidenceList.append(study.localEvidence.copy())

        if (not evidence_only) and np.isfinite(study.logEvidence):
            study.posteriorSequence[study.posteriorSequence < 10.0**-300] = 10.0**-300
            study.averagePosteriorSequence = np.logaddexp(
                study.averagePosteriorSequence,
                np.log(study.posteriorSequence)
                + study.logEvidence
                + np.log(study.flatHyperPriorValues[i]),
            )

    if not evidence_only:
        study.averagePosteriorSequence -= np.amax(study.averagePosteriorSequence)
        study.averagePosteriorSequence = np.exp(study.averagePosteriorSequence)

        normalization = np.array([np.sum(posterior) for posterior in study.averagePosteriorSequence])
        for _ in range(len(study.grid)):
            normalization = normalization[:, None]
        study.averagePosteriorSequence /= normalization
        study.posteriorSequence = study.averagePosteriorSequence

    log_hyper_parameter_distribution = (
        np.array(study.logEvidenceList)
        + np.log(study.flatHyperPriorValues)
        + np.sum(np.log(study.hyperGridConstant))
    )
    scaled = log_hyper_parameter_distribution - np.amax(log_hyper_parameter_distribution)
    study.hyperParameterDistribution = np.exp(scaled)
    study.hyperParameterDistribution /= np.sum(study.hyperParameterDistribution)
    study.hyperParameterDistribution /= np.prod(study.hyperGridConstant)

    study.logEvidence = logsumexp(log_hyper_parameter_distribution)
    study.localEvidence = np.sum(
        (np.array(study.localEvidenceList).T * study.flatHyperPriorValues).T,
        axis=0,
    )

    if not evidence_only:
        study.posteriorMeanValues = np.empty([len(study.grid), len(study.posteriorSequence)])
        for i in range(len(study.grid)):
            study.posteriorMeanValues[i] = np.array(
                [np.sum(posterior * study.grid[i]) for posterior in study.posteriorSequence]
            )

    study.localEvidenceList = []
    study._setAllHyperParameters(study.flatHyperParameters)
    return cache_mib


def fit_cached_total(study: object) -> tuple[FitSnapshot, float]:
    if isinstance(study, HyperStudy):
        cache_mib = fit_hyperstudy_with_cached_likelihoods(study)
    else:
        formatted_data, formatted_timestamps = _format_study_data(study)
        likelihoods, cache_mib = precompute_likelihoods(study, formatted_data)
        fit_study_with_cached_likelihoods(study, likelihoods, formatted_data, formatted_timestamps)

    return _snapshot(study), cache_mib


def run_baseline(case: BenchmarkCase) -> FitSnapshot:
    study = case.factory()
    _quiet(lambda: study.fit(silent=True))
    return _snapshot(study)


def run_cached(case: BenchmarkCase) -> tuple[FitSnapshot, float]:
    study = case.factory()
    return _quiet(lambda: fit_cached_total(study))


def time_case(case: BenchmarkCase, runner: Callable[[BenchmarkCase], object], repeats: int) -> tuple[list[float], object]:
    durations: list[float] = []
    last_result = None
    for _ in range(repeats):
        start = time.perf_counter()
        last_result = runner(case)
        durations.append(time.perf_counter() - start)
    return durations, last_result


def compare_snapshots(baseline: FitSnapshot, cached: FitSnapshot) -> dict[str, float | None]:
    return {
        "log_evidence_abs_diff": abs(baseline.log_evidence - cached.log_evidence),
        "posterior_mean_checksum_abs_diff": (
            None
            if baseline.posterior_mean_checksum is None or cached.posterior_mean_checksum is None
            else abs(baseline.posterior_mean_checksum - cached.posterior_mean_checksum)
        ),
        "posterior_last_norm_abs_diff": (
            None
            if baseline.posterior_last_norm is None or cached.posterior_last_norm is None
            else abs(baseline.posterior_last_norm - cached.posterior_last_norm)
        ),
        "hyper_entropy_abs_diff": (
            None
            if baseline.hyper_entropy is None or cached.hyper_entropy is None
            else abs(baseline.hyper_entropy - cached.hyper_entropy)
        ),
    }


def make_cases(quick: bool) -> list[BenchmarkCase]:
    poisson_n, poisson_grid = (900, 1200) if quick else (2500, 2500)
    gaussian_n, gaussian_grid = (120, 80) if quick else (300, 140)
    hyper_n, hyper_grid, hyper_values = (90, 58, 6) if quick else (180, 95, 10)
    ar_n, ar_grid = (450, 58) if quick else (1300, 95)
    bivar_n, bivar_grid = (60, 48) if quick else (130, 80)

    def poisson_static() -> object:
        rng = np.random.default_rng(101)
        data = rng.poisson(2.4, size=poisson_n).astype(float)
        study = bl.Study(silent=True)
        study.loadData(data, silent=True)
        study.setOM(bl.om.Poisson("rate", bl.oint(0, 8, poisson_grid)), silent=True)
        study.setTM(bl.tm.Static(), silent=True)
        return study

    def gaussian_random_walk_2d() -> object:
        rng = np.random.default_rng(202)
        data = rng.normal(loc=0.15, scale=0.9, size=gaussian_n)
        study = bl.Study(silent=True)
        study.loadData(data, silent=True)
        study.setOM(
            bl.om.Gaussian(
                "mean",
                bl.cint(-2.2, 2.2, gaussian_grid),
                "std",
                bl.oint(0.25, 2.5, gaussian_grid),
            ),
            silent=True,
        )
        study.setTM(bl.tm.GaussianRandomWalk("sigma", 0.08, target="mean"), silent=True)
        return study

    def hyper_gaussian_random_walk_2d() -> object:
        rng = np.random.default_rng(303)
        data = rng.normal(loc=0.1, scale=1.0, size=hyper_n)
        study = bl.HyperStudy(silent=True)
        study.loadData(data, silent=True)
        study.setOM(
            bl.om.Gaussian(
                "mean",
                bl.cint(-2.5, 2.5, hyper_grid),
                "std",
                bl.oint(0.25, 2.8, hyper_grid),
            ),
            silent=True,
        )
        study.setTM(
            bl.tm.GaussianRandomWalk("sigma", bl.cint(0.0, 0.22, hyper_values), target="mean"),
            silent=True,
        )
        return study

    def ar1_static() -> object:
        rng = np.random.default_rng(404)
        data = np.empty(ar_n)
        data[0] = rng.normal()
        for i in range(1, ar_n):
            data[i] = 0.72 * data[i - 1] + 0.65 * rng.normal()
        study = bl.Study(silent=True)
        study.loadData(data, silent=True)
        study.setOM(
            bl.om.AR1(
                "rho",
                bl.oint(-0.98, 0.98, ar_grid),
                "noise",
                bl.oint(0.15, 1.8, ar_grid),
            ),
            silent=True,
        )
        study.setTM(bl.tm.Static(), silent=True)
        return study

    def bivariate_random_walk() -> object:
        rng = np.random.default_rng(505)
        data = rng.normal(loc=0.0, scale=1.0, size=bivar_n)
        study = bl.Study(silent=True)
        study.loadData(data, silent=True)
        study.setOM(
            bl.om.Gaussian(
                "mean",
                bl.cint(-2.2, 2.2, bivar_grid),
                "std",
                bl.oint(0.25, 2.5, bivar_grid),
            ),
            silent=True,
        )
        study.setTM(
            bl.tm.BivariateRandomWalk(
                "sigma_mean",
                0.08,
                "sigma_std",
                0.04,
                "rho",
                0.0,
            ),
            silent=True,
        )
        return study

    return [
        BenchmarkCase(
            "poisson_static_1d",
            "1D Poisson observation model with Static transition; likelihood work dominates.",
            poisson_static,
        ),
        BenchmarkCase(
            "gaussian_random_walk_2d",
            "2D Gaussian observation model with GaussianRandomWalk on the mean axis.",
            gaussian_random_walk_2d,
        ),
        BenchmarkCase(
            "hyper_gaussian_random_walk_2d",
            "HyperStudy over GaussianRandomWalk sigma values; repeated likelihoods are expected.",
            hyper_gaussian_random_walk_2d,
        ),
        BenchmarkCase(
            "ar1_static_2d",
            "2D AR1 likelihood with overlapping data windows and Static transition.",
            ar1_static,
        ),
        BenchmarkCase(
            "bivariate_random_walk_2d",
            "2D Gaussian observation model with BivariateRandomWalk convolution.",
            bivariate_random_walk,
            default=False,
        ),
    ]


def profile_case(case: BenchmarkCase, top: int) -> str:
    profiler = cProfile.Profile()

    def run() -> FitSnapshot:
        return run_baseline(case)

    profiler.enable()
    run()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime")
    stats.print_stats(top)
    return stream.getvalue()


def _time_callable(name: str, callable_: Callable[[], np.ndarray], repeats: int) -> dict[str, object]:
    durations: list[float] = []
    checksum = None
    shape = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = callable_()
        durations.append(time.perf_counter() - start)
        checksum = float(np.mean(output))
        shape = list(output.shape)
    return {
        "name": name,
        "shape": shape,
        "checksum": checksum,
        **_duration_stats(durations),
    }


def run_likelihood_microbenchmarks(quick: bool, repeats: int, include_numba: bool) -> list[dict[str, object]]:
    n_time, grid_size = (180, 85) if quick else (500, 150)
    rng = np.random.default_rng(707)
    data = rng.normal(loc=0.1, scale=0.9, size=n_time)

    gaussian_study = bl.Study(silent=True)
    gaussian_study.loadData(data, silent=True)
    gaussian_study.setOM(
        bl.om.Gaussian(
            "mean",
            bl.cint(-2.5, 2.5, grid_size),
            "std",
            bl.oint(0.25, 2.8, grid_size),
        ),
        silent=True,
    )
    gaussian_study.setTM(bl.tm.Static(), silent=True)
    gaussian_data, _ = _format_study_data(gaussian_study)
    mean_grid, std_grid = gaussian_study.grid

    def gaussian_current_loop() -> np.ndarray:
        return precompute_likelihoods(gaussian_study, gaussian_data)[0]

    def gaussian_invariant_loop() -> np.ndarray:
        output = np.empty([len(gaussian_data)] + gaussian_study.gridSize, dtype=float)
        std2 = std_grid * std_grid
        inv_two_std2 = 1.0 / (2.0 * std2)
        log_norm = -0.5 * np.log(2.0 * np.pi * std2)
        for i, data_segment in enumerate(gaussian_data):
            np.exp(-((data_segment[0] - mean_grid) ** 2.0) * inv_two_std2 + log_norm, out=output[i])
        return output

    def gaussian_vectorized_all() -> np.ndarray:
        std2 = std_grid * std_grid
        inv_two_std2 = 1.0 / (2.0 * std2)
        log_norm = -0.5 * np.log(2.0 * np.pi * std2)
        x = gaussian_data[:, 0]
        return np.exp(-((x[:, None, None] - mean_grid[None, :, :]) ** 2.0) * inv_two_std2[None, :, :] + log_norm)

    benchmarks = [
        _time_callable("gaussian_current_processedPdf_loop", gaussian_current_loop, repeats),
        _time_callable("gaussian_numpy_invariant_loop", gaussian_invariant_loop, repeats),
        _time_callable("gaussian_numpy_vectorized_all_time", gaussian_vectorized_all, repeats),
    ]

    poisson_n, poisson_grid_size = (2500, 1800) if quick else (8000, 5000)
    poisson_data_values = rng.poisson(2.5, size=poisson_n).astype(float)
    poisson_study = bl.Study(silent=True)
    poisson_study.loadData(poisson_data_values, silent=True)
    poisson_study.setOM(bl.om.Poisson("rate", bl.oint(0, 8, poisson_grid_size)), silent=True)
    poisson_study.setTM(bl.tm.Static(), silent=True)
    poisson_data, _ = _format_study_data(poisson_study)

    def poisson_current_loop() -> np.ndarray:
        return precompute_likelihoods(poisson_study, poisson_data)[0]

    def poisson_unique_value_cache() -> np.ndarray:
        values = poisson_data[:, 0].astype(int)
        unique_values, inverse = np.unique(values, return_inverse=True)
        cache = np.empty([len(unique_values)] + poisson_study.gridSize, dtype=float)
        for i, value in enumerate(unique_values):
            cache[i] = poisson_study.observationModel.processedPdf(
                poisson_study.grid,
                np.array([value], dtype=float),
            )
        return cache[inverse]

    benchmarks.extend(
        [
            _time_callable("poisson_current_processedPdf_loop", poisson_current_loop, repeats),
            _time_callable("poisson_unique_observation_cache", poisson_unique_value_cache, repeats),
        ]
    )

    if include_numba:
        benchmarks.extend(_run_numba_microbenchmarks(gaussian_data, mean_grid, std_grid, repeats))

    return benchmarks


def _run_numba_microbenchmarks(
    formatted_data: np.ndarray,
    mean_grid: np.ndarray,
    std_grid: np.ndarray,
    repeats: int,
) -> list[dict[str, object]]:
    try:
        import numba as nb
    except ImportError:
        return [
            {
                "name": "numba_gaussian_loop",
                "skipped": "numba is not installed",
            }
        ]

    @nb.njit(cache=True)
    def gaussian_numba_loop(data: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        n = data.shape[0]
        rows = means.shape[0]
        cols = means.shape[1]
        output = np.empty((n, rows, cols), dtype=np.float64)
        two_pi = 2.0 * math.pi
        for t in range(n):
            x = data[t, 0]
            for i in range(rows):
                for j in range(cols):
                    sigma = stds[i, j]
                    sigma2 = sigma * sigma
                    exponent = -((x - means[i, j]) * (x - means[i, j])) / (2.0 * sigma2)
                    output[t, i, j] = math.exp(exponent - 0.5 * math.log(two_pi * sigma2))
        return output

    # Compile before measuring.
    gaussian_numba_loop(formatted_data, mean_grid, std_grid)

    return [
        _time_callable(
            "numba_gaussian_loop_compile_excluded",
            lambda: gaussian_numba_loop(formatted_data, mean_grid, std_grid),
            repeats,
        )
    ]


def run_benchmarks(args: argparse.Namespace) -> dict[str, object]:
    cases = make_cases(args.quick)
    selected_names = set(args.case or [])
    if selected_names:
        cases = [case for case in cases if case.name in selected_names]
    elif not args.all_cases:
        cases = [case for case in cases if case.default]

    results: dict[str, object] = {
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "quick": args.quick,
        "repeats": args.repeats,
        "cases": [],
        "microbenchmarks": [],
        "profiles": {},
    }

    for case in cases:
        baseline_durations, baseline_snapshot = time_case(case, run_baseline, args.repeats)
        cached_durations, cached_result = time_case(case, run_cached, args.repeats)
        cached_snapshot, cache_mib = cached_result
        baseline_snapshot = baseline_snapshot

        case_result = {
            "name": case.name,
            "description": case.description,
            "baseline": _duration_stats(baseline_durations),
            "cached_likelihoods": _duration_stats(cached_durations),
            "speedup": statistics.median(baseline_durations) / statistics.median(cached_durations),
            "likelihood_cache_mib": cache_mib,
            "validation": compare_snapshots(baseline_snapshot, cached_snapshot),
        }
        results["cases"].append(case_result)

    if args.micro:
        results["microbenchmarks"] = run_likelihood_microbenchmarks(
            args.quick,
            args.repeats,
            args.numba_micro,
        )

    for profile_name in args.profile_case or []:
        matched = [case for case in make_cases(args.quick) if case.name == profile_name]
        if not matched:
            raise SystemExit(f"Unknown --profile-case value: {profile_name}")
        results["profiles"][profile_name] = profile_case(matched[0], args.profile_top)

    return results


def print_markdown(results: dict[str, object]) -> None:
    env = results["environment"]
    print("# bayesloop performance analysis")
    print()
    print(f"- Python: `{env['python'].split()[0]}`")
    print(f"- Platform: `{env['platform']}`")
    print(f"- NumPy: `{env['numpy']}`")
    print(f"- SciPy: `{env['scipy']}`")
    print(f"- Quick mode: `{results['quick']}`")
    print(f"- Repeats: `{results['repeats']}`")
    print()

    print("## Fit benchmarks")
    print()
    print("| case | baseline median s | cached median s | speedup | cache MiB | log evidence diff |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for case in results["cases"]:
        validation = case["validation"]
        print(
            "| {name} | {base:.4f} | {cached:.4f} | {speedup:.2f}x | {cache:.1f} | {diff:.3e} |".format(
                name=case["name"],
                base=case["baseline"]["median_s"],
                cached=case["cached_likelihoods"]["median_s"],
                speedup=case["speedup"],
                cache=case["likelihood_cache_mib"],
                diff=validation["log_evidence_abs_diff"],
            )
        )
    print()

    if results["microbenchmarks"]:
        print("## Likelihood microbenchmarks")
        print()
        print("| name | median s | speed vs group baseline | checksum |")
        print("| --- | ---: | ---: | ---: |")
        peer_baselines: dict[str, float] = {}
        for row in results["microbenchmarks"]:
            if row.get("skipped"):
                print(f"| {row['name']} | skipped: {row['skipped']} |  |  |")
                continue
            if row["name"].startswith(("gaussian_", "numba_gaussian")):
                group = "gaussian"
            elif row["name"].startswith("poisson_"):
                group = "poisson"
            else:
                group = row["name"]
            peer_baselines.setdefault(group, row["median_s"])
            speed = peer_baselines[group] / row["median_s"]
            print(
                "| {name} | {median:.4f} | {speed:.2f}x | {checksum:.6g} |".format(
                    name=row["name"],
                    median=row["median_s"],
                    speed=speed,
                    checksum=row["checksum"],
                )
            )
        print()

    for name, profile in results["profiles"].items():
        print(f"## cProfile: {name}")
        print()
        print("```")
        print(profile.rstrip())
        print("```")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use smaller inputs suitable for fast local iteration.")
    parser.add_argument("--all-cases", action="store_true", help="Run all cases, including heavier non-default cases.")
    parser.add_argument("--case", action="append", help="Run only the named case. Can be supplied more than once.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed repeats per benchmark.")
    parser.add_argument("--micro", action="store_true", help="Run observation-likelihood microbenchmarks.")
    parser.add_argument("--numba-micro", action="store_true", help="Include optional Numba likelihood microbenchmark.")
    parser.add_argument("--profile-case", action="append", help="Run cProfile for the named case.")
    parser.add_argument("--profile-top", type=int, default=30, help="Number of cProfile rows to print.")
    parser.add_argument("--json", type=Path, help="Write raw benchmark results to this JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmarks(args)
    print_markdown(results)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
