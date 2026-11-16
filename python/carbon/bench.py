"""Correctness-gated measurements for externally defined experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Protocol


class Completion(Protocol):
    def wait(self) -> None:
        """Block until the work completes."""

    @property
    def gpu_ms(self) -> float | None:
        """GPU milliseconds of the work, or None when the device does not report
        them."""


@dataclass(frozen=True)
class Prepared:
    """A case ready to time: `submit(n)` runs it n times over, `reset()` restores its
    inputs, and `verify(n)` checks the results of n runs; `metadata` is recorded with
    them."""

    submit: Callable[[int], Completion]
    reset: Callable[[], None]
    verify: Callable[[int], None]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Case:
    """One named workload of an experiment; `prepare()` builds it once and returns its
    Prepared form."""

    name: str
    prepare: Callable[[], Prepared]

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or not self.name
            or not callable(self.prepare)
        ):
            raise ValueError("a case needs a nonempty name and a preparation function")


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(path: Path) -> list[Case]:
    spec = importlib.util.spec_from_file_location("carbon_experiment", path)
    if spec is None or spec.loader is None:
        raise ValueError("experiment must be a Python file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cases = list(module.cases())
    if not cases or any(not isinstance(case, Case) for case in cases):
        raise ValueError("cases() must return Carbon Case objects")
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("case names must be unique")
    return cases


def git_value(path: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def environment(path: Path) -> dict[str, Any]:
    from . import __version__

    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "carbon": __version__,
        "experiment": str(path),
        "experiment_sha256": fingerprint(path),
        "harness_sha256": fingerprint(Path(__file__)),
        "experiment_git_revision": git_value(path.parent, "rev-parse", "HEAD"),
        "experiment_git_status": git_value(path.parent, "status", "--porcelain"),
    }
    try:
        from . import _metal

        result["native_path"] = _metal.__file__
        result["native_sha256"] = fingerprint(Path(_metal.__file__))
    except ImportError:
        result["native_path"] = None
    try:
        from . import _compiler

        result["compiler_path"] = _compiler.__file__
        result["compiler_sha256"] = fingerprint(Path(_compiler.__file__))
        result["llvm_version"] = _compiler.llvm_version
        result["toolchain_id"] = _compiler.toolchain_id
        result["build_id"] = _compiler.build_id
    except ImportError:
        result["compiler_path"] = None
    return result


def sample(prepared: Prepared, repetitions: int) -> dict[str, float | None]:
    """One verified timing: submission and wall time on the host, and the GPU interval,
    in microseconds."""
    prepared.reset()
    start = perf_counter_ns()
    completion = prepared.submit(repetitions)
    submitted = perf_counter_ns()
    completion.wait()
    ended = perf_counter_ns()
    prepared.verify(repetitions)
    gpu_ms = completion.gpu_ms
    if gpu_ms is not None and (not math.isfinite(gpu_ms) or gpu_ms < 0):
        raise ValueError("invalid GPU interval")
    return {
        "submit_us": (submitted - start) / 1000,
        "wall_us": (ended - start) / 1000,
        "gpu_interval_us": None if gpu_ms is None else gpu_ms * 1000,
    }


def summarize(samples: list[dict], repetitions: int) -> dict[str, dict]:
    result = {}
    for metric in ("submit_us", "wall_us", "gpu_interval_us"):
        values = sorted(row[metric] for row in samples if row[metric] is not None)
        if values:
            median = statistics.median(values)
            result[metric] = {
                "count": len(values),
                "median": median,
                "p95": values[math.ceil(0.95 * len(values)) - 1],
                "min": values[0],
                "max": values[-1],
                "amortized_median_per_iteration": median / repetitions,
            }
    return result


def run(
    cases: list[Case],
    *,
    blocks: int = 5,
    samples: int = 10,
    warmup: int = 3,
    repetitions: int = 1,
    seed: int = 0,
) -> dict:
    """Prepare, run once and warm every case, then time them in shuffled order, block by
    block. The report holds every sample, a summary per case, and each case's per-block
    wall-time ratio to the first case with a bootstrap 95% interval; a failure is
    recorded with its case and phase, and leaves `correct` False."""
    if min(blocks, samples, repetitions) < 1 or warmup < 0:
        raise ValueError(
            "blocks, samples and repetitions must be positive; warmup must be "
            "nonnegative",
        )
    if not cases or len({case.name for case in cases}) != len(cases):
        raise ValueError("cases must be nonempty and uniquely named")
    report = {
        "schema": 1,
        "correct": False,
        "authoritative": False,
        "config": dict(
            blocks=blocks,
            samples=samples,
            warmup=warmup,
            repetitions=repetitions,
            seed=seed,
        ),
        "cases": {},
        "block_order": [],
        "sample_order": [],
    }
    prepared = {}
    phase = "prepare"
    current = ""
    try:
        for case in cases:
            current = case.name
            start = perf_counter_ns()
            ready = case.prepare()
            if not isinstance(ready, Prepared):
                raise TypeError("prepare must return Prepared")
            setup_ms = (perf_counter_ns() - start) / 1e6
            prepared[current] = ready
            record = {"setup_ms": setup_ms, "metadata": ready.metadata, "samples": []}
            report["cases"][current] = record
            phase = "first execution"
            record["first_execution"] = sample(ready, repetitions)
            phase = "warmup"
            for _ in range(warmup):
                sample(ready, repetitions)
            phase = "prepare"
        phase = "measurement"
        rng = random.Random(seed)
        for block in range(blocks):
            order = list(prepared)
            rng.shuffle(order)
            report["block_order"].append(order.copy())
            for trial in range(samples):
                if trial:
                    rng.shuffle(order)
                report["sample_order"].append(
                    {"block": block, "trial": trial, "cases": order.copy()},
                )
                for current in order:
                    record = report["cases"][current]
                    record["samples"].append(
                        {"block": block, **sample(prepared[current], repetitions)},
                    )
        for record in report["cases"].values():
            record["summary"] = summarize(record["samples"], repetitions)
        baseline = cases[0].name
        report["baseline"] = baseline
        report["paired_wall_ratios"] = {}
        for case in cases[1:]:
            ratios = []
            for block in range(blocks):
                candidate = [
                    row["wall_us"]
                    for row in report["cases"][case.name]["samples"]
                    if row["block"] == block
                ]
                reference = [
                    row["wall_us"]
                    for row in report["cases"][baseline]["samples"]
                    if row["block"] == block
                ]
                ratios.append(
                    statistics.median(candidate) / statistics.median(reference),
                )
            bootstrap = sorted(
                statistics.median(rng.choices(ratios, k=len(ratios)))
                for _ in range(2000)
            )
            report["paired_wall_ratios"][case.name] = {
                "blocks": ratios,
                "median": statistics.median(ratios),
                "block_bootstrap_ci95": [bootstrap[49], bootstrap[1949]],
            }
        report["correct"] = True
    except Exception as failure:
        report["failure"] = {
            "case": current,
            "phase": phase,
            "type": type(failure).__name__,
            "message": str(failure),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--note", default="occupancy not independently verified")
    args = parser.parse_args()
    path = args.experiment.resolve()
    try:
        provenance = environment(path)
        report = run(
            load_cases(path),
            blocks=args.blocks,
            samples=args.samples,
            warmup=args.warmup,
            repetitions=args.repetitions,
            seed=args.seed,
        )
        report["environment"] = provenance
        report["note"] = args.note
    except Exception as failure:
        report = {
            "schema": 1,
            "correct": False,
            "authoritative": False,
            "failure": {"type": type(failure).__name__, "message": str(failure)},
        }
    try:
        encoded = json.dumps(report, indent=2, allow_nan=False)
    except (TypeError, ValueError) as failure:
        report = {
            "schema": 1,
            "correct": False,
            "authoritative": False,
            "failure": {"type": type(failure).__name__, "message": str(failure)},
        }
        encoded = json.dumps(report, indent=2)
    print(encoded)
    return 0 if report["correct"] else 1


if __name__ == "__main__":
    from carbon.bench import main as entry

    raise SystemExit(entry())
