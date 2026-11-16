import hashlib
import os
from dataclasses import fields, is_dataclass
from functools import cache as once
from math import ceil
from time import perf_counter

from carbon.compiler import cache

SAMPLE_MS, WARM_MS = 20.0, 100.0
_CHOSEN = {}


def fastest(device, key, candidates, dispatches, *, margin=0.03, rounds=3):
    """The candidate whose `dispatches(candidate)` runs fastest on `device`. The first
    is the default, and another replaces it only when it is faster by more than
    `margin`. Choices persist in Carbon's cache directory; CARBON_SEARCH=0 takes the
    default."""
    candidates = list(candidates)
    if len(candidates) == 1 or os.environ.get("CARBON_SEARCH") == "0":
        return candidates[0]
    name = hashlib.sha256(repr((device.name, key, candidates)).encode()).hexdigest()
    if name not in _CHOSEN:
        index = cache.load("search", name)
        if type(index) is not int or not 0 <= index < len(candidates):
            index = _measure(
                device,
                [dispatches(candidate) for candidate in candidates],
                margin,
                rounds,
            )
            cache.store("search", name, index)
        _CHOSEN[name] = index
    return candidates[_CHOSEN[name]]


def choose(device, key, candidates, dispatches, trial):
    """`fastest` with each candidate's `dispatches(sample, candidate)` built on the
    sample call `trial()` makes, once and only when the search runs."""
    sample = once(trial)
    return fastest(
        device,
        key,
        candidates,
        lambda candidate: dispatches(sample(), candidate),
    )


def stable(value):
    """`value` with its functions and dataclasses named by module and qualified name,
    so a key built from it repeats from one run to the next."""
    if is_dataclass(value) and not isinstance(value, type):
        return type(value).__qualname__, *(
            stable(getattr(value, field.name)) for field in fields(value)
        )
    if isinstance(value, (tuple, list)):
        return tuple(stable(part) for part in value)
    if callable(value):
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{getattr(value, '__module__', '')}.{name}"
    return value


def _measure(device, steps, margin, rounds):
    executables = [device.prepare(dispatches) for dispatches in steps]
    first = [max(_time(executable, 1), 1e-3) for executable in executables]
    _time(executables[0], ceil(WARM_MS / first[0]))
    best = [float("inf")] * len(executables)
    for _ in range(rounds):
        for index, executable in enumerate(executables):
            best[index] = min(
                best[index],
                _time(executable, ceil(SAMPLE_MS / first[index])),
            )
    winner = min(range(len(best)), key=best.__getitem__)
    return winner if best[winner] < best[0] * (1 - margin) else 0


def _time(executable, repetitions):
    start = perf_counter()
    completion = executable.submit(repetitions)
    completion.wait()
    elapsed = (
        completion.gpu_ms
        if completion.gpu_ms is not None
        else (perf_counter() - start) * 1e3
    )
    return elapsed / repetitions
