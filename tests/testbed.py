"""Shared test harness: storage conversion, guarded execution, and comparisons.

Every run surrounds each output with guard bytes and checks that inputs are unchanged,
so a kernel that writes outside its binding or into its inputs fails even when the
visible results are right.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("CARBON_SEARCH", "0")

GUARD = 16
FILL = 0xA5
NUMPY = {
    "f32": np.float32,
    "f16": np.float16,
    "bf16": np.uint16,
    "u32": np.uint32,
    "i32": np.uint32,
    "ui8": np.uint8,
    "si8": np.int8,
    "ui16": np.uint16,
    "si16": np.int16,
}
_devices = []


def device():
    pytest.importorskip("carbon._compiler")
    if not _devices:
        from carbon import Device

        _devices.append(Device())
    return _devices[0]


def bf16_bits(values):
    """Carbon's f32 -> bf16 rounding: nearest-even, NaN kept quiet with its upper
    payload bits."""
    words = np.asarray(values, np.float32).view(np.uint32)
    rounded = (words + 0x7FFF + ((words >> 16) & 1)) >> 16
    quiet = (words >> 16) | 0x40
    return np.where((words & 0x7FFFFFFF) > 0x7F800000, quiet, rounded).astype(np.uint16)


def bf16_values(bits):
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def storage(values, dtype):
    """Values as stored in a buffer of `dtype` (bf16 as raw bits)."""
    name = dtype if isinstance(dtype, str) else dtype.name
    if name == "bf16":
        return bf16_bits(values)
    return np.asarray(values).astype(NUMPY[name])


def widen(array, dtype):
    """Stored values as f32 (u32 stays integral)."""
    name = dtype if isinstance(dtype, str) else dtype.name
    if name == "bf16":
        return bf16_values(array)
    with np.errstate(invalid="ignore"):
        return np.asarray(array).astype(
            np.uint32 if name in ("u32", "i32") else np.float32,
        )


def upload(metal, array):
    buffer = metal.buffer(max(4, array.nbytes))
    buffer.write(np.ascontiguousarray(array).tobytes())
    return buffer


def run(
    kernel,
    *,
    inputs=(),
    outputs=(),
    constants=None,
    scalars=None,
    threads=32,
    grid=(1, 1, 1),
    **options,
):
    """Compile, dispatch once, and return output arrays; `outputs` are (count, numpy
    dtype) pairs.

    `options` pass through to `Kernel.compile`, as `max_scratch`, `views`, `ranges` or
    `min_bytes`.
    """
    artifact = kernel.compile(
        constants=constants or {},
        threads=threads,
        device=device(),
        **options,
    )
    return run_compiled(
        artifact,
        inputs=inputs,
        outputs=outputs,
        scalars=scalars,
        grid=grid,
    )


def run_compiled(artifact, *, inputs=(), outputs=(), scalars=None, grid=(1, 1, 1)):
    metal = device()
    input_buffers = []
    for array in inputs:
        payload = np.ascontiguousarray(array).tobytes()
        buffer = metal.buffer(max(len(payload), 4))
        buffer.write(payload)
        input_buffers.append((buffer, payload))
    output_buffers = []
    for count, dtype in outputs:
        size = max(count * np.dtype(dtype).itemsize, 4)
        buffer = metal.buffer(size + 2 * GUARD)
        buffer.write(bytes([FILL]) * buffer.size)
        output_buffers.append((buffer, size, count, dtype))
    bindings = [buffer for buffer, _ in input_buffers]
    bindings += [buffer.view(GUARD, size) for buffer, size, _, _ in output_buffers]
    dispatch = artifact.dispatch(metal, bindings, grid=grid, scalars=scalars or {})
    metal.prepare([dispatch]).submit().wait()
    results = []
    for buffer, size, count, dtype in output_buffers:
        raw = buffer.read()
        assert raw[:GUARD] == raw[GUARD + size :] == bytes([FILL]) * GUARD, (
            "write outside the output binding"
        )
        results.append(
            np.frombuffer(
                raw[GUARD : GUARD + count * np.dtype(dtype).itemsize],
                dtype,
            ).copy(),
        )
    for buffer, payload in input_buffers:
        assert buffer.read()[: len(payload)] == payload, "an input buffer changed"
    return results


def execute(artifact, bindings, *, grid=(1, 1, 1), scalars=None, repetitions=1):
    metal = device()
    metal.prepare(
        [artifact.dispatch(metal, bindings, grid=grid, scalars=scalars or {})],
    ).submit(repetitions).wait()


def untouched(dtype):
    """The value an output element holds if the kernel never wrote it."""
    return np.frombuffer(bytes([FILL]) * np.dtype(dtype).itemsize, dtype)[0]


def assert_same(actual, expected, context=""):
    """Bit-identical, except that any NaN matches any NaN."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, context
    if actual.dtype.kind == "f":
        both_nan = np.isnan(actual) & np.isnan(expected)
        bits = actual.view(np.uint32 if actual.itemsize == 4 else np.uint16)
        reference = expected.astype(actual.dtype).view(bits.dtype)
        wrong = ~both_nan & (bits != reference)
    else:
        wrong = actual != expected
    if wrong.any():
        index = np.flatnonzero(wrong)[:8]
        raise AssertionError(
            f"{context}: {wrong.sum()} of {wrong.size} differ; first "
            f"at {index.tolist()}: got {actual.flat[index].tolist()}, expected "
            f"{expected.flat[index].tolist()}",
        )


def ulp_distance(actual, expected):
    """Distance in f32 units in the last place, on the monotonic integer line; NaN pairs
    count as 0."""

    def ordered(values):
        bits = np.asarray(values, np.float32).view(np.int32).astype(np.int64)
        return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)

    actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    distance = np.abs(ordered(actual) - ordered(expected))
    nan = np.isnan(actual) | np.isnan(expected)
    return np.where(
        nan,
        np.where(np.isnan(actual) & np.isnan(expected), 0, np.iinfo(np.int64).max),
        distance,
    )


def special_f32():
    bits = [
        0,
        0x80000000,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
        0x7F7FFFFF,
        0x00800000,
        0x00000001,
        0x007FFFFF,
        0x3F800000,
        0x3F800001,
        0x3F7FFFFF,
        0x4B000000,
        0x4B000001,
        0x33800000,
    ]
    bits += [
        value | 0x80000000
        for value in bits
        if not value & 0x80000000 and value != 0x7FC00000
    ]
    return np.array(bits, np.uint32).view(np.float32)


def random_f32(count, seed=0, *, finite=True):
    rng = np.random.default_rng(seed)
    if finite:
        return (
            rng.standard_normal(count) * np.exp2(rng.integers(-20, 20, count))
        ).astype(np.float32)
    return (
        rng.integers(0, 2**32, count, dtype=np.uint64)
        .astype(np.uint32)
        .view(np.float32)
    )


def flush(values):
    """Metal's f32 subnormal rule, keeping the sign of flushed values.

    Metal's arithmetic and math functions flush subnormal inputs and results, except
    sin, tan, tanh and abs, which keep them; comparisons, maximum and minimum do not
    flush.
    """
    values = np.asarray(values, np.float32)
    subnormal = (values != 0) & (np.abs(values) < np.finfo(np.float32).tiny)
    return np.where(subnormal, np.copysign(np.float32(0), values), values).astype(
        np.float32,
    )


def guarded(metal, size):
    """A buffer of `size` bytes between two guard bands of FILL bytes, and its binding
    view."""
    buffer = metal.buffer(size + 2 * GUARD)
    buffer.write(bytes([FILL]) * buffer.size)
    return buffer, buffer.view(GUARD, size)


def unguarded(buffer):
    """The bytes between a `guarded` buffer's guard bands, after checking both bands."""
    raw = buffer.read()
    assert raw[:GUARD] == raw[-GUARD:] == bytes([FILL]) * GUARD, (
        "write outside the binding"
    )
    return raw[GUARD:-GUARD]
