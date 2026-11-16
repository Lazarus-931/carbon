import operator
from functools import cache

import numpy as np
import pytest

from carbon import language as cl
from testbed import (
    assert_same,
    bf16_bits,
    flush,
    random_f32,
    run,
    special_f32,
    ulp_distance,
)

BLOCK = 256
BINARY = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "and": operator.and_,
    "or": operator.or_,
    "xor": operator.xor,
}
MATH = {
    "exp": (np.exp, 1, (-20, 20)),
    "log": (np.log, 2, (1e-3, 1e3)),
    "sqrt": (np.sqrt, 0, (0, 1e6)),
    "rsqrt": (lambda x: 1 / np.sqrt(x), 1, (1e-3, 1e3)),
    "sin": (np.sin, 3, (-10, 10)),
    "tanh": (np.tanh, 2, (-5, 5)),
    "sigmoid": (lambda x: 1 / (1 + np.exp(-x)), 3, (-20, 20)),
}


@cache
def binary_kernel(element):
    zero, one = (0.0, 1.0) if element == cl.f32 else (0, 1)

    @cl.kernel
    def binary(
        left: cl.buffer(element),
        right: cl.buffer(element),
        output: cl.buffer(element),
        count: cl.u32,
        OP: cl.constexpr,
    ):
        index = cl.program_id() * BLOCK + cl.arange(BLOCK)
        active = index < count
        a = cl.load(left, index, active, other=zero)
        b = cl.load(right, index, active, other=one)
        cl.store(output, index, BINARY[OP](a, b), active)

    return binary


@cl.kernel
def unary(
    source: cl.buffer(cl.f32),
    output: cl.buffer(cl.f32),
    count: cl.u32,
    OP: cl.constexpr,
):
    index = cl.program_id() * BLOCK + cl.arange(BLOCK)
    active = index < count
    cl.store(
        output,
        index,
        getattr(cl, OP)(cl.load(source, index, active, other=1.0)),
        active,
    )


@cache
def cast_kernel(target):
    @cl.kernel
    def convert(source: cl.buffer(cl.f32), output: cl.buffer(target), count: cl.u32):
        index = cl.program_id() * BLOCK + cl.arange(BLOCK)
        active = index < count
        value = cl.load(source, index, active, other=0.0)
        cl.store(output, index, cl.cast(value, target), active)

    return convert


def launch(kernel, inputs, result, constants=None):
    count = inputs[0].size
    return run(
        kernel,
        inputs=inputs,
        outputs=[(count, result)],
        constants=constants,
        scalars={"count": count},
        threads=BLOCK,
        grid=(-(-count // BLOCK), 1, 1),
    )[0]


@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_f32_arithmetic_is_correctly_rounded(op):
    special = special_f32()
    left, right = (
        np.concatenate([grid.ravel(), random_f32(4096, seed=seed)])
        for seed, grid in enumerate(np.meshgrid(special, special), 1)
    )
    flushed = flush(left), flush(right)
    with np.errstate(all="ignore"):
        expected = BINARY[op](*flushed)
        exact = BINARY[op](*(value.astype(np.float64) for value in flushed))
    tiny = (exact != 0) & (np.abs(exact) < np.finfo(np.float32).tiny)
    expected = np.where(tiny, np.copysign(np.float32(0), exact), expected)
    assert_same(
        launch(binary_kernel(cl.f32), [left, right], np.float32, {"OP": op}),
        expected,
        op,
    )


@pytest.mark.parametrize("op", ["add", "sub", "mul", "and", "or", "xor"])
def test_u32_arithmetic_wraps(op):
    rng = np.random.default_rng(3)
    left, right = rng.integers(0, 2**32, (2, 5000), dtype=np.uint64).astype(np.uint32)
    with np.errstate(over="ignore"):
        expected = BINARY[op](left, right).astype(np.uint32)
    assert_same(
        launch(binary_kernel(cl.u32), [left, right], np.uint32, {"OP": op}),
        expected,
        op,
    )


@pytest.mark.parametrize("op", sorted(MATH))
def test_math_functions_within_measured_ulps(op):
    function, ulps, (low, high) = MATH[op]
    values = np.random.default_rng(5).uniform(low, high, 5000).astype(np.float32)
    expected = function(values.astype(np.float64)).astype(np.float32)
    actual = launch(unary, [values], np.float32, {"OP": op})
    assert ulp_distance(actual, expected).max() <= ulps, op


def test_casts_round_to_nearest_even_once():
    values = (np.random.default_rng(7).standard_normal(5000) * 100).astype(np.float32)
    assert_same(
        launch(cast_kernel(cl.bf16), [values], np.uint16),
        bf16_bits(values),
        "bf16",
    )
    assert_same(
        launch(cast_kernel(cl.f16), [values], np.float16),
        values.astype(np.float16),
        "f16",
    )
