from functools import cache

import numpy as np
import pytest

from carbon import language as cl
from carbon.compiler import Layout
from testbed import assert_same, run

SHAPES = [((32,), 32), ((93,), 64), ((257,), 128), ((4, 33), 64)]
FLAT = [shape for shape in SHAPES if len(shape[0]) == 1]


@cache
def reduce_kernel(element, kind):
    zero = 0.0 if element == cl.f32 else 0

    @cl.kernel
    def reduce(
        source: cl.buffer(element),
        output: cl.buffer(element),
        LAYOUT: cl.constexpr,
    ):
        shared = cl.scratch(LAYOUT.shape, element)
        value = cl.load(source, cl.arange(LAYOUT.size, layout=LAYOUT), True, other=zero)
        total = getattr(cl, kind)(value, 0, scratch=shared)
        cl.store(output, cl.full((), 0, cl.u32), total, True)

    return reduce


@cache
def scan_kernel(element):
    zero = 0.0 if element == cl.f32 else 0

    @cl.kernel
    def scan(
        source: cl.buffer(element),
        output: cl.buffer(element),
        LAYOUT: cl.constexpr,
    ):
        shared = cl.scratch(LAYOUT.shape, element)
        index = cl.arange(LAYOUT.size, layout=LAYOUT)
        value = cl.load(source, index, True, other=zero)
        cl.store(
            output,
            index,
            cl.cumsum(value, len(LAYOUT.shape) - 1, scratch=shared),
            True,
        )

    return scan


def values(shape, element):
    rng = np.random.default_rng(int(np.prod(shape)))
    if element == cl.u32:
        return rng.integers(0, 2**32, shape, dtype=np.uint64).astype(np.uint32)
    return rng.integers(-8, 9, shape).astype(np.float32)


@pytest.mark.parametrize("element", [cl.f32, cl.u32], ids=["f32", "u32"])
@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("shape, threads", FLAT, ids=[str(s) for s, _ in FLAT])
def test_reductions_match_numpy(shape, threads, kind, element):
    data = values(shape, element)
    (out,) = run(
        reduce_kernel(element, kind),
        inputs=[data.ravel()],
        outputs=[(1, data.dtype)],
        constants={"LAYOUT": Layout.striped(shape, threads=threads)},
        threads=threads,
    )
    with np.errstate(over="ignore"):
        expected = data.sum(dtype=data.dtype) if kind == "sum" else data.max()
    assert_same(out, np.array([expected], data.dtype), kind)


@pytest.mark.parametrize("element", [cl.f32, cl.u32], ids=["f32", "u32"])
@pytest.mark.parametrize("shape, threads", SHAPES, ids=[str(s) for s, _ in SHAPES])
def test_scans_match_numpy(shape, threads, element):
    data = values(shape, element)
    (out,) = run(
        scan_kernel(element),
        inputs=[data.ravel()],
        outputs=[(data.size, data.dtype)],
        constants={"LAYOUT": Layout.striped(shape, threads=threads)},
        threads=threads,
    )
    with np.errstate(over="ignore"):
        expected = np.cumsum(data, axis=-1, dtype=data.dtype)
    assert_same(out.reshape(shape), expected)
