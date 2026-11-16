import numpy as np
import pytest

from carbon import language as cl
from carbon.compiler import Layout
from testbed import assert_same, run


@cl.kernel
def exchange(
    source: cl.buffer(cl.u32),
    output: cl.buffer(cl.u32),
    count: cl.u32,
    SOURCE: cl.constexpr,
    TARGET: cl.constexpr,
):
    shared = cl.scratch(SOURCE.shape, cl.u32)
    index = cl.program_id() * SOURCE.size + cl.arange(SOURCE.size, layout=SOURCE)
    value = cl.convert_layout(
        cl.load(source, index, index < count, other=0),
        TARGET,
        scratch=shared,
    )
    index = cl.program_id() * TARGET.size + cl.arange(TARGET.size, layout=TARGET)
    cl.store(output, index, value, index < count)


@cl.kernel
def reversed_rows(
    source: cl.buffer(cl.f32),
    output: cl.buffer(cl.f32),
    SIZE: cl.constexpr,
):
    shared = cl.scratch((SIZE,), cl.f32)
    index = cl.arange(SIZE)
    cl.store(shared, index, cl.load(source, index, True, other=0.0), True)
    cl.barrier()
    cl.store(output, index, cl.load(shared, SIZE - 1 - index, True, other=0.0), True)


@pytest.mark.parametrize(
    "shape, threads",
    [((1,), 64), ((93,), 64), ((257,), 128), ((3, 31), 64)],
)
def test_layout_exchange_through_scratch_moves_every_bit(shape, threads):
    size = int(np.prod(shape))
    values = (0x80000000 + np.arange(3 * size - 1, dtype=np.uint64) * 13337).astype(
        np.uint32,
    )
    constants = {
        "SOURCE": Layout.striped(shape, threads=threads),
        "TARGET": Layout.blocked(shape, threads=threads),
    }
    staged = exchange.stage(constants=constants, threads=threads).ir
    assert '"carbon.scratch"' in staged and '"carbon.convert_layout"' in staged
    (out,) = run(
        exchange,
        inputs=[values],
        outputs=[(values.size, np.uint32)],
        constants=constants,
        scalars={"count": values.size},
        threads=threads,
        grid=(3, 1, 1),
    )
    assert_same(out, values)


@pytest.mark.parametrize("size", [32, 100, 512])
def test_threads_read_what_other_threads_wrote(size):
    values = np.arange(size, dtype=np.float32) * 1.5
    (out,) = run(
        reversed_rows,
        inputs=[values],
        outputs=[(size, np.float32)],
        constants={"SIZE": size},
        threads=min(size, 256) // 32 * 32 or 32,
    )
    assert_same(out, values[::-1])
