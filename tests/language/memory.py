import numpy as np
import pytest

from carbon import language as cl
from carbon.compiler import View
from testbed import assert_same, device, execute, run, untouched, upload

BLOCK = 64


@cl.kernel
def doubled(source: cl.buffer(cl.f32), output: cl.buffer(cl.f32), count: cl.u32):
    index = cl.program_id() * BLOCK + cl.arange(BLOCK)
    active = index < count
    cl.store(output, index, cl.load(source, index, active, other=0.0) * 2.0, active)


@cl.kernel
def block_ids(output: cl.buffer(cl.u32)):
    index = cl.program_id() * BLOCK + cl.arange(BLOCK)
    cl.store(output, index, cl.full((BLOCK,), cl.program_id(), cl.u32), True)


@cl.kernel
def histogram(source: cl.buffer(cl.u32), counts: cl.buffer(cl.u32), count: cl.u32):
    index = cl.program_id() * BLOCK + cl.arange(BLOCK)
    active = index < count
    bins = cl.load(source, index, active, other=0)
    cl.atomic_add(counts, bins, cl.full((BLOCK,), 1, cl.u32), active)


@cl.kernel
def gathered(
    source: cl.buffer(cl.f32),
    output: cl.buffer(cl.f32),
    SOURCE: cl.constexpr,
):
    rows, columns = SOURCE.shape
    index = cl.arange(rows * columns)
    cl.store(
        output,
        index,
        cl.load(cl.view(source, SOURCE), index, True, other=0.0),
        True,
    )


@pytest.mark.parametrize("count", [0, 1, 63, 64, 65, 1000])
def test_masked_stores_touch_only_active_elements(count):
    values = np.arange(count, dtype=np.float32) - 7.5
    (out,) = run(
        doubled,
        inputs=[values],
        outputs=[(count + 9, np.float32)],
        scalars={"count": count},
        threads=BLOCK,
        grid=(max(1, -(-count // BLOCK)), 1, 1),
    )
    assert_same(out[:count], values * 2)
    assert (out[count:].view(np.uint32) == untouched(np.float32).view(np.uint32)).all()


def test_every_threadgroup_runs_once():
    groups = 37
    (out,) = run(
        block_ids,
        outputs=[(groups * BLOCK, np.uint32)],
        threads=BLOCK,
        grid=(groups, 1, 1),
    )
    assert_same(out, np.repeat(np.arange(groups, dtype=np.uint32), BLOCK))


def test_atomic_adds_count_every_update():
    metal = device()
    bins = np.random.default_rng(0).integers(0, 17, 5000).astype(np.uint32)
    counts = upload(metal, np.zeros(17, np.uint32))
    artifact = histogram.compile(constants={}, threads=BLOCK, device=metal)
    execute(
        artifact,
        [upload(metal, bins), counts],
        grid=(-(-bins.size // BLOCK), 1, 1),
        scalars={"count": bins.size},
    )
    assert_same(
        np.frombuffer(counts.read(), np.uint32)[:17],
        np.bincount(bins, minlength=17).astype(np.uint32),
    )


@pytest.mark.parametrize("rows, columns, stride", [(1, 1, 1), (3, 31, 37), (7, 16, 16)])
def test_strided_views_read_logical_elements(rows, columns, stride):
    padded = np.arange(rows * stride, dtype=np.float32)
    view = View("f32", (rows, columns), (stride, 1), padded.nbytes)
    (out,) = run(
        gathered,
        inputs=[padded],
        outputs=[(rows * columns, np.float32)],
        constants={"SOURCE": view},
        threads=32,
    )
    assert_same(out, padded.reshape(rows, stride)[:, :columns].ravel())
