import numpy as np
import pytest

from carbon import language as cl
from testbed import run


@cl.kernel
def gemv(
    vector: cl.buffer(cl.f32),
    matrix: cl.buffer(cl.f32),
    output: cl.buffer(cl.f32),
    rows: cl.u32,
    columns: cl.u32,
    alpha: cl.f32,
    BLOCK: cl.constexpr,
):
    column = cl.program_id() * BLOCK + cl.arange(BLOCK)
    active = column < columns
    accumulator = cl.full((BLOCK,), 0.0, cl.f32)
    for row in range(rows):
        left = cl.load(vector, cl.full((BLOCK,), row, cl.u32), active, other=0.0)
        accumulator = accumulator + left * cl.load(
            matrix,
            row * columns + column,
            active,
            other=0.0,
        )
    if alpha != 0.0:
        accumulator = accumulator * alpha
    cl.store(output, column, accumulator, active)


@cl.kernel
def clipped(source: cl.buffer(cl.f32), output: cl.buffer(cl.f32), count: cl.u32):
    index = cl.arange(128)
    active = index < count
    value = cl.load(source, index, active, other=0.0)
    cl.store(output, index, cl.select(value > 0.0, value, value * 0.25), active)


def test_static_kernel_ir_has_a_runtime_loop():
    staged = gemv.stage(constants={"BLOCK": 64}, threads=32).ir
    assert "scf.for" in staged and "iter_args" in staged


@pytest.mark.parametrize(
    "rows, columns, block, threads, alpha",
    [
        (0, 33, 64, 32, 1.0),
        (1, 1, 32, 32, 0.0),
        (7, 93, 96, 32, 2.0),
        (19, 257, 96, 64, -0.5),
    ],
)
def test_runtime_loops_and_branches(rows, columns, block, threads, alpha):
    a = ((np.arange(rows) % 11 - 5) / 8).astype(np.float32)
    b = (
        ((np.arange(rows * columns) % 7 - 3) / 8)
        .astype(np.float32)
        .reshape(rows, columns)
    )
    (out,) = run(
        gemv,
        inputs=[a, b],
        outputs=[(columns, np.float32)],
        constants={"BLOCK": block},
        scalars={"rows": rows, "columns": columns, "alpha": alpha},
        threads=threads,
        grid=(-(-columns // block), 1, 1),
    )
    expected = a @ b if rows else np.zeros(columns, np.float32)
    assert np.array_equal(out, expected * alpha if alpha != 0.0 else expected)


def test_select_is_elementwise():
    values = np.linspace(-4, 4, 100, dtype=np.float32)
    (out,) = run(
        clipped,
        inputs=[values],
        outputs=[(100, np.float32)],
        scalars={"count": 100},
        threads=128,
    )
    assert np.array_equal(out, np.where(values > 0, values, values * 0.25))
