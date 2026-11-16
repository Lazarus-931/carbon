from functools import cache

import numpy as np
import pytest

from carbon import language as cl
from carbon.compiler import Atom, TensorAtom, TiledAtom, View
from testbed import device, run, run_compiled, storage


@cache
def tile_product(dtype):
    @cl.kernel
    def product(
        left: cl.buffer(dtype),
        right: cl.buffer(dtype),
        output: cl.buffer(cl.f32),
        LEFT: cl.constexpr,
        RIGHT: cl.constexpr,
        TILED: cl.constexpr,
        STEPS: cl.constexpr,
    ):
        fragments_m, fragments_n = TILED.fragments
        row = cl.simd_id() // TILED.groups[1] * fragments_m
        column = cl.simd_id() % TILED.groups[1] * fragments_n
        accumulators = TILED.zeros()
        for step in range(STEPS):
            accumulators = TILED.mma(
                TILED.load_a(cl.view(left, LEFT), row, step),
                TILED.load_b(cl.view(right, RIGHT), column, step),
                accumulators,
            )
        rows, columns = TILED.shape
        cl.store(
            output,
            cl.arange(rows * columns, layout=TILED.layout()),
            TILED.unpack(accumulators),
            True,
        )

    return product


@pytest.mark.parametrize("dtype", [cl.f32, cl.f16, cl.bf16], ids=["f32", "f16", "bf16"])
@pytest.mark.parametrize(
    "groups, fragments",
    [((1, 1), (1, 1)), ((1, 1), (2, 2)), ((2, 1), (1, 2)), ((2, 2), (2, 1))],
)
def test_matrix_atom_products_are_exact(dtype, groups, fragments):
    tiled = TiledAtom(Atom(dtype), groups, fragments)
    rows, columns = tiled.shape
    depth = 24
    rng = np.random.default_rng(rows * 31 + columns)
    a = rng.integers(-4, 5, (rows, depth)).astype(np.float32)
    b = rng.integers(-4, 5, (depth, columns)).astype(np.float32)
    size = dtype.itemsize
    (out,) = run(
        tile_product(dtype),
        inputs=[storage(a, dtype), storage(b, dtype)],
        outputs=[(rows * columns, np.float32)],
        constants={
            "LEFT": View(dtype.name, (rows, depth), (depth, 1), rows * depth * size),
            "RIGHT": View(
                dtype.name,
                (depth, columns),
                (columns, 1),
                depth * columns * size,
            ),
            "TILED": tiled,
            "STEPS": depth // 8,
        },
        threads=tiled.threads,
    )
    assert np.array_equal(out.reshape(rows, columns), a @ b)


@cache
def repeated_product(dtype):
    @cl.kernel
    def product(
        left: cl.buffer(dtype),
        right: cl.buffer(dtype),
        output: cl.buffer(cl.f32),
        count: cl.u32,
        ATOM: cl.constexpr,
        A: cl.constexpr,
        B: cl.constexpr,
        C: cl.constexpr,
    ):
        a = cl.load(left, cl.arange(A.size, layout=A), True, other=0.0)
        b = cl.load(right, cl.arange(B.size, layout=B), True, other=0.0)
        zero = ATOM.zeros()
        total = zero
        for step in range(count):
            first = ATOM.mma(a, b, zero)
            second = ATOM.mma(a, b, zero)
            _unused = ATOM.mma(a, b, first)
            total = total + first + second
        cl.store(output, cl.arange(C.size, layout=C), total, True)

    return product


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize(
    "atom",
    [
        Atom(cl.f32),
        Atom(cl.f16),
        Atom(cl.bf16),
        TensorAtom(),
        TensorAtom(operand=cl.bf16),
    ],
)
def test_matrix_effects_remove_repeated_work_without_speculating(atom, count):
    if isinstance(atom, TensorAtom) and not device().tensor_ops:
        pytest.skip("Metal 4 tensor operations unavailable")
    m, n, k = atom.shape
    dtype = atom.operand
    rng = np.random.default_rng(32)
    a = rng.integers(-2, 3, (2, m, k)).astype(np.float32)
    b = rng.integers(-2, 3, (2, k, n)).astype(np.float32)
    artifact = repeated_product(dtype).compile(
        constants={
            "ATOM": atom,
            "A": atom.layout("a", threads=64),
            "B": atom.layout("b", threads=64),
            "C": atom.layout(threads=64),
        },
        threads=64,
        device=device(),
    )
    assert artifact.ir.count('"carbon.mma"') == 1
    assert artifact.thread_ir.count('"msl.mma"') == 1
    assert artifact.thread_ir.index("scf.for") < artifact.thread_ir.index('"msl.mma"')
    (actual,) = run_compiled(
        artifact,
        inputs=[storage(a, dtype), storage(b, dtype)],
        outputs=[(2 * m * n, np.float32)],
        scalars={"count": count},
    )
    np.testing.assert_array_equal(actual.reshape(2, m, n), (a @ b) * (2 * count))
