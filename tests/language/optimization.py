from functools import cache

import numpy as np
import pytest

from carbon import language as cl
from carbon.compiler import Layout, Mode
from testbed import assert_same, device, run_compiled, upload


@cl.kernel
def layout_chain(
    source: cl.buffer(cl.f32),
    output: cl.buffer(cl.f32),
    A: cl.constexpr,
    B: cl.constexpr,
    SHARED: cl.constexpr,
):
    index = cl.arange(A.size, layout=A)
    value = cl.convert_layout(cl.load(source, index, True, other=0.0), B)
    squared = value * value + 0.25
    if SHARED:
        cl.store(output, cl.arange(B.size, layout=B) + B.size, squared, True)
    result = cl.convert_layout(squared, A)
    cl.store(output, index, result, True)


@pytest.mark.parametrize("size", [93, 128, 257, 512])
@pytest.mark.parametrize("shared", [False, True])
def test_layout_propagation_preserves_values_and_shared_users(size, shared):
    values = np.linspace(-2, 2, size, dtype=np.float32)
    values[:5] = [0.0, -0.0, np.inf, -np.inf, np.nan]
    artifact = layout_chain.compile(
        constants={
            "A": Layout.striped((size,)),
            "B": Layout.blocked((size,)),
            "SHARED": shared,
        },
        threads=32,
        device=device(),
    )
    (actual,) = run_compiled(
        artifact,
        inputs=[values],
        outputs=[(size * (2 if shared else 1), np.float32)],
    )
    expected = values * values + np.float32(0.25)
    assert_same(actual, np.tile(expected, 2 if shared else 1))
    assert ('"carbon.convert_layout"' in artifact.ir) == shared
    assert ("simd_shuffle" in artifact.source) == shared


@cl.kernel
def loop_reads(
    source: cl.buffer(cl.f32),
    output: cl.buffer(cl.f32),
    count: cl.u32,
    MUTATE: cl.constexpr,
    ADVANCE: cl.constexpr,
    A: cl.constexpr,
    SYNCHRONIZE: cl.constexpr,
):
    index = cl.arange(A.size, layout=A)
    total = cl.full(A.shape, 0.0, cl.f32, layout=A)
    for step in range(count):
        if SYNCHRONIZE:
            cl.barrier()
        at = index + step * A.size if ADVANCE else index
        value = cl.load(source, at, True, other=0.0)
        total = total + value
        if MUTATE:
            cl.store(source, index, value + 1.0, True)
    cl.store(output, index, total, True)


@pytest.mark.parametrize("count", [0, 1, 17])
@pytest.mark.parametrize(
    "size,layout,missing",
    [
        (64, Layout.striped, 0),
        (512, Layout.blocked, 0),
        (512, Layout.blocked, 3),
        (257, Layout.blocked, 3),
    ],
)
@pytest.mark.parametrize(
    "mutate,advance",
    [(False, False), (True, False), (False, True)],
)
def test_load_hoisting_respects_mutation_indices_and_zero_trip(
    count,
    mutate,
    advance,
    size,
    layout,
    missing,
):
    metal = device()
    values = np.arange(
        size * max(1, count if advance else 1) - missing,
        dtype=np.float32,
    )
    source, output = upload(metal, values), metal.buffer(size * 4)
    artifact = loop_reads.compile(
        constants={
            "MUTATE": mutate,
            "ADVANCE": advance,
            "A": layout((size,), threads=64),
            "SYNCHRONIZE": size != 64,
        },
        threads=64,
        device=metal,
    )
    metal.prepare(
        [
            artifact.dispatch(
                metal,
                [source, output],
                grid=(1, 1, 1),
                scalars={"count": count},
            ),
        ],
    ).submit().wait()
    expected = (
        np.pad(values, (0, missing))[: count * size].reshape(count, size).sum(0)
        if advance
        else np.pad(
            values * count + (count * (count - 1) / 2 if mutate else 0),
            (0, missing),
        )
    )
    np.testing.assert_array_equal(np.frombuffer(output.read(), np.float32), expected)
    np.testing.assert_array_equal(
        np.frombuffer(source.read(), np.float32),
        values + (count if mutate else 0),
    )
    assert (artifact.source.index("arg0[") < artifact.source.index("for (")) == (
        not mutate and not advance
    )
    if not mutate and not advance:
        assert artifact.source.index("if (") < artifact.source.index("arg0[")
        assert "arg0[" not in artifact.source[artifact.source.index("for (") :]
        if size == 512:
            assert "device const float4*" in artifact.source


@cl.kernel
def reduction_item(
    source: cl.buffer(cl.f32),
    output: cl.buffer(cl.f32),
    A: cl.constexpr,
):
    index = cl.arange(A.size, layout=A)
    value = cl.load(source, index, True, other=0.0)
    scratch = cl.scratch(A.shape, cl.f32)
    total = cl.item(cl.sum(value, 0, scratch=scratch), scratch=scratch)
    cl.store(output, index, value + total, True)


@pytest.mark.parametrize(
    "size,threads",
    [(512, 32), (257, 64), (2048, 128), (8192, 256)],
)
def test_native_reduction_and_item_use_only_partial_scratch(size, threads):
    artifact = reduction_item.compile(
        constants={"A": Layout.striped((size,), threads=threads)},
        threads=threads,
        device=device(),
        max_scratch=32,
    )
    expected_bytes = 0 if threads == 32 else ((threads // 32) * 4 + 15) // 16 * 16
    assert artifact.scratch_bytes == expected_bytes
    assert artifact.scratch_layout[0].bytes == expected_bytes
    values = np.arange(size, dtype=np.float32) / 8192
    (actual,) = run_compiled(artifact, inputs=[values], outputs=[(size, np.float32)])
    np.testing.assert_array_equal(actual, values + values.sum())


@cl.kernel
def local_collective_workspace(
    source: cl.buffer(cl.f32),
    output: cl.buffer(cl.f32),
    A: cl.constexpr,
    KIND: cl.constexpr,
    OBSERVE: cl.constexpr,
):
    shared = cl.scratch(A.shape, cl.f32)
    index = cl.arange(A.size, layout=A)
    value = cl.load(source, index, True, other=0.0)
    if KIND == "cumsum":
        result = cl.cumsum(value, 1, scratch=shared)
        target = A
    else:
        target = A.reduced(1)
        result = getattr(cl, KIND)(value, 1, layout=target, scratch=shared)
    cl.store(output, cl.arange(target.size, layout=target), result, True)
    if OBSERVE:
        cl.store(shared, index, value, True)
        cl.barrier()
        cl.store(
            output,
            target.size + index,
            cl.load(shared, index, True, other=0.0),
            True,
        )


@pytest.mark.parametrize("kind", ["sum", "max", "cumsum"])
@pytest.mark.parametrize("observe", [False, True])
def test_unused_collective_workspace_respects_explicit_accesses(kind, observe):
    layout = Layout((4, 256), Mode(4, 256), Mode(32, 1), Mode(8, 32))
    values = (np.arange(layout.size, dtype=np.float32).reshape(layout.shape) % 17) / 16
    expected = getattr(np, kind)(values, axis=1).ravel()
    if observe:
        expected = np.concatenate((expected, values.ravel()))
    artifact = local_collective_workspace.compile(
        constants={"A": layout, "KIND": kind, "OBSERVE": observe},
        threads=128,
        device=device(),
    )
    assert artifact.scratch_bytes == (values.nbytes if observe else 0)
    (actual,) = run_compiled(
        artifact,
        inputs=[values.ravel()],
        outputs=[(expected.size, np.float32)],
    )
    np.testing.assert_array_equal(actual, expected)


def test_unused_workspace_can_exceed_the_physical_memory_limit():
    metal = device()
    width = metal.max_threadgroup_memory // 16 + 32
    layout = Layout((4, width), Mode(4, width), Mode(32, 1), Mode(width // 32, 32))
    options = {"A": layout, "KIND": "sum", "OBSERVE": False}
    artifact = local_collective_workspace.compile(
        constants=options,
        threads=128,
        device=metal,
        max_scratch=16,
    )
    assert artifact.scratch_bytes == 0
    values = np.ones(layout.size, np.float32)
    (actual,) = run_compiled(artifact, inputs=[values], outputs=[(4, np.float32)])
    np.testing.assert_array_equal(actual, np.full(4, width, np.float32))
    with pytest.raises(ValueError, match="bytes of threadgroup memory"):
        local_collective_workspace.compile(
            constants=dict(options, OBSERVE=True),
            threads=128,
            device=metal,
        )


@cache
def narrow_copy(element):
    @cl.kernel
    def copy(
        source: cl.buffer(element),
        output: cl.buffer(element),
        count: cl.u32,
        A: cl.constexpr,
    ):
        index = cl.arange(A.size, layout=A)
        value = cl.load(source, index, index < count, other=0)
        cl.store(output, index, value, index < count)

    return copy


@pytest.mark.parametrize("element", [cl.u8, cl.i8, cl.u16, cl.i16, cl.f16, cl.bf16])
@pytest.mark.parametrize(
    "size,count",
    [(1024, 0), (1024, 1), (1024, 1009), (1024, 1024), (512, 497)],
)
def test_wide_storage_accesses_preserve_bits_and_masked_tails(element, size, count):
    dtype = np.uint8 if element.itemsize == 1 else np.uint16
    values = (np.arange(size, dtype=np.uint32) * 0x9E37).astype(dtype)
    if element.itemsize == 2:
        values[:6] = [0x0000, 0x8000, 0x7C00, 0xFC00, 0x7C01, 0x7E00]
        values[6:10] = [0x7F80, 0xFF80, 0x7F81, 0x7FC1]
    artifact = narrow_copy(element).compile(
        constants={"A": Layout.blocked((size,), threads=64)},
        threads=64,
        device=device(),
    )
    assert ("uint4*>" in artifact.source) == (size == 1024)
    assert artifact.binding_alignment == ((16, 16) if size == 1024 else (8, 8))
    (actual,) = run_compiled(
        artifact,
        inputs=[values],
        outputs=[(size, dtype)],
        scalars={"count": count},
    )
    np.testing.assert_array_equal(actual[:count], values[:count])
    assert actual[count:].tobytes() == bytes([0xA5]) * (
        (size - count) * element.itemsize
    )


@cl.kernel
def scratch_layout_loop(
    source: cl.buffer(cl.f32),
    output: cl.buffer(cl.f32),
    count: cl.u32,
    A: cl.constexpr,
    B: cl.constexpr,
    OBSERVE: cl.constexpr,
):
    index = cl.arange(A.size, layout=A)
    state = cl.load(source, index, True, other=0.0)
    observed = cl.full(A.shape, 0.0, cl.f32, layout=A)
    scratch = cl.scratch(A.shape, cl.f32)
    for step in range(count):
        moved = cl.convert_layout(state, B, scratch=scratch)
        state = cl.convert_layout(moved * moved + 0.25, A, scratch=scratch)
        if OBSERVE:
            observed = cl.load(scratch, index, True, other=0.0)
            cl.barrier()
    cl.store(output, index, state, True)
    if OBSERVE:
        cl.store(output, index + A.size, observed, True)


@pytest.mark.parametrize("size", [93, 512])
@pytest.mark.parametrize("count", [0, 1, 7])
@pytest.mark.parametrize("observe", [False, True])
def test_scratch_layout_round_trip_preserves_loop_state_and_observers(
    size,
    count,
    observe,
):
    values = np.linspace(-0.25, 0.25, size, dtype=np.float32)
    expected = values.copy()
    for _ in range(count):
        expected = expected * expected + np.float32(0.25)
    artifact = scratch_layout_loop.compile(
        constants={
            "A": Layout.striped((size,), threads=64),
            "B": Layout.blocked((size,), threads=64),
            "OBSERVE": observe,
        },
        threads=64,
        device=device(),
    )
    assert (artifact.scratch_bytes > 0) == observe
    (actual,) = run_compiled(
        artifact,
        inputs=[values],
        outputs=[(size * (2 if observe else 1), np.float32)],
        scalars={"count": count},
    )
    np.testing.assert_array_equal(actual[:size], expected)
    if observe:
        np.testing.assert_array_equal(
            actual[size:],
            expected if count else np.zeros_like(expected),
        )


@cl.kernel
def scratch_conversion_fence(
    source: cl.buffer(cl.f32),
    output: cl.buffer(cl.f32),
    count: cl.u32,
    A: cl.constexpr,
    B: cl.constexpr,
):
    index = cl.arange(A.size, layout=A)
    value = cl.load(source, index, True, other=0.0)
    other = cl.scratch(A.shape, cl.f32)
    workspace = cl.scratch(A.shape, cl.f32)
    for step in range(count):
        cl.store(other, index, value + cl.cast(step, cl.f32), True)
        moved = cl.convert_layout(value, B, scratch=workspace)
        reversed_value = cl.load(other, A.size - 1 - index, True, other=0.0)
        result = cl.convert_layout(moved * moved, A, scratch=workspace)
        cl.store(output, index, result, True)
        cl.store(output, index + A.size, reversed_value, True)


@pytest.mark.parametrize("count", [1, 17])
def test_eliminated_scratch_conversion_keeps_fences_for_other_allocations(count):
    size = 512
    artifact = scratch_conversion_fence.compile(
        constants={
            "A": Layout.striped((size,), threads=64),
            "B": Layout.blocked((size,), threads=64),
        },
        threads=64,
        device=device(),
    )
    assert '"carbon.convert_layout"' not in artifact.ir
    assert artifact.scratch_bytes == size * 4
    assert artifact.source.count("threadgroup_barrier(") == 2
    values = np.arange(size, dtype=np.float32) / 512
    (actual,) = run_compiled(
        artifact,
        inputs=[values],
        outputs=[(2 * size, np.float32)],
        scalars={"count": count},
    )
    np.testing.assert_array_equal(
        actual,
        np.concatenate((values * values, values[::-1] + count - 1)),
    )


@cl.kernel
def group_dependent_layout(
    source: cl.buffer(cl.f32),
    output: cl.buffer(cl.f32),
    A: cl.constexpr,
    B: cl.constexpr,
):
    index = cl.arange(A.size, layout=A)
    scratch = cl.scratch(A.shape, cl.f32)
    value = cl.convert_layout(
        cl.load(source, index, True, other=0.0),
        B,
        scratch=scratch,
    )
    group = cl.cast(cl.full(B.shape, cl.simd_id(), cl.u32, layout=B), cl.f32)
    value = cl.convert_layout(value + group, A, scratch=scratch)
    cl.store(output, index, value, True)


@pytest.mark.parametrize("replicated", [False, True])
def test_layout_propagation_preserves_group_dependent_broadcasts(replicated):
    size = 512
    target = (
        Layout.striped((size,)).across_groups(4)
        if replicated
        else Layout.blocked((size,), threads=128)
    )
    artifact = group_dependent_layout.compile(
        constants={"A": Layout.striped((size,), threads=128), "B": target},
        threads=128,
        device=device(),
    )
    values = np.arange(size, dtype=np.float32) / 512
    (actual,) = run_compiled(artifact, inputs=[values], outputs=[(size, np.float32)])
    owners = np.array([target.owner((i,)).group for i in range(size)], dtype=np.float32)
    np.testing.assert_array_equal(actual, values + owners)
    assert artifact.source.count("threadgroup_barrier(") == 4


@cache
def storage_exchange(element):
    @cl.kernel
    def exchange(
        source: cl.buffer(element),
        output: cl.buffer(element),
        A: cl.constexpr,
        B: cl.constexpr,
        SIZE: cl.constexpr,
    ):
        value = cl.load(source, cl.arange(A.size, layout=A), True, other=0)
        scratch = cl.scratch((SIZE,), element)
        value = cl.convert_layout(value, B, scratch=scratch)
        cl.store(output, cl.arange(B.size, layout=B), value, True)

    return exchange


@pytest.mark.parametrize(
    "element",
    [cl.f32, cl.f16, cl.bf16, cl.u32, cl.u16, cl.i16, cl.u8, cl.i8],
)
def test_typed_scratch_conversion_preserves_storage_bits(element):
    size = 257
    dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32}[element.itemsize]
    values = (np.arange(size, dtype=np.uint64) * 0x9E3779B9).astype(dtype)
    options = {
        "A": Layout.striped((size,), threads=64),
        "B": Layout.blocked((size,), threads=64),
        "SIZE": size,
    }
    kernel = storage_exchange(element)
    artifact = kernel.compile(constants=options, threads=64, device=device())
    assert (
        kernel.stage(constants=options, threads=64).ir.count('"carbon.convert_layout"')
        == 1
    )
    (actual,) = run_compiled(artifact, inputs=[values], outputs=[(size, dtype)])
    np.testing.assert_array_equal(actual, values)
    with pytest.raises(ValueError, match="conversion scratch must match"):
        kernel.compile(
            constants=dict(options, SIZE=size + 1),
            threads=64,
            device=device(),
        )


def test_scratch_limit_applies_after_fusion_and_footprint_planning():
    metal = device()
    size = metal.max_threadgroup_memory // 4 + 32
    layout = Layout.striped((size,), threads=256)
    options = {"A": layout, "B": Layout.blocked((size,), threads=256), "OBSERVE": False}
    fused = scratch_layout_loop.compile(
        constants=options,
        threads=256,
        device=metal,
        max_scratch=32,
    )
    reduced = reduction_item.compile(
        constants={"A": layout},
        threads=256,
        device=metal,
        max_scratch=32,
    )
    assert fused.scratch_bytes == 0 and reduced.scratch_bytes == 32
    values = (np.arange(size, dtype=np.float32) % 17) / 128
    (actual,) = run_compiled(
        fused,
        inputs=[values],
        outputs=[(size, np.float32)],
        scalars={"count": 1},
    )
    np.testing.assert_array_equal(actual, values * values + np.float32(0.25))
    (actual,) = run_compiled(reduced, inputs=[values], outputs=[(size, np.float32)])
    np.testing.assert_array_equal(actual, values + values.sum())
    with pytest.raises(ValueError, match="bytes of threadgroup memory"):
        scratch_layout_loop.compile(
            constants=dict(options, OBSERVE=True),
            threads=256,
            device=metal,
        )
