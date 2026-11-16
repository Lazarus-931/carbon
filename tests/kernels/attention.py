from dataclasses import dataclass
from itertools import product

import numpy as np
import pytest

import reference
from carbon import language as cl
from carbon.kernels.attention import (
    OutputMod,
    ScoreMod,
    Verify,
    attention,
    paged_attention,
)
from testbed import device, storage, upload, widen

PAGE = 16


@dataclass(frozen=True)
class Geometry:
    heads: int
    kv_heads: int
    depth: int
    window: int = 0

    def __str__(self):
        window = f"-w{self.window}" if self.window else ""
        return f"{self.heads}x{self.kv_heads}xD{self.depth}{window}"


CASES = [
    Geometry(8, 8, 64),
    Geometry(16, 4, 128),
    Geometry(16, 2, 128, window=100),
    Geometry(8, 1, 256),
    Geometry(4, 2, 512),
]
VERIFY = [
    (Geometry(16, 8, 128), 2),
    (Geometry(16, 8, 128), 4),
    (Geometry(16, 8, 128), 16),
    (Geometry(8, 8, 128), 9),
    (Geometry(32, 8, 128), 5),
    (Geometry(16, 2, 256), 7),
    (Geometry(16, 1, 128), 3),
    (Geometry(8, 2, 512), 4),
    (Geometry(16, 8, 128, window=100), 6),
    (Geometry(16, 8, 256, window=64), 12),
    (Geometry(8, 1, 256), 8),
]
SWEEP = [
    Geometry(heads, kv_heads, depth, window)
    for depth, (heads, kv_heads), window in product(
        (64, 128, 256, 512),
        ((4, 4), (8, 4), (16, 4), (40, 8), (16, 2), (16, 1)),
        (0, 1000),
    )
]


def paged(rng, caches, pages):
    """Caches (kv_heads, positions, depth) cut into shuffled pages, and the page
    table."""
    kv_heads, positions, depth = caches[0].shape
    order = rng.permutation(pages)
    stored = []
    for cache in caches:
        padded = np.zeros((kv_heads, pages * PAGE, depth), cache.dtype)
        padded[:, :positions] = cache
        blocks = padded.reshape(kv_heads, pages, PAGE, depth).transpose(1, 0, 2, 3)
        physical = np.zeros_like(blocks)
        physical[order] = blocks
        stored.append(np.ascontiguousarray(physical))
    return stored, order.astype(np.uint32)


def check_prefill(g, rows=37, plan=None):
    metal, dtype = device(), cl.bf16
    start = g.window + 21 if g.window else 211
    rng = np.random.default_rng(g.heads * g.depth + g.window)
    total = start + rows
    pages = -(-total // PAGE)
    q = rng.standard_normal((rows, g.heads, g.depth)) * 0.5
    caches = [rng.standard_normal((g.kv_heads, total, g.depth)) * s for s in (0.5, 1.0)]
    (k_pages, v_pages), table = paged(rng, [storage(c, dtype) for c in caches], pages)
    output = metal.buffer(rows * g.heads * g.depth * 2)
    metal.prepare(
        paged_attention(
            metal,
            upload(metal, storage(q, dtype)),
            upload(metal, k_pages),
            upload(metal, v_pages),
            upload(metal, table),
            upload(metal, np.array([start], np.uint32)),
            output,
            rows=rows,
            heads=g.heads,
            kv_heads=g.kv_heads,
            head_dim=g.depth,
            page_size=PAGE,
            window=g.window,
            dtype=dtype,
            plan=plan,
        ),
    ).submit().wait()
    expected = reference.attention(
        reference.rounded(q),
        *(reference.rounded(c) for c in caches),
        range(start, total),
        g.depth**-0.5,
        window=g.window,
    )
    actual = reference.read(output, dtype, rows, g.heads, g.depth)
    np.testing.assert_allclose(
        actual,
        expected,
        atol=3e-2,
        err_msg=f"prefill {g} rows {rows} {plan}",
    )


def check_decode(g, sequences=3):
    metal, dtype = device(), cl.bf16
    rng = np.random.default_rng(g.heads * g.depth + g.window + 1)
    max_pages = -(-(g.window + 85) // PAGE) if g.window else 20
    low = g.window + 1 if g.window else 1
    lengths = rng.integers(low, max_pages * PAGE, sequences).astype(np.uint32)
    q = rng.standard_normal((sequences, g.heads, g.depth)) * 0.5
    caches = [
        rng.standard_normal((g.kv_heads, sequences * max_pages * PAGE, g.depth))
        for _ in range(2)
    ]
    physical = [
        storage(c, dtype)
        .reshape(g.kv_heads, sequences * max_pages, PAGE, g.depth)
        .transpose(1, 0, 2, 3)
        for c in caches
    ]
    table = (
        rng.permutation(sequences * max_pages)
        .astype(np.uint32)
        .reshape(sequences, max_pages)
    )
    output = metal.buffer(sequences * g.heads * g.depth * 2)
    metal.prepare(
        paged_attention(
            metal,
            upload(metal, storage(q, dtype)),
            *(upload(metal, np.ascontiguousarray(p)) for p in physical),
            upload(metal, table),
            upload(metal, lengths),
            output,
            rows=sequences,
            heads=g.heads,
            kv_heads=g.kv_heads,
            head_dim=g.depth,
            page_size=PAGE,
            window=g.window,
            max_pages=max_pages,
            dtype=dtype,
        ),
    ).submit().wait()
    actual = reference.read(output, dtype, sequences, g.heads, g.depth)
    stored = [widen(p, dtype).astype(np.float64) for p in physical]
    widened = reference.rounded(q)
    for row in range(sequences):
        logical = [
            s[table[row]].transpose(1, 0, 2, 3).reshape(g.kv_heads, -1, g.depth)
            for s in stored
        ]
        expected = reference.attention(
            widened[row : row + 1],
            *logical,
            [int(lengths[row])],
            g.depth**-0.5,
            window=g.window,
        )
        np.testing.assert_allclose(
            actual[row : row + 1],
            expected,
            atol=3e-2,
            err_msg=f"decode {g} row {row}",
        )


def check_dense(g, causal, rows=40, positions=201, plan=None):
    metal, dtype = device(), cl.bf16
    rng = np.random.default_rng(g.heads * g.depth + positions)
    q = rng.standard_normal((rows, g.heads, g.depth)) * 0.5
    caches = [
        rng.standard_normal((positions, g.kv_heads, g.depth)) * s for s in (0.5, 1.0)
    ]
    output = metal.buffer(rows * g.heads * g.depth * 2)
    metal.prepare(
        attention(
            metal,
            upload(metal, storage(q, dtype)),
            *(upload(metal, storage(c, dtype)) for c in caches),
            output,
            rows=rows,
            positions=positions,
            heads=g.heads,
            kv_heads=g.kv_heads,
            head_dim=g.depth,
            causal=causal,
            window=g.window,
            dtype=dtype,
            plan=plan,
        ),
    ).submit().wait()
    keys, values = (reference.rounded(c).transpose(1, 0, 2) for c in caches)
    expected = reference.attention(
        reference.rounded(q),
        keys,
        values,
        range(positions - rows, positions),
        g.depth**-0.5,
        window=g.window,
        causal=causal,
    )
    actual = reference.read(output, dtype, rows, g.heads, g.depth)
    np.testing.assert_allclose(
        actual,
        expected,
        atol=3e-2,
        err_msg=f"dense {g} causal={causal} rows {rows} {plan}",
    )


@pytest.mark.parametrize("geometry", CASES, ids=str)
def test_paged_prefill(geometry):
    check_prefill(geometry)


@pytest.mark.parametrize("geometry", CASES, ids=str)
def test_paged_decode(geometry):
    check_decode(geometry)


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
@pytest.mark.parametrize("geometry", CASES[:3], ids=str)
def test_dense(geometry, causal):
    check_dense(geometry, causal)


@pytest.mark.sweep
@pytest.mark.parametrize("geometry", SWEEP, ids=str)
def test_paged_sweep(geometry):
    check_prefill(geometry)
    check_decode(geometry)


@pytest.mark.parametrize(
    ("geometry", "rows"),
    VERIFY,
    ids=[f"{geometry}-rows{rows}" for geometry, rows in VERIFY],
)
def test_paged_verify(geometry, rows):
    for plan in (None, Verify(1), Verify(24), Verify(5, 1), Verify(12, 2)):
        check_prefill(geometry, rows, plan)


def check_packed(g, counts, plan=None):
    metal, dtype = device(), cl.bf16
    rng = np.random.default_rng(g.heads * g.depth + sum(counts))
    sequences, rows, max_pages = len(counts), sum(counts), 20
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
    low = g.window + 1 if g.window else 1
    starts = rng.integers(low, max_pages * PAGE - max(counts), sequences)
    q = rng.standard_normal((rows, g.heads, g.depth)) * 0.5
    caches = [
        rng.standard_normal((g.kv_heads, sequences * max_pages * PAGE, g.depth))
        for _ in range(2)
    ]
    physical = [
        storage(c, dtype)
        .reshape(g.kv_heads, sequences * max_pages, PAGE, g.depth)
        .transpose(1, 0, 2, 3)
        for c in caches
    ]
    table = (
        rng.permutation(sequences * max_pages)
        .astype(np.uint32)
        .reshape(sequences, max_pages)
    )
    output = metal.buffer(rows * g.heads * g.depth * 2)
    metal.prepare(
        paged_attention(
            metal,
            upload(metal, storage(q, dtype)),
            *(upload(metal, np.ascontiguousarray(p)) for p in physical),
            upload(metal, table),
            upload(metal, starts.astype(np.uint32)),
            output,
            rows=rows,
            heads=g.heads,
            kv_heads=g.kv_heads,
            head_dim=g.depth,
            page_size=PAGE,
            window=g.window,
            max_pages=max_pages,
            segments=upload(metal, offsets),
            sequences=sequences,
            longest=max(counts),
            dtype=dtype,
            plan=plan,
        ),
    ).submit().wait()
    actual = reference.read(output, dtype, rows, g.heads, g.depth)
    stored = [widen(p, dtype).astype(np.float64) for p in physical]
    widened = reference.rounded(q)
    for sequence, (first, count) in enumerate(zip(offsets, counts)):
        logical = [
            s[table[sequence]].transpose(1, 0, 2, 3).reshape(g.kv_heads, -1, g.depth)
            for s in stored
        ]
        expected = reference.attention(
            widened[first : first + count],
            *logical,
            range(starts[sequence], starts[sequence] + count),
            g.depth**-0.5,
            window=g.window,
        )
        np.testing.assert_allclose(
            actual[first : first + count],
            expected,
            atol=3e-2,
            err_msg=f"packed {g} sequence {sequence} {plan}",
        )


PACKED = [
    (Geometry(16, 8, 128), (3, 0, 5, 1)),
    (Geometry(8, 1, 256), (4, 2)),
    (Geometry(8, 2, 512), (1, 7, 2)),
    (Geometry(16, 8, 128, window=100), (6, 3)),
]


@pytest.mark.parametrize(
    ("geometry", "counts"),
    PACKED,
    ids=[f"{geometry}-rows{'-'.join(map(str, c))}" for geometry, c in PACKED],
)
def test_packed_verify(geometry, counts):
    for plan in (None, Verify(1), Verify(3, 1), Verify(8, 2)):
        check_packed(geometry, counts, plan)


DENSE_VERIFY = [
    (Geometry(16, 8, 128), True, 4, 60),
    (Geometry(16, 4, 128), False, 5, 70),
    (Geometry(8, 2, 256, window=20), False, 3, 50),
    (Geometry(16, 8, 128, window=30), True, 16, 90),
]


@pytest.mark.parametrize(
    ("geometry", "causal", "rows", "positions"),
    DENSE_VERIFY,
    ids=[f"{g}-{'causal' if c else 'full'}-rows{r}" for g, c, r, _ in DENSE_VERIFY],
)
def test_dense_verify(geometry, causal, rows, positions):
    for plan in (None, Verify(1), Verify(8)):
        check_dense(geometry, causal, rows, positions, plan)


def _tilted(score, row, head, key):
    return score + cl.cast((key + row) % 7, cl.f32) * cl.cast(head + 1, cl.f32) * 0.03


def _scaled(value, row, head, column):
    return value * (1.0 + cl.cast(head % 3, cl.f32) * 0.25)


@pytest.mark.parametrize(
    ("geometry", "rows"),
    [(Geometry(16, 8, 128), 4), (Geometry(16, 4, 256, window=40), 6)],
    ids=["16x8xD128-rows4", "16x4xD256-w40-rows6"],
)
def test_verify_hooks(geometry, rows):
    g, metal, dtype = geometry, device(), cl.bf16
    start, rng = 150, np.random.default_rng(rows)
    total = start + rows
    q = rng.standard_normal((rows, g.heads, g.depth)) * 0.5
    caches = [rng.standard_normal((g.kv_heads, total, g.depth)) * s for s in (0.5, 1.0)]
    stored, table = paged(rng, [storage(c, dtype) for c in caches], -(-total // PAGE))
    expected = reference.attention(
        reference.rounded(q),
        *(reference.rounded(c) for c in caches),
        range(start, total),
        g.depth**-0.5,
        window=g.window,
        bias=lambda row, head, keys: (keys + row) % 7 * (head + 1) * 0.03,
    ) * (1.0 + np.arange(g.heads)[None, :, None] % 3 * 0.25)
    for plan in (None, Verify(1)):
        output = metal.buffer(rows * g.heads * g.depth * 2)
        metal.prepare(
            paged_attention(
                metal,
                upload(metal, storage(q, dtype)),
                *(upload(metal, s) for s in stored),
                upload(metal, table),
                upload(metal, np.array([start], np.uint32)),
                output,
                rows=rows,
                heads=g.heads,
                kv_heads=g.kv_heads,
                head_dim=g.depth,
                page_size=PAGE,
                window=g.window,
                score_mod=ScoreMod(_tilted),
                output_mod=OutputMod(_scaled),
                dtype=dtype,
                plan=plan,
            ),
        ).submit().wait()
        actual = reference.read(output, dtype, rows, g.heads, g.depth)
        np.testing.assert_allclose(actual, expected, atol=3e-2, err_msg=f"{g} {plan}")
