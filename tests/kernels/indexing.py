import numpy as np
import pytest

from carbon import language as cl
from carbon.kernels.ops import indexing
from testbed import device, storage, upload


def submit(metal, steps):
    metal.prepare(steps if isinstance(steps, list) else [steps]).submit().wait()


@pytest.mark.parametrize("columns", [300, 2056])
def test_gather_rows(columns):
    metal = device()
    rng = np.random.default_rng(columns)
    source = storage(rng.standard_normal((1000, columns)), cl.bf16)
    order = rng.integers(0, 1000, 300).astype(np.uint32)
    output = metal.buffer(300 * columns * 2)
    submit(
        metal,
        indexing.gather_rows(
            metal,
            upload(metal, source),
            upload(metal, order),
            output,
            rows=300,
            columns=columns,
        ),
    )
    assert np.array_equal(
        np.frombuffer(output.read(), np.uint16).reshape(300, columns),
        source[order],
    )


def test_copy_slots_moves_whole_slots():
    metal, slots, words = device(), 6, 64
    state = np.arange(slots * words, dtype=np.uint32).reshape(slots, words)
    buffer = upload(metal, state)
    submit(
        metal,
        indexing.copy_slots(
            metal,
            buffer,
            upload(metal, np.array([4, 1], np.uint32)),
            upload(metal, np.array([0, 5], np.uint32)),
            count=2,
            slot_bytes=words * 4,
        ),
    )
    expected = state.copy()
    expected[[0, 5]] = state[[4, 1]]
    assert np.array_equal(
        np.frombuffer(buffer.read(), np.uint32).reshape(slots, words),
        expected,
    )


def test_copy_rows_applies_pairs_in_order():
    metal, kv_heads, head_dim, page_size = device(), 2, 64, 16
    rng = np.random.default_rng(5)
    cache = storage(rng.standard_normal((5, kv_heads, page_size, head_dim)), cl.bf16)
    table = np.array([3, 0, 4], np.uint32)
    sources = np.array([18, 20, 17, 33, 7], np.uint32)
    targets = np.array([17, 18, 30, 19, 7], np.uint32)
    buffer = upload(metal, cache)
    submit(
        metal,
        indexing.copy_rows(
            metal,
            buffer,
            upload(metal, table),
            upload(metal, sources),
            upload(metal, targets),
            count=len(sources),
            kv_heads=kv_heads,
            head_dim=head_dim,
            page_size=page_size,
        ),
    )

    def at(position):
        return table[position // page_size], slice(None), position % page_size

    expected = cache.copy()
    for source, target in zip(sources, targets):
        expected[at(target)] = expected[at(source)]
    assert np.array_equal(
        np.frombuffer(buffer.read(), np.uint16).reshape(cache.shape),
        expected,
    )
