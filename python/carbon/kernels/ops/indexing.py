from functools import cache

from carbon import language as cl
from carbon.compiler import CopyAtom, Layout

SEGMENT_BLOCK, GATHER_BLOCK, COPY_BLOCK = 256, 1024, 4096


@cache
def make_segment_rows():
    schedule = Layout.blocked((SEGMENT_BLOCK,), threads=SEGMENT_BLOCK)

    @cl.kernel
    def segment_rows(
        offsets: cl.buffer(cl.u32),
        bounds: cl.buffer(cl.u32),
        rows: cl.u32,
        sequences: cl.u32,
        steps: cl.u32,
    ):
        row = cl.program_id() * SEGMENT_BLOCK + cl.arange(
            SEGMENT_BLOCK,
            layout=schedule,
        )
        low = cl.full((SEGMENT_BLOCK,), 0, cl.u32, layout=schedule)
        high = cl.full((SEGMENT_BLOCK,), sequences, cl.u32, layout=schedule)
        for step in range(steps):
            middle = (low + high) // 2
            ahead = cl.load(offsets, middle, True, other=0) <= row
            low = cl.select(ahead, middle, low)
            high = cl.select(ahead, high, middle)
        cl.store(bounds, row * 2, cl.load(offsets, low, True, other=0), row < rows)
        cl.store(
            bounds,
            row * 2 + 1,
            cl.load(offsets, low + 1, True, other=0),
            row < rows,
        )

    return segment_rows


@cache
def make_gather_rows(dtype=cl.bf16):
    layout = Layout.blocked((GATHER_BLOCK,), threads=256)

    @cl.kernel
    def gather_rows(
        source: cl.buffer(dtype),
        indices: cl.buffer(cl.u32),
        output: cl.buffer(dtype),
        rows: cl.u32,
        C: cl.constexpr,
    ):
        element = cl.program_id() * GATHER_BLOCK + cl.arange(
            GATHER_BLOCK,
            layout=layout,
        )
        live = element < rows * C
        row = cl.load(indices, element // C, live, other=0)
        cl.store(
            output,
            element,
            cl.load(source, row * C + element % C, live, other=0.0),
            live,
        )

    return gather_rows


def segment_rows(device, offsets, bounds, *, rows, sequences):
    """Dispatch: bounds[2r] and bounds[2r + 1], the first row and the row past the last
    of the sequence that holds row r, from `sequences` + 1 ascending u32 row offsets."""
    kernel = make_segment_rows().compile(
        constants={},
        threads=SEGMENT_BLOCK,
        device=device,
        max_grid=(2**32 - 1) // SEGMENT_BLOCK,
    )
    return kernel.dispatch(
        device,
        [offsets, bounds],
        grid=(-(-rows // SEGMENT_BLOCK), 1, 1),
        scalars={
            "rows": rows,
            "sequences": sequences,
            "steps": (sequences - 1).bit_length(),
        },
    )


def gather_rows(device, source, indices, output, *, rows, columns, dtype=cl.bf16):
    """Dispatch output row r = source row indices[r] for `rows` rows of `columns`
    elements and u32 indices, as a window partition, its inverse or an embedding lookup
    moves rows."""
    if rows < 1 or columns < 1 or rows * columns >= 2**32:
        raise ValueError("gather_rows takes rows and columns, under 2^32 elements")
    artifact = make_gather_rows(dtype).compile(
        constants={"C": columns},
        threads=256,
        device=device,
        max_grid=(2**32 - 1) // GATHER_BLOCK,
    )
    return [
        artifact.dispatch(
            device,
            [source, indices, output],
            grid=(-(-rows * columns // GATHER_BLOCK), 1, 1),
            scalars={"rows": rows},
        ),
    ]


@cache
def make_copy_slots():
    layout = CopyAtom(cl.u32, 4).layout((COPY_BLOCK,), threads=256)

    @cl.kernel
    def copy_slots(
        state: cl.buffer(cl.u32),
        sources: cl.buffer(cl.u32),
        targets: cl.buffer(cl.u32),
        WORDS: cl.constexpr,
        BLOCKS: cl.constexpr,
    ):
        pair = cl.program_id() // BLOCKS
        index = cl.program_id() % BLOCKS * COPY_BLOCK + cl.arange(
            COPY_BLOCK,
            layout=layout,
        )
        active = index < WORDS
        source = cl.load(sources, pair, True, other=0) * WORDS
        target = cl.load(targets, pair, True, other=0) * WORDS
        cl.store(
            state,
            target + index,
            cl.load(state, source + index, active, other=0),
            active,
        )

    return copy_slots


def copy_slots(device, state, sources, targets, *, count, slot_bytes):
    """Dispatch: slot targets[i] = slot sources[i] of a state buffer for i < count, from
    u32 slot buffers on the device, each slot `slot_bytes` bytes, a multiple of 4: a
    checkpoint of a sequence's state, or its restore. No target may be another pair's
    source."""
    if slot_bytes % 4:
        raise ValueError("slots copy as whole u32 words")
    words = slot_bytes // 4
    blocks = -(-words // COPY_BLOCK)
    artifact = make_copy_slots().compile(
        constants={"WORDS": words, "BLOCKS": blocks},
        threads=256,
        device=device,
    )
    return [
        artifact.dispatch(
            device,
            [state, sources, targets],
            grid=(count * blocks, 1, 1),
        ),
    ]


@cache
def make_copy_rows():
    layout = CopyAtom(cl.u32, 4).layout((COPY_BLOCK,), threads=256)

    def row_of(table, positions, pair, kv, HK, PAGE):
        position = cl.load(positions, pair, True, other=0)
        page = cl.load(table, position // PAGE, True, other=0)
        return (page * HK + kv) * PAGE + position % PAGE

    @cl.kernel
    def copy_rows(
        pages: cl.buffer(cl.u32),
        table: cl.buffer(cl.u32),
        sources: cl.buffer(cl.u32),
        targets: cl.buffer(cl.u32),
        count: cl.u32,
        HK: cl.constexpr,
        WORDS: cl.constexpr,
        PAGE: cl.constexpr,
    ):
        index = cl.program_id() * COPY_BLOCK + cl.arange(COPY_BLOCK, layout=layout)
        active = index < HK * WORDS
        kv, word = index // WORDS, index % WORDS
        for pair in range(count):
            source = row_of(table, sources, pair, kv, HK, PAGE) * WORDS + word
            target = row_of(table, targets, pair, kv, HK, PAGE) * WORDS + word
            cl.store(pages, target, cl.load(pages, source, active, other=0), active)

    return copy_rows


def copy_rows(
    device,
    pages,
    table,
    sources,
    targets,
    *,
    count,
    kv_heads,
    head_dim,
    page_size,
    dtype=cl.bf16,
):
    """Dispatch: position targets[i] = position sources[i] of one sequence's keys or
    values, for i < count in order, so a pair may overwrite a position that an earlier
    pair read. `pages` and `table` are as `paged_attention` takes them, for one
    sequence; sources and targets are u32 positions on the device. A tree verify uses
    it to move the accepted branch into place."""
    if head_dim * dtype.itemsize % 4:
        raise ValueError("rows copy as whole u32 words")
    words = head_dim * dtype.itemsize // 4
    artifact = make_copy_rows().compile(
        constants={"HK": kv_heads, "WORDS": words, "PAGE": page_size},
        threads=256,
        device=device,
    )
    return [
        artifact.dispatch(
            device,
            [pages, table, sources, targets],
            grid=(-(-kv_heads * words // COPY_BLOCK), 1, 1),
            scalars={"count": count},
        ),
    ]
