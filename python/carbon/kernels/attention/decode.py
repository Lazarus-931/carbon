from dataclasses import dataclass
from functools import cache, partial

from carbon import language as cl
from carbon.compiler import Atom, Layout, Mode, View
from carbon.kernels.attention.common import KEYS, scored, span
from carbon.kernels.common import applied, operand_buffers, place

MIN_KEYS, CHUNK = 64, 8


@dataclass(frozen=True)
class Decode:
    """Each row's keys in `splits` ranges, on matrix atoms with `tiles`."""

    splits: int
    tiles: bool


@dataclass(frozen=True)
class Verify:
    """The rows of one sequence together on matrix atoms, in `splits` ranges of keys,
    with up to `group` tiles of 8 query vectors per threadgroup: each key block is read
    once for every row and query head of its KV head, as a verify step of speculative
    decoding or a short prefill needs."""

    splits: int
    group: int = 8


def page_base(table, row, number, kv, HK, MAXP, PAGES, *, paged, page_size, D):
    if not paged:
        return number * KEYS
    within = page_size // KEYS
    page = cl.minimum(
        cl.load(table[0], row * MAXP + number // within, True, other=0),
        PAGES - 1,
    )
    return ((page * HK + kv) * page_size + number % within * KEYS) * D


def visible(lengths, bounds, row, rows, *, sequence, causal, window):
    if not sequence:
        return span(
            row,
            cl.load(lengths, row, True, other=0),
            0,
            rows,
            (),
            causal=True,
            window=window,
        )
    start = cl.load(lengths, 0, True, other=0)
    return span(row, start + row, start, rows, bounds, causal=causal, window=window)


def key_ranges(end, start, splits):
    first, blocks = start // KEYS, (end + KEYS - 1) // KEYS
    share = cl.maximum((blocks - first + splits - 1) // splits, MIN_KEYS // KEYS)
    return first, blocks, share, (blocks - first + share - 1) // share


@cache
def make_decode(
    dtype,
    *,
    head_dim,
    group,
    page_size,
    paged,
    causal,
    window,
    segmented,
    sequence,
    score_mod,
    splits,
):
    D, G, each = head_dim, group, head_dim // 32
    step = min(KEYS, max(1, 1024 // D))
    banded = segmented or bool(window)
    heads = Layout((G, D), Mode(G, D), Mode(32, each), Mode(each, 1))
    lines = Layout(
        (step, D),
        Mode(1, 0),
        Mode(32, each),
        Mode((step, each), (D, 1)),
    ).across_groups(G)
    product = Layout(
        (G, step, D),
        Mode(G, step * D),
        Mode(32, each),
        Mode((step, each), (D, 1)),
    )
    scores = product.reduced(2)
    rows_of = scores.reduced(1)

    block_base = partial(page_base, paged=paged, page_size=page_size, D=D)

    def element_at(base, pair, element, kv, HK, last):
        if paged:
            return base + pair * step * D + element
        return (
            cl.minimum(base + pair * step + element // D, last) * HK + kv
        ) * D + element % D

    @cl.kernel
    def decode(
        q: cl.buffer(dtype),
        keys: cl.buffer(dtype),
        values: cl.buffer(dtype),
        table: (cl.buffer(cl.u32),) if paged else (),
        lengths: cl.buffer(cl.u32),
        bounds: (cl.buffer(cl.u32),) if segmented else (),
        partials: cl.buffer(cl.f32),
        stats: cl.buffer(cl.f32),
        scores_ops: operand_buffers(score_mod),
        rows: cl.u32,
        HQ: cl.constexpr,
        HK: cl.constexpr,
        MAXP: cl.constexpr,
        CACHE: cl.constexpr,
        SCALE: cl.constexpr,
    ):
        PAGES = CACHE.shape[0] // (HK * page_size * D) if paged else 0
        kv = cl.program_id() % HK
        split = cl.program_id() // HK % splits
        row = cl.program_id() // HK // splits
        keys, values = cl.view(keys, CACHE), cl.view(values, CACHE)
        low, end = visible(
            lengths,
            bounds,
            row,
            rows,
            sequence=sequence,
            causal=causal,
            window=window,
        )
        last = cl.minimum(end - 1, CACHE.shape[0] // (HK * D) - 1)
        first, blocks, share, used = key_ranges(end, low, splits)
        working = split < used
        index = cl.arange(heads.size, layout=heads)
        loaded = cl.load(q, (row * HQ + kv * G) * D + index, working, other=0.0)
        queries = cl.broadcast_to(
            cl.cast(loaded, cl.f32),
            product.shape,
            axes=(0, 2),
            layout=product,
        )
        element = cl.arange(lines.size, layout=lines)
        line = cl.arange(scores.size, layout=scores)
        peak = cl.full(rows_of.shape, -cl.inf, cl.f32, layout=rows_of)
        denominator = cl.full(rows_of.shape, 0.0, cl.f32, layout=rows_of)
        total = cl.full(heads.shape, 0.0, cl.f32, layout=heads)
        for number in range(
            cl.minimum(first + split * share, blocks),
            cl.minimum(first + split * share + share, blocks),
        ):
            base = block_base(table, row, number, kv, HK, MAXP, PAGES)
            for pair in range(KEYS // step):
                offset = number * KEYS + pair * step
                position = offset + element // D
                live = (
                    (position < end) & (position >= low) if banded else position < end
                )
                at = element_at(base, pair, element, kv, HK, last)
                k = cl.cast(cl.load(keys, at, True, other=0.0), cl.f32)
                v = cl.cast(cl.load(values, at, live, other=0.0), cl.f32)
                product_sum = cl.sum(
                    queries
                    * cl.broadcast_to(k, product.shape, axes=(1, 2), layout=product),
                    2,
                    layout=scores,
                )
                score = scored(
                    score_mod,
                    product_sum,
                    SCALE,
                    row,
                    kv * G + line // step,
                    offset + line % step,
                    scores_ops,
                )
                seen = line % step + offset
                score = cl.select(
                    (seen < end) & (seen >= low) if banded else seen < end,
                    score,
                    -cl.inf,
                )
                next_peak = cl.maximum(peak, cl.max(score, 1, layout=rows_of))
                safe = cl.select(next_peak == -cl.inf, 0.0, next_peak)
                correction = cl.exp2(peak - safe)
                weight = cl.exp2(
                    score
                    - cl.broadcast_to(safe, scores.shape, axes=(0,), layout=scores),
                )
                denominator = denominator * correction + cl.sum(
                    weight,
                    1,
                    layout=rows_of,
                )
                mixed = cl.broadcast_to(
                    weight,
                    product.shape,
                    axes=(0, 1),
                    layout=product,
                ) * cl.broadcast_to(v, product.shape, axes=(1, 2), layout=product)
                total = total * cl.broadcast_to(
                    correction,
                    heads.shape,
                    axes=(0,),
                    layout=heads,
                ) + cl.sum(mixed, 1, layout=heads)
                peak = next_peak
        entry = (split * rows + row) * HQ + kv * G
        cl.store(partials, (entry + index // D) * D + index % D, total, working)
        head = cl.arange(rows_of.size, layout=rows_of)
        cl.store(stats, (entry + head) * 2, peak, working)
        cl.store(stats, (entry + head) * 2 + 1, denominator, working)

    return decode


@cache
def make_decode_tiles(
    dtype,
    *,
    head_dim,
    group,
    page_size,
    paged,
    causal,
    window,
    segmented,
    sequence,
    score_mod,
    splits,
    together=False,
    tiles=1,
    packed=False,
):
    D, G = head_dim, group
    slices, head_tiles = D // 128, G // 8
    threads = 32 * slices * tiles
    product, mix = Atom(dtype), Atom(cl.f32)
    score = product.layout(threads=threads)
    lines = score.reduced(1)
    steps = 128 // 8
    exchange = View(
        "f32",
        (8 * slices * tiles, KEYS),
        (KEYS, 1),
        8 * slices * tiles * KEYS * 4,
        address_space="threadgroup",
    )

    broadcast = partial(cl.broadcast_to, shape=score.shape, axes=(0,), layout=score)

    def heads_of(kv, head_tile):
        return kv * G + head_tile * 8 if head_tiles > 1 else kv * G

    def vectors(tile, local, kv, count):
        vector = tile * 8 + local
        return vector // G, kv * G + vector % G, vector < count * G

    def reach(row, segment):
        _, count, start = segment
        return span(row, start + row, start, count, (), causal=causal, window=window)

    def vector_masks(segment, tile, kv):
        row, head, _ = vectors(tile, place(score)[1], kv, segment[1])
        low, end = reach(row, segment)
        return segment[0] + row, head, low, end

    block_base = partial(page_base, paged=paged, page_size=page_size, D=D)

    def key_row(base, at, kv, HK, last):
        return base + at * D if paged else (cl.minimum(base + at, last) * HK + kv) * D

    def load_queries(q, row, kv, head_tile, HQ, live):
        simd, local, column = place(product.layout("a", threads=threads))
        first = (
            (row * HQ + heads_of(kv, head_tile) + simd // slices * 8 + local) * D
            + simd % slices * 128
            + column
        )
        return tuple(
            cl.load(q, first + 8 * step, live, other=0.0) for step in range(steps)
        )

    def load_vectors(q, tile, kv, segment, HQ, live):
        simd, local, column = place(product.layout("a", threads=threads))
        row, head, real = vectors(tile, local, kv, segment[1])
        first = ((segment[0] + row) * HQ + head) * D + simd % slices * 128 + 16 * column
        return tuple(
            cl.load(q, first + step, live & real, other=0.0) for step in range(steps)
        )

    def block(
        state,
        queries,
        keys,
        values,
        share,
        base,
        logical,
        low,
        end,
        last,
        scale,
        row,
        kv,
        head_tile,
        HK,
        scores_ops,
        masks=(),
    ):
        outputs, peak, denominator = state
        simd, local, column = place(score)
        offset = simd % slices * 128
        if together:
            scores = []
            for tile in range(2):
                line = key_row(base, 8 * tile + column, kv, HK, last) + offset
                loaded = [
                    cl.load(keys, line + 16 * local + step, True, other=0.0)
                    for step in range(steps)
                ]
                total = product.zeros()
                for step in range(steps):
                    total = product.mma(queries[step], loaded[step], total)
                scores.append(total)
        else:
            scores = [product.zeros(), product.zeros()]
            for step in range(steps):
                scores = [
                    product.mma(
                        queries[step],
                        cl.load(
                            keys,
                            key_row(base, 8 * tile + column, kv, HK, last)
                            + offset
                            + 8 * step
                            + local,
                            True,
                            other=0.0,
                        ),
                        total,
                    )
                    for tile, total in enumerate(scores)
                ]
        if slices > 1:
            cl.barrier()
            for tile, value in enumerate(scores):
                product.store(share, (cl.simd_id(), tile), value)
            cl.barrier()
            lead = cl.simd_id() // slices * slices
            scores = [
                sum(
                    (
                        product.load(share, (lead + other, tile), role="accumulator")
                        for other in range(1, slices)
                    ),
                    product.load(share, (lead, tile), role="accumulator"),
                )
                for tile in range(2)
            ]
        if not masks:
            head = heads_of(kv, head_tile) + simd // slices * 8 + local
            rows_of, heads, lows, ends = row, head, low, end
        else:
            rows_of, heads, lows, ends = masks
        seen = [logical + 8 * tile + column for tile in range(2)]
        scores = [
            cl.select(
                (at < ends) & (at >= lows),
                scored(score_mod, value, scale, rows_of, heads, at, scores_ops),
                -cl.inf,
            )
            for at, value in zip(seen, scores)
        ]
        next_peak = cl.maximum(
            peak,
            cl.max(cl.maximum(scores[0], scores[1]), 1, layout=lines),
        )
        safe = cl.select(next_peak == -cl.inf, 0.0, next_peak)
        correction = cl.exp2(peak - safe)
        weights = [cl.exp2(value - broadcast(safe)) for value in scores]
        denominator = denominator * correction + cl.sum(
            weights[0] + weights[1],
            1,
            layout=lines,
        )
        outputs = [output * broadcast(correction) for output in outputs]
        _, key, dim = place(mix.layout("b", threads=threads))
        for tile, weight in enumerate(weights):
            left = cl.convert_layout(
                weight,
                mix.layout("a", threads=threads),
                free=True,
            )
            position = logical + 8 * tile + key
            live = (position < end) & (position >= low)
            if together:
                at = key_row(base, 8 * tile + key, kv, HK, last) + offset + 16 * dim
                loaded = [
                    cl.load(values, at + step, live, other=0.0) for step in range(steps)
                ]
                outputs = [
                    mix.mma(left, cl.cast(value, cl.f32), output)
                    for value, output in zip(loaded, outputs)
                ]
            else:
                at = key_row(base, 8 * tile + key, kv, HK, last) + offset + dim
                outputs = [
                    mix.mma(
                        left,
                        cl.cast(
                            cl.load(values, at + 8 * step, live, other=0.0),
                            cl.f32,
                        ),
                        output,
                    )
                    for step, output in enumerate(outputs)
                ]
        return outputs, next_peak, denominator

    def finish(partials, stats, state, split, row, rows, kv, head_tile, HQ, working):
        outputs, peak, denominator = state
        simd, local, column = place(score)
        entry = (
            (split * rows + row) * HQ
            + heads_of(kv, head_tile)
            + simd // slices * 8
            + local
        )
        for step, value in enumerate(outputs):
            cl.store(
                partials,
                entry * D + simd % slices * 128 + 8 * step + column,
                value,
                working,
            )
        index = cl.arange(lines.size, layout=lines)
        line = (
            (split * rows + row) * HQ
            + heads_of(kv, head_tile)
            + index // 8 // slices * 8
            + index % 8
        )
        lead = working & (index // 8 % slices == 0)
        cl.store(stats, line * 2, peak, lead)
        cl.store(stats, line * 2 + 1, denominator, lead)

    def finish_vectors(
        partials,
        stats,
        state,
        split,
        rows,
        segment,
        kv,
        tile,
        HQ,
        working,
    ):
        outputs, peak, denominator = state
        origin, count, _ = segment
        simd, local, column = place(score)
        row, head, real = vectors(tile, local, kv, count)
        entry = (split * rows + origin + row) * HQ + head
        for step, value in enumerate(outputs):
            cl.store(
                partials,
                entry * D + simd % slices * 128 + 16 * column + step,
                value,
                working & real,
            )
        index = cl.arange(lines.size, layout=lines)
        row, head, real = vectors(tile, index % 8, kv, count)
        line = (split * rows + origin + row) * HQ + head
        lead = (True if packed else working) & real & (index // 8 % slices == 0)
        cl.store(stats, line * 2, peak, lead)
        cl.store(stats, line * 2 + 1, denominator, lead)

    @cl.kernel
    def decode_tiles(
        q: cl.buffer(dtype),
        keys: cl.buffer(dtype),
        values: cl.buffer(dtype),
        table: (cl.buffer(cl.u32),) if paged else (),
        lengths: cl.buffer(cl.u32),
        bounds: (cl.buffer(cl.u32),) if segmented else (),
        partials: cl.buffer(cl.f32),
        stats: cl.buffer(cl.f32),
        scores_ops: operand_buffers(score_mod),
        rows: cl.u32,
        HQ: cl.constexpr,
        HK: cl.constexpr,
        MAXP: cl.constexpr,
        CACHE: cl.constexpr,
        SCALE: cl.constexpr,
        LONGEST: cl.constexpr,
    ):
        PAGES = CACHE.shape[0] // (HK * page_size * D) if paged else 0
        if together:
            bands = (((LONGEST if packed else rows) * G + 7) // 8 + tiles - 1) // tiles
            head_tile = cl.program_id() % bands * tiles + cl.simd_id() // slices
            kv = cl.program_id() // bands % HK
            split = cl.program_id() // bands // HK % splits
            row = cl.program_id() // bands // HK // splits
        elif head_tiles > 1:
            head_tile = cl.program_id() % head_tiles
            kv = cl.program_id() // head_tiles % HK
            split = cl.program_id() // head_tiles // HK % splits
            row = cl.program_id() // head_tiles // HK // splits
        else:
            head_tile = 0
            kv = cl.program_id() % HK
            split = cl.program_id() // HK % splits
            row = cl.program_id() // HK // splits
        keys, values = cl.view(keys, CACHE), cl.view(values, CACHE)
        if together:
            segment = (0, rows, cl.load(lengths, 0, True, other=0))
            if packed:
                origin = cl.load(bounds[0], row, True, other=0)
                segment = (
                    origin,
                    cl.load(bounds[0], row + 1, True, other=0) - origin,
                    cl.load(lengths, row, True, other=0),
                )
            low = reach(0, segment)[0]
            end = reach(segment[1] - 1, segment)[1]
        else:
            low, end = visible(
                lengths,
                bounds,
                row,
                rows,
                sequence=sequence,
                causal=causal,
                window=window,
            )
        last = cl.minimum(end - 1, CACHE.shape[0] // (HK * D) - 1)
        first, blocks, share, used = key_ranges(end, low, splits)
        working = split < used
        exchanged = cl.view(
            cl.scratch((8 * slices * tiles * KEYS,), cl.f32),
            exchange,
        )
        if together:
            queries = load_vectors(q, head_tile, kv, segment, HQ, working)
            masks = vector_masks(segment, head_tile, kv)
        else:
            queries = load_queries(q, row, kv, head_tile, HQ, working)
            masks = ()
        state = (
            [mix.zeros() for _ in range(steps)],
            cl.full(lines.shape, -cl.inf, cl.f32, layout=lines),
            cl.full(lines.shape, 0.0, cl.f32, layout=lines),
        )
        for number in range(
            cl.minimum(first + split * share, blocks),
            cl.minimum(first + split * share + share, blocks),
        ):
            base = block_base(table, row, number, kv, HK, MAXP, PAGES)
            state = block(
                state,
                queries,
                keys,
                values,
                exchanged,
                base,
                number * KEYS,
                low,
                end,
                last,
                SCALE,
                row,
                kv,
                head_tile,
                HK,
                scores_ops,
                masks,
            )
        if together:
            finish_vectors(
                partials,
                stats,
                state,
                split,
                rows,
                segment,
                kv,
                head_tile,
                HQ,
                working,
            )
        else:
            finish(partials, stats, state, split, row, rows, kv, head_tile, HQ, working)

    return decode_tiles


def chunk_peak(stats, first, size, stride, base, used, peak):
    """The running peak over `size` splits from `first`, their loads issued together."""
    entries = [first + offset for offset in range(size)]
    tops = [
        cl.load(stats, (entry * stride + base) * 2, entry < used, other=-cl.inf)
        for entry in entries
    ]
    for top in tops:
        peak = cl.maximum(peak, top)
    return peak


def chunk_sum(partials, stats, first, size, stride, base, used, column, D, safe, sums):
    """`size` more splits added, in split order, into the running total and
    denominator; their loads are issued together."""
    total, denominator = sums
    lines = [(first + offset) * stride + base for offset in range(size)]
    live = [first + offset < used for offset in range(size)]
    tops = [
        cl.load(stats, line * 2, mask, other=-cl.inf) for line, mask in zip(lines, live)
    ]
    shares = [
        cl.load(stats, line * 2 + 1, mask, other=0.0) for line, mask in zip(lines, live)
    ]
    parts = [
        cl.load(partials, line * D + column, top > -cl.inf, other=0.0)
        for line, top in zip(lines, tops)
    ]
    for top, share, part in zip(tops, shares, parts):
        scale = cl.exp2(top - safe)
        denominator = denominator + share * scale
        total = total + part * scale
    return total, denominator


@cache
def make_merge(
    dtype,
    *,
    causal,
    window,
    segmented,
    sequence,
    output_mod,
    together=False,
    packed=False,
):
    def used_splits(lengths, bounds, row, rows, entries):
        if packed:
            return entries
        reach = partial(
            visible,
            lengths,
            bounds,
            sequence=sequence,
            causal=causal,
            window=window,
        )
        if together:
            low, end = reach(0, rows)[0], reach(rows - 1, rows)[1]
        else:
            low, end = reach(row, rows)
        return key_ranges(end, low, entries)[3]

    @cl.kernel
    def merge(
        partials: cl.buffer(cl.f32),
        stats: cl.buffer(cl.f32),
        lengths: cl.buffer(cl.u32),
        bounds: (cl.buffer(cl.u32),) if segmented else (),
        output: cl.buffer(dtype),
        outs: operand_buffers(output_mod),
        rows: cl.u32,
        HQ: cl.constexpr,
        D: cl.constexpr,
        ENTRIES: cl.constexpr,
        SCHEDULE: cl.constexpr,
    ):
        head = cl.program_id() % HQ
        row = cl.program_id() // HQ
        column = cl.arange(D, layout=SCHEDULE)
        used = used_splits(lengths, bounds, row, rows, ENTRIES)
        stride, base, size = rows * HQ, row * HQ + head, min(CHUNK, ENTRIES)
        sums = (cl.full((D,), 0.0, cl.f32, layout=SCHEDULE), 0.0)
        if ENTRIES <= CHUNK:
            peak = chunk_peak(stats, 0, size, stride, base, used, -cl.inf)
            safe = cl.select(peak == -cl.inf, 0.0, peak)
            total, denominator = chunk_sum(
                partials,
                stats,
                0,
                size,
                stride,
                base,
                used,
                column,
                D,
                safe,
                sums,
            )
        else:
            peak = -cl.inf
            for first in range(0, ENTRIES, size):
                peak = chunk_peak(stats, first, size, stride, base, used, peak)
            safe = cl.select(peak == -cl.inf, 0.0, peak)
            total, denominator = sums
            for first in range(0, ENTRIES, size):
                total, denominator = chunk_sum(
                    partials,
                    stats,
                    first,
                    size,
                    stride,
                    base,
                    used,
                    column,
                    D,
                    safe,
                    (total, denominator),
                )
        result = applied(output_mod, total / denominator, row, head, column, outs)
        cl.store(output, cl.program_id() * D + column, cl.cast(result, dtype), True)

    return merge


def decode_steps(device, call, plan):
    sequence = not call.max_pages
    together = isinstance(plan, Verify)
    packed = together and call.segmented
    if together and (bool(call.max_pages) != call.segmented or call.head_dim % 128):
        raise ValueError(
            "Verify takes the rows of one sequence, or sequences packed with max_pages "
            "and segments, and head_dim a multiple of 128",
        )
    most = (2**32 - 1) // (plan.splits * call.heads * call.head_dim)
    if call.rows > most:
        raise ValueError(
            f"decode takes at most {most} rows of these heads, for u32 indexing",
        )
    partials = device.buffer(plan.splits * call.rows * call.heads * call.head_dim * 4)
    stats = device.buffer(plan.splits * call.rows * call.heads * 8)
    slices = call.head_dim // 128
    shape = {"together": True, "packed": True} if packed else {"together": True}
    if together:
        count = -(-(call.longest if packed else call.rows) * call.group // 8)
        group = min(count, plan.group)
        bands = -(-count // group)
        build, options = make_decode_tiles, {**shape, "tiles": group}
        threads = 32 * slices * group
        if packed:
            bound, grid = most * bands, call.sequences * bands
        else:
            bound, grid = -(-(-(-most * call.group // 8)) // group), bands
        grid *= call.kv_heads * plan.splits
    else:
        head_tiles = call.group // 8 if plan.tiles else 1
        build, options = (make_decode_tiles if plan.tiles else make_decode), {}
        threads = 32 * slices if plan.tiles else 32 * call.group
        bound = most * head_tiles
        grid = call.rows * call.kv_heads * head_tiles * plan.splits
    decode = build(
        call.dtype,
        **call.options,
        sequence=sequence,
        score_mod=call.score_mod,
        splits=plan.splits,
        **options,
    ).compile(
        constants={
            "HQ": call.heads,
            "HK": call.kv_heads,
            "MAXP": call.max_pages,
            "CACHE": call.cache,
            "SCALE": call.factor,
            **(
                {"LONGEST": call.longest if packed else 0}
                if together or plan.tiles
                else {}
            ),
        },
        threads=threads,
        device=device,
        max_grid=bound * call.kv_heads * plan.splits,
    )
    width = min(128, call.head_dim)
    merge = make_merge(
        call.dtype,
        causal=call.causal,
        window=call.window,
        segmented=call.segmented,
        sequence=sequence,
        output_mod=call.output_mod,
        **(shape if together else {}),
    ).compile(
        constants={
            "HQ": call.heads,
            "D": call.head_dim,
            "ENTRIES": plan.splits,
            "SCHEDULE": Layout.striped((call.head_dim,), threads=width),
        },
        threads=width,
        device=device,
        max_grid=most * call.heads,
    )
    scalars = {"rows": call.rows}
    return [
        decode.dispatch(
            device,
            [
                call.q,
                call.keys,
                call.values,
                *call.tables,
                call.lengths,
                *call.cuts,
                partials,
                stats,
                *call.score_operands,
            ],
            grid=(grid, 1, 1),
            scalars=scalars,
        ),
        merge.dispatch(
            device,
            [
                partials,
                stats,
                call.lengths,
                *call.cuts,
                call.output,
                *call.output_operands,
            ],
            grid=(call.rows * call.heads, 1, 1),
            scalars=scalars,
        ),
    ]
