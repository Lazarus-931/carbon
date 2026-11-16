from dataclasses import dataclass
from functools import cache, partial, reduce

from carbon import language as cl
from carbon.compiler import Atom, CopyAtom, Layout, TiledAtom, View
from carbon.kernels import shared_workspace
from carbon.kernels.attention.common import KEYS, ceil_div, scored, span
from carbon.kernels.common import applied, operand_buffers, place

PADDING = 8
TILE_KEYS = 64
TILE_GROUPS, TILE_FRAGMENTS = (4, 2), (4, 4)
WEIGHT_GROUPS, WEIGHT_FRAGMENTS = (4, 1), (4, 4)
PASS_BYTES, PASS_ROWS = 128 << 20, 2048
WEIGHT_TARGET, WEIGHT_SPLITS, SUM_BLOCK = 256, 8, 1024


@dataclass(frozen=True)
class Prefill:
    """One fused kernel on `simds` SIMD groups, staging `block_keys` keys at a time,
    with the partly visible key blocks in loops of their own when `edges`."""

    simds: int
    block_keys: int
    edges: bool


@dataclass(frozen=True)
class LongHead:
    """A scores product, softmax factors and a weighted product per pass of query rows,
    through a workspace."""


def line_span(line, first, count, start, rows, bounds, *, group, causal, window):
    row = first + cl.minimum(line, count * group - 1) // group
    return span(row, start + row, start, rows, bounds, causal=causal, window=window)


def block_band(
    block,
    height,
    first,
    count,
    start,
    rows,
    bounds,
    *,
    group,
    causal,
    window,
):
    last = cl.minimum(block * height + height, count * group) - 1
    top = (
        line_span(
            last,
            first,
            count,
            start,
            rows,
            bounds,
            group=group,
            causal=causal,
            window=window,
        )[1]
        - 1
    )
    if not (bounds or window):
        return 0, top
    low = line_span(
        block * height,
        first,
        count,
        start,
        rows,
        bounds,
        group=group,
        causal=causal,
        window=window,
    )[0]
    return low // TILE_KEYS * TILE_KEYS, top


@cache
def make_prefill(
    dtype,
    *,
    head_dim,
    group,
    page_size,
    paged,
    causal,
    window,
    segmented,
    score_mod,
    output_mod,
    plan,
):
    D, G, B, simds = head_dim, group, plan.block_keys, plan.simds
    banded = segmented or bool(window)
    product, mix, threads = Atom(dtype), Atom(cl.f32), 32 * simds
    score = product.layout(threads=threads)
    lines = score.reduced(1)
    steps, height = D // 8, 8 * simds
    keys_view = View(
        dtype.name,
        (B, D),
        (1, B + PADDING),
        D * (B + PADDING) * dtype.itemsize,
        address_space="threadgroup",
    )
    key_tiles = View(
        dtype.name,
        (D, B),
        (B + PADDING, 1),
        D * (B + PADDING) * dtype.itemsize,
        address_space="threadgroup",
    )
    values_view = View(
        dtype.name,
        (B, D),
        (D + PADDING, 1),
        B * (D + PADDING) * dtype.itemsize,
        address_space="threadgroup",
    )

    broadcast = partial(cl.broadcast_to, shape=score.shape, axes=(0,), layout=score)

    def tile_row(index):
        return index // 64 * 8 + index // 8 % 8

    def load_queries(q, first, rows, HQ, kv):
        a = cl.arange(64 * simds, layout=product.layout("a", threads=threads))
        line = first + tile_row(a)
        row = line // G
        at = (row * HQ + kv * G + line % G) * D + a % 8
        return tuple(
            cl.load(q, at + 8 * step, row < rows, other=0.0) for step in range(steps)
        )

    def within(key, low, end):
        return (key < end) & (key >= low) if banded else key < end

    def addresses(table, logical, kv, HK, spread, limit):
        if not paged:
            return (cl.minimum(logical + spread // D, limit) * HK + kv) * D + spread % D
        first = (
            (cl.load(table[0], logical // page_size, True, other=0) * HK + kv)
            * page_size
            + logical % page_size
        ) * D
        if B == KEYS:
            return first + spread
        upper = logical + KEYS
        second = (
            (cl.load(table[0], upper // page_size, True, other=0) * HK + kv) * page_size
            + upper % page_size
        ) * D - KEYS * D
        return cl.select(spread < KEYS * D, first + spread, second + spread)

    def block(
        state,
        queries,
        held_keys,
        held_values,
        keys,
        values,
        table,
        index,
        kv,
        HK,
        spread,
        limit,
        column,
        low,
        end,
        row,
        head,
        masked,
        scale,
        scores_ops,
    ):
        outputs, peak, denominator = state
        logical = index * B
        at = addresses(table, logical, kv, HK, spread, limit)
        cl.store(
            cl.view(held_keys, keys_view),
            spread,
            cl.load(keys, at, True, other=0.0),
            True,
        )
        cl.barrier()
        tiles = cl.view(held_keys, key_tiles)
        scores = [product.zeros() for _ in range(B // 8)]
        for step in range(steps):
            fragments = [
                product.load(tiles, (step, tile), role="b") for tile in range(B // 8)
            ]
            scores = [
                product.mma(queries[step], fragment, value)
                for fragment, value in zip(fragments, scores)
            ]
        cl.store(
            cl.view(held_values, values_view),
            spread,
            cl.load(values, at, True, other=0.0),
            True,
        )
        scores = [
            scored(
                score_mod,
                value,
                scale,
                row,
                head,
                logical + 8 * tile + column,
                scores_ops,
            )
            for tile, value in enumerate(scores)
        ]
        if masked:
            scores = [
                cl.select(within(logical + 8 * tile + column, low, end), value, -cl.inf)
                for tile, value in enumerate(scores)
            ]
        next_peak = cl.maximum(
            peak,
            cl.max(reduce(cl.maximum, scores), 1, layout=lines),
        )
        safe = cl.select(next_peak == -cl.inf, 0.0, next_peak)
        correction = cl.exp2(peak - safe)
        weights = [cl.exp2(value - broadcast(safe)) for value in scores]
        denominator = denominator * correction + cl.sum(
            reduce(lambda a, b: a + b, weights),
            1,
            layout=lines,
        )
        outputs = [output * broadcast(correction) for output in outputs]
        cl.barrier()
        staged = cl.view(held_values, values_view)
        for tile, weight in enumerate(weights):
            left = cl.convert_layout(
                weight,
                mix.layout("a", threads=threads),
                free=True,
            )
            outputs = [
                mix.mma(left, mix.load(staged, (tile, step), role="b"), output)
                for step, output in enumerate(outputs)
            ]
        return outputs, next_peak, denominator

    def store(output, outs, outputs, inverse, first, rows, HQ, kv):
        for step, total in enumerate(outputs):
            index = cl.arange(64 * simds, layout=total.layout)
            line = first + tile_row(index)
            row, head = line // G, kv * G + line % G
            column = 8 * step + index % 8
            result = applied(output_mod, total * inverse, row, head, column, outs)
            cl.store(
                output,
                (row * HQ + head) * D + column,
                cl.cast(result, dtype),
                row < rows,
            )

    def visible(row, start, rows, bounds):
        return span(row, start + row, start, rows, bounds, causal=causal, window=window)

    @cl.kernel
    def prefill(
        q: cl.buffer(dtype),
        keys: cl.buffer(dtype),
        values: cl.buffer(dtype),
        table: (cl.buffer(cl.u32),) if paged else (),
        lengths: cl.buffer(cl.u32),
        bounds: (cl.buffer(cl.u32),) if segmented else (),
        output: cl.buffer(dtype),
        scores_ops: operand_buffers(score_mod),
        outs: operand_buffers(output_mod),
        rows: cl.u32,
        HQ: cl.constexpr,
        HK: cl.constexpr,
        CACHE: cl.constexpr,
        SCALE: cl.constexpr,
    ):
        BLOCKS = (rows * G + height - 1) // height
        keys, values = cl.view(keys, CACHE), cl.view(values, CACHE)
        kv = cl.program_id() // BLOCKS
        first = cl.program_id() % BLOCKS * height
        start = cl.load(lengths, 0, True, other=0)
        limit = cl.minimum(start + rows - 1, CACHE.shape[0] // (HK * D) - 1)
        held_keys = cl.scratch((D * (B + PADDING),), dtype)
        held_values = cl.scratch((B * (D + PADDING),), dtype)
        spread = cl.arange(
            B * D,
            layout=CopyAtom(dtype, 4).layout((B, D), threads=threads),
        )
        queries = load_queries(q, first, rows, HQ, kv)
        element = cl.arange(64 * simds, layout=score)
        column = element % 8
        line = first + tile_row(element)
        row, head = line // G, kv * G + line % G
        low, end = visible(row, start, rows, bounds)
        head_low, head_end = visible(first // G, start, rows, bounds)
        tail_low, tail_end = visible(
            cl.minimum(first + height - 1, rows * G - 1) // G,
            start,
            rows,
            bounds,
        )
        begin, stop = head_low // B, ceil_div(tail_end, B)
        state = (
            [mix.zeros() for _ in range(steps)],
            cl.full(lines.shape, -cl.inf, cl.f32, layout=lines),
            cl.full(lines.shape, 0.0, cl.f32, layout=lines),
        )
        if plan.edges:
            whole = cl.minimum(cl.maximum((tail_low + B - 1) // B, begin), stop)
            clean = cl.maximum(cl.minimum(head_end // B, stop), whole)
            for index in range(begin, whole):
                state = block(
                    state,
                    queries,
                    held_keys,
                    held_values,
                    keys,
                    values,
                    table,
                    index,
                    kv,
                    HK,
                    spread,
                    limit,
                    column,
                    low,
                    end,
                    row,
                    head,
                    True,
                    SCALE,
                    scores_ops,
                )
            for index in range(whole, clean):
                state = block(
                    state,
                    queries,
                    held_keys,
                    held_values,
                    keys,
                    values,
                    table,
                    index,
                    kv,
                    HK,
                    spread,
                    limit,
                    column,
                    low,
                    end,
                    row,
                    head,
                    False,
                    SCALE,
                    scores_ops,
                )
            for index in range(clean, stop):
                state = block(
                    state,
                    queries,
                    held_keys,
                    held_values,
                    keys,
                    values,
                    table,
                    index,
                    kv,
                    HK,
                    spread,
                    limit,
                    column,
                    low,
                    end,
                    row,
                    head,
                    True,
                    SCALE,
                    scores_ops,
                )
        else:
            for index in range(begin, stop):
                state = block(
                    state,
                    queries,
                    held_keys,
                    held_values,
                    keys,
                    values,
                    table,
                    index,
                    kv,
                    HK,
                    spread,
                    limit,
                    column,
                    low,
                    end,
                    row,
                    head,
                    True,
                    SCALE,
                    scores_ops,
                )
        outputs, _, denominator = state
        store(output, outs, outputs, broadcast(1.0 / denominator), first, rows, HQ, kv)

    return prefill


@cache
def make_scores(
    dtype,
    *,
    head_dim,
    group,
    page_size,
    paged,
    causal,
    window,
    segmented,
    score_mod,
):
    D, G = head_dim, group
    banded = segmented or bool(window)
    (_, GN), (FM, FN) = TILE_GROUPS, TILE_FRAGMENTS
    atom = Atom(dtype)
    tiled = TiledAtom(atom, TILE_GROUPS, TILE_FRAGMENTS)
    BM, BN = tiled.shape
    simds = tiled.threads // 32
    score = atom.layout(threads=tiled.threads)
    lines_of = score.reduced(1)
    exchange = View(
        "f32",
        (8 * simds * FM, 8),
        (8, 1),
        8 * simds * FM * 8 * 4,
        address_space="threadgroup",
    )

    broadcast = partial(cl.broadcast_to, shape=score.shape, axes=(0,), layout=score)

    visible = partial(line_span, group=G, causal=causal, window=window)

    def key_at(table, key, kv, HK):
        if not paged:
            return (key * HK + kv) * D
        page = cl.load(table[0], key // page_size, True, other=0)
        return ((page * HK + kv) * page_size + key % page_size) * D

    def operands(q, table, block, origin, top, first, count, kv, HQ, HK):
        simd, local, column = place(score)
        lines = [
            cl.minimum(block * BM + (simd // GN * FM + i) * 8 + local, count * G - 1)
            for i in range(FM)
        ]
        queries = tuple(
            ((first + line // G) * HQ + kv * G + line % G) * D + column
            for line in lines
        )
        bases = tuple(
            key_at(
                table,
                cl.minimum(origin + (simd % GN * FN + j) * 8 + column, top),
                kv,
                HK,
            )
            + local
            for j in range(FN)
        )
        return queries, bases

    def fragments(buffer, bases, step):
        return tuple(cl.load(buffer, at + 8 * step, True, other=0.0) for at in bases)

    def combine(share, parts, operation):
        lead = cl.simd_id() // GN * GN
        cl.barrier()
        for i, part in enumerate(parts):
            atom.store(share, (cl.simd_id() * FM + i, 0), part)
        cl.barrier()
        return [
            reduce(
                operation,
                [
                    atom.load(share, ((lead + other) * FM + i, 0), role="accumulator")
                    for other in range(GN)
                ],
            )
            for i in range(FM)
        ]

    def finish(
        accumulators,
        share,
        weights,
        stats,
        number,
        origin,
        base,
        first,
        count,
        start,
        rows,
        bounds,
        block,
        kv,
        LINES,
        WIDTH,
        TILES,
        scale,
        scores_ops,
    ):
        simd, local, column = place(score)
        lines = [block * BM + (simd // GN * FM + i) * 8 + local for i in range(FM)]
        keys = [origin + (simd % GN * FN + j) * 8 + column for j in range(FN)]
        scores = []
        for line, values in zip(lines, accumulators):
            low, end = visible(line, first, count, start, rows, bounds)
            row, head = first + line // G, kv * G + line % G
            scores.append(
                [
                    cl.select(
                        (key < end) & (key >= low) if banded else key < end,
                        scored(score_mod, value, scale, row, head, key, scores_ops),
                        -cl.inf,
                    )
                    for key, value in zip(keys, values)
                ],
            )
        peaks = [
            cl.max(value, 1, layout=lines_of)
            for value in combine(
                share,
                [reduce(cl.maximum, row) for row in scores],
                cl.maximum,
            )
        ]
        safe = [cl.select(peak == -cl.inf, 0.0, peak) for peak in peaks]
        exponents = [
            [cl.exp2(value - broadcast(peak)) for value in row]
            for peak, row in zip(safe, scores)
        ]
        sums = [
            cl.sum(value, 1, layout=lines_of)
            for value in combine(
                share,
                [reduce(lambda a, b: a + b, row) for row in exponents],
                lambda a, b: a + b,
            )
        ]
        for line, row in zip(lines, exponents):
            for key, value in zip(keys, row):
                cl.store(
                    weights,
                    (kv * LINES + line) * WIDTH + key - base,
                    cl.cast(value, dtype),
                    line < count * G,
                )
        index = cl.arange(lines_of.size, layout=lines_of)
        for i, (peak, total) in enumerate(zip(peaks, sums)):
            line = block * BM + (index // 8 // GN * FM + i) * 8 + index % 8
            at = ((kv * TILES + number) * LINES + line) * 2
            lead = (line < count * G) & (index // 8 % GN == 0)
            cl.store(stats, at, peak, lead)
            cl.store(stats, at + 1, total, lead)

    @cl.kernel
    def scores(
        q: cl.buffer(dtype),
        keys: cl.buffer(dtype),
        table: (cl.buffer(cl.u32),) if paged else (),
        lengths: cl.buffer(cl.u32),
        bounds: (cl.buffer(cl.u32),) if segmented else (),
        weights: cl.buffer(dtype),
        stats: cl.buffer(cl.f32),
        scores_ops: operand_buffers(score_mod),
        rows: cl.u32,
        first: cl.u32,
        count: cl.u32,
        HQ: cl.constexpr,
        HK: cl.constexpr,
        CACHE: cl.constexpr,
        SCALE: cl.constexpr,
        LINES: cl.constexpr,
        WIDTH: cl.constexpr,
    ):
        TILES = WIDTH // BN
        BLOCKS = (count * G + BM - 1) // BM
        keys = cl.view(keys, CACHE)
        number = cl.program_id() % TILES
        block = cl.program_id() // TILES % BLOCKS
        kv = cl.program_id() // TILES // BLOCKS
        start = cl.load(lengths, 0, True, other=0)
        base, top = block_band(
            block,
            BM,
            first,
            count,
            start,
            rows,
            bounds,
            group=G,
            causal=causal,
            window=window,
        )
        origin = base + number * BN
        share = cl.view(cl.scratch((8 * simds * FM * 8,), cl.f32), exchange)
        if origin <= top:
            queries, bases = operands(
                q,
                table,
                block,
                origin,
                top,
                first,
                count,
                kv,
                HQ,
                HK,
            )
            accumulators = tiled.zeros()
            for step in range(D // 8):
                accumulators = tiled.mma(
                    fragments(q, queries, step),
                    fragments(keys, bases, step),
                    accumulators,
                )
            finish(
                accumulators,
                share,
                weights,
                stats,
                number,
                origin,
                base,
                first,
                count,
                start,
                rows,
                bounds,
                block,
                kv,
                LINES,
                WIDTH,
                TILES,
                SCALE,
                scores_ops,
            )

    return scores


@cache
def make_factors(*, group, causal, window, segmented):
    G, slices = group, 8
    banded = segmented or bool(window)
    height = WEIGHT_GROUPS[0] * WEIGHT_FRAGMENTS[0] * 8
    schedule = Layout.striped((slices, 32), threads=32 * slices)

    visible = partial(line_span, group=G, causal=causal, window=window)

    def within(live, tile, seen, lower):
        return (
            live & (tile < seen) & (tile >= lower) if banded else live & (tile < seen)
        )

    @cl.kernel
    def factors(
        stats: cl.buffer(cl.f32),
        lengths: cl.buffer(cl.u32),
        scales: cl.buffer(cl.f32),
        bounds: (cl.buffer(cl.u32),) if segmented else (),
        rows: cl.u32,
        first: cl.u32,
        count: cl.u32,
        LINES: cl.constexpr,
        TILES: cl.constexpr,
    ):
        BLOCKS = (count * G + 31) // 32
        block = cl.program_id() % BLOCKS
        kv = cl.program_id() // BLOCKS
        start = cl.load(lengths, 0, True, other=0)
        index = cl.arange(slices * 32, layout=schedule)
        part = index // 32
        line = block * 32 + index % 32
        live = line < count * G
        lead = block * 32 // height * height
        low, end = visible(line, first, count, start, rows, bounds)
        base = (
            visible(lead, first, count, start, rows, bounds)[0] // TILE_KEYS
            if banded
            else 0
        )
        lower = low // TILE_KEYS - base if banded else 0
        seen = (end - 1) // TILE_KEYS - base + 1
        last = visible(
            cl.minimum(block * 32 + 32, count * G) - 1,
            first,
            count,
            start,
            rows,
            bounds,
        )[1]
        reader = visible(
            cl.minimum(lead + height, count * G) - 1,
            first,
            count,
            start,
            rows,
            bounds,
        )[1]
        most = cl.minimum((last - 1) // TILE_KEYS - base + 1, TILES)
        read = cl.minimum((reader - 1) // TILE_KEYS - base + 1, TILES)
        workspace = cl.scratch((slices, 32), cl.f32)
        rowspace = cl.scratch((32,), cl.f32)
        peak = cl.full((slices, 32), -cl.inf, cl.f32, layout=schedule)
        for step in range(0, most, slices):
            used = within(live, step + part, seen, lower)
            peak = cl.maximum(
                peak,
                cl.load(
                    stats,
                    ((kv * TILES + step + part) * LINES + line) * 2,
                    used,
                    other=-cl.inf,
                ),
            )
        peak = cl.broadcast_to(
            cl.max(peak, 0, scratch=workspace),
            (slices, 32),
            axes=(1,),
            layout=schedule,
            scratch=rowspace,
        )
        total = cl.full((slices, 32), 0.0, cl.f32, layout=schedule)
        for step in range(0, most, slices):
            used = within(live, step + part, seen, lower)
            at = ((kv * TILES + step + part) * LINES + line) * 2
            total = total + cl.load(stats, at + 1, used, other=0.0) * cl.exp2(
                cl.load(stats, at, used, other=-cl.inf) - peak,
            )
        inverse = cl.broadcast_to(
            1.0 / cl.sum(total, 0, scratch=workspace),
            (slices, 32),
            axes=(1,),
            layout=schedule,
            scratch=rowspace,
        )
        for step in range(0, read, slices):
            used = within(live, step + part, seen, lower)
            at = (kv * TILES + step + part) * LINES + line
            cl.store(
                scales,
                at,
                cl.exp2(cl.load(stats, at * 2, used, other=-cl.inf) - peak) * inverse,
                live & (step + part < read),
            )

    return factors


@cache
def make_weighted(
    dtype,
    *,
    head_dim,
    group,
    page_size,
    paged,
    causal,
    window,
    segmented,
    output_mod,
    splits,
):
    D, G = head_dim, group
    (_, GN), (FM, FN) = WEIGHT_GROUPS, WEIGHT_FRAGMENTS
    atom = Atom(dtype)
    tiled = TiledAtom(atom, WEIGHT_GROUPS, WEIGHT_FRAGMENTS)
    BM, BN = tiled.shape
    layout = atom.layout(threads=tiled.threads)

    def operands(block, column_block, count, kv, LINES, WIDTH):
        simd, local, column = place(layout)
        lines = tuple(
            cl.minimum(block * BM + (simd // GN * FM + i) * 8 + local, count * G - 1)
            for i in range(FM)
        )
        return (
            lines,
            tuple((kv * LINES + line) * WIDTH + column for line in lines),
            tuple(
                column_block * BN + (simd % GN * FN + j) * 8 + column for j in range(FN)
            ),
        )

    def value_at(table, key, local, top, kv, HK):
        if not paged:
            return (cl.minimum(key + local, top) * HK + kv) * D
        page = cl.load(
            table[0],
            cl.minimum(key // page_size, top // page_size),
            True,
            other=0,
        )
        return ((page * HK + kv) * page_size + key % page_size + local) * D

    def tile(
        accumulators,
        weights,
        values,
        table,
        scales,
        lines,
        weight_at,
        columns,
        number,
        base,
        top,
        kv,
        HK,
        LINES,
        TILES,
    ):
        _, local, _ = place(layout)
        factors = tuple(
            cl.load(scales, (kv * TILES + number) * LINES + line, True, other=0.0)
            for line in lines
        )
        bases = [
            value_at(table, base + number * TILE_KEYS + 8 * step, local, top, kv, HK)
            for step in range(TILE_KEYS // 8)
        ]
        partial = tiled.zeros()
        for step, at in enumerate(bases):
            offset = number * TILE_KEYS + 8 * step
            partial = tiled.mma(
                tuple(
                    cl.load(weights, where + offset, True, other=0.0)
                    for where in weight_at
                ),
                tuple(
                    cl.load(values, at + column, True, other=0.0) for column in columns
                ),
                partial,
            )
        return tuple(
            tuple(total + part * factor for total, part in zip(totals, parts))
            for totals, parts, factor in zip(accumulators, partial, factors)
        )

    def finish(
        output,
        outs,
        accumulators,
        part,
        block,
        column_block,
        first,
        count,
        kv,
        HQ,
        HK,
        LINES,
    ):
        index = cl.arange(BM * BN, layout=tiled.layout())
        line = block * BM + index // BN
        column = column_block * BN + index % BN
        if splits > 1:
            cl.store(
                output,
                ((part * HK + kv) * LINES + line) * D + column,
                tiled.unpack(accumulators),
                line < count * G,
            )
            return
        row, head = first + line // G, kv * G + line % G
        result = applied(
            output_mod,
            tiled.unpack(accumulators),
            row,
            head,
            column,
            outs,
        )
        cl.store(
            output,
            (row * HQ + head) * D + column,
            cl.cast(result, dtype),
            line < count * G,
        )

    @cl.kernel
    def weighted(
        weights: cl.buffer(dtype),
        scales: cl.buffer(cl.f32),
        values: cl.buffer(dtype),
        table: (cl.buffer(cl.u32),) if paged else (),
        lengths: cl.buffer(cl.u32),
        bounds: (cl.buffer(cl.u32),) if segmented else (),
        output: cl.buffer(cl.f32 if splits > 1 else dtype),
        outs: operand_buffers(None if splits > 1 else output_mod),
        rows: cl.u32,
        first: cl.u32,
        count: cl.u32,
        HQ: cl.constexpr,
        HK: cl.constexpr,
        CACHE: cl.constexpr,
        LINES: cl.constexpr,
        WIDTH: cl.constexpr,
    ):
        TILES = WIDTH // TILE_KEYS
        COLUMNS = D // BN
        BLOCKS = (count * G + BM - 1) // BM
        values = cl.view(values, CACHE)
        column_block = cl.program_id() % COLUMNS
        block = cl.program_id() // COLUMNS % BLOCKS
        kv = cl.program_id() // COLUMNS // BLOCKS % HK
        part = cl.program_id() // COLUMNS // BLOCKS // HK
        start = cl.load(lengths, 0, True, other=0)
        base, top = block_band(
            block,
            BM,
            first,
            count,
            start,
            rows,
            bounds,
            group=G,
            causal=causal,
            window=window,
        )
        seen = cl.minimum((top - base) // TILE_KEYS + 1, TILES)
        share = (seen + splits - 1) // splits
        low = cl.minimum(part * share, seen)
        lines, weight_at, columns = operands(
            block,
            column_block,
            count,
            kv,
            LINES,
            WIDTH,
        )
        accumulators = tiled.zeros()
        for number in range(low, cl.minimum(low + share, seen)):
            accumulators = tile(
                accumulators,
                weights,
                values,
                table,
                scales,
                lines,
                weight_at,
                columns,
                number,
                base,
                top,
                kv,
                HK,
                LINES,
                TILES,
            )
        finish(
            output,
            outs,
            accumulators,
            part,
            block,
            column_block,
            first,
            count,
            kv,
            HQ,
            HK,
            LINES,
        )

    return weighted


@cache
def make_partial_sums(dtype, *, head_dim, group, splits, output_mod):
    D, G = head_dim, group
    schedule = Layout.blocked((SUM_BLOCK,), threads=256)

    @cl.kernel
    def partial_sums(
        partials: cl.buffer(cl.f32),
        output: cl.buffer(dtype),
        outs: operand_buffers(output_mod),
        first: cl.u32,
        count: cl.u32,
        HQ: cl.constexpr,
        HK: cl.constexpr,
        LINES: cl.constexpr,
    ):
        element = cl.program_id() * SUM_BLOCK + cl.arange(SUM_BLOCK, layout=schedule)
        live = element < HK * count * G * D
        kv = element // (count * G * D)
        line = element // D % (count * G)
        column = element % D
        total = cl.full((SUM_BLOCK,), 0.0, cl.f32, layout=schedule)
        for part in range(splits):
            total = total + cl.load(
                partials,
                ((part * HK + kv) * LINES + line) * D + column,
                live,
                other=0.0,
            )
        row, head = first + line // G, kv * G + line % G
        cl.store(
            output,
            (row * HQ + head) * D + column,
            cl.cast(applied(output_mod, total, row, head, column, outs), dtype),
            live,
        )

    return partial_sums


def prefill_steps(device, call, plan):
    height = 8 * plan.simds
    most = (2**32 - 1) // (call.heads * call.head_dim)
    if call.rows > most:
        raise ValueError(
            f"prefill takes at most {most} rows of these heads, for u32 indexing",
        )
    kernel = make_prefill(
        call.dtype,
        **call.options,
        score_mod=call.score_mod,
        output_mod=call.output_mod,
        plan=plan,
    ).compile(
        constants={
            "HQ": call.heads,
            "HK": call.kv_heads,
            "CACHE": call.cache,
            "SCALE": call.factor,
        },
        threads=32 * plan.simds,
        device=device,
        max_grid=call.kv_heads * -(-most * call.group // height),
    )
    grid = call.kv_heads * -(-call.rows * call.group // height)
    return [
        kernel.dispatch(
            device,
            [
                call.q,
                call.keys,
                call.values,
                *call.tables,
                call.lengths,
                *call.cuts,
                call.output,
                *call.score_operands,
                *call.output_operands,
            ],
            grid=(grid, 1, 1),
            scalars={"rows": call.rows},
        ),
    ]


def pass_rows(heads, width, dtype):
    return max(
        16,
        min(PASS_ROWS, PASS_BYTES // (heads * width * dtype.itemsize)) // 16 * 16,
    )


def band(reach, window, causal, group, segmented):
    if window:
        before = window - 1 if causal else window
        before = min(reach - 1, before) if segmented else before
        after = 0 if causal else min(reach - 1, window)
    else:
        before, after = reach - 1, 0 if causal else reach - 1
    block_rows = -(-TILE_GROUPS[0] * TILE_FRAGMENTS[0] * 8 // group) + 1
    return -(-(block_rows + before + after + TILE_KEYS - 1) // TILE_KEYS) * TILE_KEYS


def long_head_steps(device, call, plan):
    group, heads, kv_heads, head_dim, dtype = (
        call.group,
        call.heads,
        call.kv_heads,
        call.head_dim,
        call.dtype,
    )
    most = (2**32 - 1) // (heads * head_dim)
    if call.rows > most:
        raise ValueError(
            f"prefill takes at most {most} rows of these heads, for u32 indexing",
        )
    width = -(-call.context // TILE_KEYS) * TILE_KEYS
    if call.segmented or call.window:
        width = min(
            width,
            band(call.longest, call.window, call.causal, group, call.segmented),
        )
    count = pass_rows(heads, width, dtype)
    lines, tiles = count * group, width // TILE_KEYS
    sizes = [
        lines * kv_heads * width * dtype.itemsize,
        lines * kv_heads * tiles * 8,
        lines * kv_heads * tiles * 4,
    ]
    workspace = shared_workspace(device, sum(sizes))
    weights, stats, scales = (
        workspace.view(sum(sizes[:index]), size) for index, size in enumerate(sizes)
    )
    constants = {
        "HQ": heads,
        "HK": kv_heads,
        "CACHE": call.cache,
        "LINES": lines,
        "WIDTH": width,
    }
    mask = {"causal": call.causal, "window": call.window, "segmented": call.segmented}
    tall = TILE_GROUPS[0] * TILE_FRAGMENTS[0] * 8
    scores = make_scores(dtype, **call.options, score_mod=call.score_mod).compile(
        constants={**constants, "SCALE": call.factor},
        threads=32 * TILE_GROUPS[0] * TILE_GROUPS[1],
        device=device,
        max_grid=kv_heads * -(-lines // tall) * tiles,
    )
    factors = make_factors(group=group, **mask).compile(
        constants={"LINES": lines, "TILES": tiles},
        threads=256,
        device=device,
        max_grid=kv_heads * -(-lines // 32),
    )
    columns = head_dim // (WEIGHT_GROUPS[1] * WEIGHT_FRAGMENTS[1] * 8)
    height = WEIGHT_GROUPS[0] * WEIGHT_FRAGMENTS[0] * 8
    base = kv_heads * -(-lines // height) * columns
    splits = min(WEIGHT_SPLITS, tiles, -(-WEIGHT_TARGET // base))
    weighted = make_weighted(
        dtype,
        **call.options,
        output_mod=None if splits > 1 else call.output_mod,
        splits=splits,
    ).compile(
        constants=constants,
        threads=32 * WEIGHT_GROUPS[0] * WEIGHT_GROUPS[1],
        device=device,
        max_grid=base * splits,
    )
    if splits > 1:
        partials = device.buffer(splits * kv_heads * lines * head_dim * 4)
        summed = make_partial_sums(
            dtype,
            head_dim=head_dim,
            group=group,
            splits=splits,
            output_mod=call.output_mod,
        ).compile(
            constants={"HQ": heads, "HK": kv_heads, "LINES": lines},
            threads=256,
            device=device,
            max_grid=-(-kv_heads * lines * head_dim // SUM_BLOCK),
        )
    steps = []
    for first in range(0, call.rows, count):
        scalars = {
            "rows": call.rows,
            "first": first,
            "count": min(count, call.rows - first),
        }
        used = scalars["count"] * group
        outputs = [partials] if splits > 1 else [call.output]
        steps += [
            scores.dispatch(
                device,
                [
                    call.q,
                    call.keys,
                    *call.tables,
                    call.lengths,
                    *call.cuts,
                    weights,
                    stats,
                    *call.score_operands,
                ],
                grid=(kv_heads * -(-used // tall) * tiles, 1, 1),
                scalars=scalars,
            ),
            factors.dispatch(
                device,
                [stats, call.lengths, scales, *call.cuts],
                grid=(kv_heads * -(-used // 32), 1, 1),
                scalars=scalars,
            ),
            weighted.dispatch(
                device,
                [
                    weights,
                    scales,
                    call.values,
                    *call.tables,
                    call.lengths,
                    *call.cuts,
                    *outputs,
                    *([] if splits > 1 else call.output_operands),
                ],
                grid=(kv_heads * -(-used // height) * columns * splits, 1, 1),
                scalars=scalars,
            ),
        ]
        if splits > 1:
            steps.append(
                summed.dispatch(
                    device,
                    [partials, call.output, *call.output_operands],
                    grid=(-(-kv_heads * used * head_dim // SUM_BLOCK), 1, 1),
                    scalars={"first": first, "count": scalars["count"]},
                ),
            )
    return steps
