from dataclasses import replace

import numpy as np

from carbon import language as cl
from carbon.kernels import search
from carbon.kernels.attention.common import KEYS, LOG2E, Call
from carbon.kernels.attention.decode import Decode, Verify, decode_steps
from carbon.kernels.attention.prefill import (
    LongHead,
    Prefill,
    long_head_steps,
    prefill_steps,
)
from carbon.kernels.ops.indexing import segment_rows

FEW_ROWS, VERIFY_ROWS, VERIFY_GROUPS, WIDE_GROUPS, FEW_SPLITS = 4, 16, 64, 96, 4
DECODE_SPLITS, ROW_GROUPS, WINDOW_KEYS, FEW_GROUPS, FEW_HEADS = 32, 40, 128, 10, 32
STEPS = {
    Prefill: prefill_steps,
    LongHead: long_head_steps,
    Decode: decode_steps,
    Verify: decode_steps,
}


def launch(
    device,
    q,
    keys,
    values,
    output,
    *,
    rows,
    heads,
    kv_heads,
    head_dim,
    table=None,
    lengths=None,
    page_size=16,
    positions=None,
    max_pages=0,
    context=None,
    causal=True,
    window=0,
    segments=None,
    sequences=0,
    longest=None,
    scale=None,
    score_mod=None,
    score_operands=(),
    output_mod=None,
    output_operands=(),
    dtype=cl.bf16,
    plan=None,
):
    """Return the dispatches for output = softmax(scale · q kᵀ) v, with q and output
    (rows, heads, head_dim).

    Args:
        keys, values: pages of (kv_heads, page_size, head_dim) that `table` names, or
            without `table` one (positions, kv_heads, head_dim) array.
        table, lengths: the page table and u32 positions; row r is at lengths[0] + r, or
            with `max_pages` one token of its own sequence at lengths[r], its pages in
            row r of a (rows, max_pages) table. With `max_pages` and `segments`, the
            rows of sequence s are packed: row r of it is at lengths[s] + r, its pages
            in row s of a (sequences, max_pages) table, as a batched verify step
            needs.
        positions: without `table`, the positions the keys hold; the rows are the last
            `rows` of them.
        causal: whether a row sees only the positions up to its own.
        window: when nonzero, the positions within `window` of a row's own: before it,
            or on both sides.
        segments, sequences, longest: `sequences` + 1 ascending row offsets of a packed
            batch, and the rows of its longest sequence (at most 16 with
            `max_pages`); a row sees only its own sequence.
        context: a bound on lengths[0] + rows.
        score_mod, score_operands, output_mod, output_operands: a ScoreMod and an
            OutputMod with their buffers.
        plan: a Prefill, LongHead, Decode or Verify in place of the runtime search's
            choice.

    Returns:
        The dispatches to prepare and submit, in order.
    """
    capacity = keys.size // (kv_heads * head_dim * dtype.itemsize)
    if table is None:
        if positions is None or positions > capacity or causal and rows > positions:
            raise ValueError(
                "attention over one array needs positions within it, and "
                "rows ≤ positions if causal",
            )
        lengths = device.buffer(4)
        lengths.write(np.uint32((positions - rows) % 2**32).tobytes())
        page_size, context = 0, capacity
    call = Call(
        q,
        keys,
        values,
        table,
        lengths,
        output,
        rows,
        heads,
        kv_heads,
        head_dim,
        page_size,
        max_pages,
        capacity if context is None else context,
        causal,
        window,
        segments,
        sequences,
        rows if longest is None else longest,
        score_mod,
        tuple(score_operands),
        output_mod,
        tuple(output_operands),
        (head_dim**-0.5 if scale is None else scale) * LOG2E,
        dtype,
    )
    check(call)
    if plan is None:
        plan = search.choose(
            device,
            key(call),
            candidates(call),
            lambda sample, candidate: dispatches(device, sample, candidate),
            lambda: trial(device, call),
        )
    return dispatches(device, call, plan)


def paged_attention(device, q, keys, values, table, lengths, output, **options):
    """`launch` over keys and values in pages."""
    return launch(
        device,
        q,
        keys,
        values,
        output,
        table=table,
        lengths=lengths,
        **options,
    )


def attention(device, q, keys, values, output, **options):
    """`launch` over keys and values in one array."""
    return launch(device, q, keys, values, output, **options)


def dispatches(device, call, plan):
    if call.packed:
        return STEPS[type(plan)](device, replace(call, bounds=call.segments), plan)
    if not call.segmented:
        return STEPS[type(plan)](device, call, plan)
    bounds = device.buffer(call.rows * 8)
    head = segment_rows(
        device,
        call.segments,
        bounds,
        rows=call.rows,
        sequences=call.sequences,
    )
    return [head, *STEPS[type(plan)](device, replace(call, bounds=bounds), plan)]


def candidates(call):
    if call.max_pages:
        return verify_plans(call) if call.packed else decode_plans(call)
    prefill = prefill_plans(call)
    few = decode_plans(call) + prefill if call.rows <= FEW_ROWS else prefill
    return verify_plans(call) + few


def verify_plans(call):
    rows = call.longest if call.packed else call.rows
    if (
        not (1 if call.packed else 2) <= rows <= VERIFY_ROWS
        or call.segmented != call.packed
        or call.head_dim % 128
    ):
        return []
    tiles = -(-rows * call.group // 8)
    if call.head_dim > 128:
        group, other, least = 1, 2, FEW_SPLITS
        groups = min(WIDE_GROUPS // tiles, VERIFY_GROUPS)
    else:
        group, other, least = 8, 2, 1
        groups = VERIFY_GROUPS * 3 // 4 if tiles > 2 else VERIFY_GROUPS
    heads = call.kv_heads * (call.sequences if call.packed else 1)
    splits = max(least, min(DECODE_SPLITS, groups // heads))
    plans = [
        Verify(count, group)
        for count in (splits, 2 * splits, splits // 2, splits // 3)
        if least <= count <= 2 * DECODE_SPLITS
    ]
    return list(dict.fromkeys([*plans, Verify(splits, other)]))


def prefill_plans(call):
    if call.head_dim > 128:
        return [LongHead()]
    fused = Prefill(
        simds=8 if call.head_dim > 64 else 4,
        block_keys=2 * KEYS if call.head_dim <= 32 else KEYS,
        edges=call.head_dim <= 32
        or call.head_dim == 128
        and (call.window > 0 or call.segmented),
    )
    plans = [fused, replace(fused, edges=not fused.edges)]
    if call.head_dim < 128:
        return plans
    if call.rows >= 32 and not (call.causal and call.window):
        return [LongHead(), *plans]
    return [*plans, LongHead()]


def decode_plans(call):
    tiles = call.group % 8 == 0 and call.head_dim >= 512
    splits = decode_splits(call, tiles)
    plans = [Decode(splits, tiles)]
    if call.window or call.rows == 1:
        plans += [
            Decode(count, tiles)
            for count in (splits // 2, splits * 2)
            if 1 <= count <= 2 * DECODE_SPLITS
        ]
    if call.group % 8 == 0 and call.head_dim % 128 == 0 and call.head_dim >= 256:
        plans.append(Decode(decode_splits(call, not tiles), not tiles))
    return plans


def decode_splits(call, tiles):
    if call.window:
        splits = max(1, min(DECODE_SPLITS, call.window // WINDOW_KEYS))
        groups = call.rows * call.kv_heads * splits
        if groups <= FEW_GROUPS and groups * call.group <= FEW_HEADS:
            splits = max(1, min(DECODE_SPLITS, 2 * call.window // WINDOW_KEYS))
        return splits
    if call.rows == 1:
        return max(
            1,
            min(
                DECODE_SPLITS,
                ROW_GROUPS // (call.kv_heads * (call.group // 8 if tiles else 1)),
            ),
        )
    return DECODE_SPLITS


def key(call):
    rows = (
        call.rows
        if call.max_pages or call.rows <= 2 * FEW_ROWS
        else 1 << (call.rows - 1).bit_length()
    )
    hooks = tuple(search.stable(hook) for hook in (call.score_mod, call.output_mod))
    return (
        "attention",
        call.paged,
        call.dtype.name,
        call.heads,
        call.kv_heads,
        call.head_dim,
        call.page_size,
        bool(call.max_pages),
        rows,
        1 << (call.context - 1).bit_length(),
        call.causal,
        call.window,
        call.segmented,
        (call.sequences, call.longest) if call.packed else (),
        hooks,
    )


def trial(device, call):
    def upload(array):
        buffer = device.buffer(array.nbytes)
        buffer.write(array.tobytes())
        return buffer

    changes = {"output": device.buffer(call.output.size)}
    if call.segmented:
        changes["segments"] = upload(
            np.linspace(0, call.rows, call.sequences + 1).astype(np.uint32),
        )
    if call.paged:
        pages = call.keys.size // (
            call.kv_heads * call.page_size * call.head_dim * call.dtype.itemsize
        )
        if call.max_pages:
            length = min(call.context, call.max_pages * call.page_size)
            count = call.sequences if call.packed else call.rows
            changes["table"] = upload(
                (np.arange(count * call.max_pages) % pages).astype(np.uint32),
            )
            last = call.longest if call.packed else 1
            changes["lengths"] = upload(
                np.full(count, max(0, length - last), np.uint32),
            )
        else:
            length = min(call.context, pages * call.page_size)
            changes["table"] = upload(
                (np.arange(-(-length // call.page_size)) % pages).astype(np.uint32),
            )
            changes["lengths"] = upload(
                np.array([max(0, length - call.rows)], np.uint32),
            )
    return replace(call, **changes)


def check(call):
    if (
        call.heads % call.kv_heads
        or not 1 <= call.group <= 16
        or call.head_dim % 32
        or not 32 <= call.head_dim <= 512
        or call.head_dim > 128
        and call.head_dim % 128
        or call.paged
        and call.page_size % KEYS
        or call.rows < 1
    ):
        raise ValueError(
            "attention needs kv_heads dividing heads with at most 16 query heads "
            "each, head_dim a multiple of 32 up to 512 and of 128 above 128, "
            "page_size a multiple of 16 and a row",
        )
    if call.max_pages and not call.causal:
        raise ValueError("non-causal attention is a prefill, without max_pages")
    if call.packed and (call.head_dim % 128 or not 1 <= call.longest <= VERIFY_ROWS):
        raise ValueError(
            "sequences packed with max_pages take head_dim a multiple of 128 and at "
            f"most {VERIFY_ROWS} rows each",
        )
    for hook, buffers in (
        (call.score_mod, call.score_operands),
        (call.output_mod, call.output_operands),
    ):
        if len(buffers) != (0 if hook is None else len(hook.operands)):
            raise ValueError(
                "score_mod and output_mod take one buffer per operand type",
            )
    if call.segmented and call.sequences < 1:
        raise ValueError("segments need at least one sequence")
