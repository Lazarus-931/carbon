from collections import namedtuple
from functools import cache

import numpy as np

from carbon import language as cl
from carbon.compiler import Layout

THREADS = 1024
RUN = 4
WIDTH = THREADS * RUN
SETTINGS = 6
ATTEMPTS = 4
LOG2E = 1.4426950408889634
SIGNIFICANT = {"bf16": 16, "f16": 19, "f32": 32}
SEQUENCES = 256
ACCEPT_DRAW, RESIDUAL_DRAW = 0, 1
Vocabulary = namedtuple(
    "Vocabulary",
    "slots logit weight weighed draw lowest nucleus",
)


def settings(seeds, temperatures, top_k=None, top_p=None, min_p=None, penalties=None):
    """The u32 (rows, 6) settings `sample` reads per row: seed, top-k (0 keeps every
    token), and the f32 bits of the temperature (0 is greedy), top-p (1 keeps every
    token), min-p (0 keeps every token) and the repetition penalty (1 is none)."""
    rows = len(seeds)
    table = np.zeros((rows, SETTINGS), np.uint32)
    table[:, 0] = list(seeds)
    table[:, 1] = 0 if top_k is None else top_k
    for column, values, default in (
        (2, temperatures, 0.0),
        (3, top_p, 1.0),
        (4, min_p, 0.0),
        (5, penalties, 1.0),
    ):
        table[:, column] = np.asarray(
            np.full(rows, default) if values is None else values,
            np.float32,
        ).view(np.uint32)
    return table


def seen_bits(tokens, vocab):
    bits = np.zeros((len(tokens), -(-vocab // 32)), np.uint32)
    for row, ids in enumerate(tokens):
        for token in {int(token) for token in ids}:
            bits[row, token // 32] |= np.uint32(1 << (token % 32))
    return bits


def _key(value):
    bits = cl.bitcast(value, cl.u32)
    return cl.select(bits >> 31 == 1, bits ^ 0xFFFFFFFF, bits ^ 0x80000000)


def _value(key):
    return cl.bitcast(
        cl.select(key >> 31 == 1, key ^ 0x80000000, key ^ 0xFFFFFFFF),
        cl.f32,
    )


def _mix(h):
    h = (h ^ (h >> 16)) * 0x7FEB352D
    h = (h ^ (h >> 15)) * 0x846CA68B
    return h ^ (h >> 16)


def _uniform(seed, position, counter):
    h = _mix(_mix(_mix(seed) ^ position) ^ counter)
    return (cl.cast(h >> 8, cl.f32) + 0.5) * (1.0 / 16777216.0)


@cache
def _vocabulary(dtype):
    slots = Layout.blocked((WIDTH,), threads=THREADS)
    low = 32 - SIGNIFICANT[dtype.name]

    def logit(logits, base, index, V):
        return cl.cast(cl.load(logits, base + index, index < V, other=-cl.inf), cl.f32)

    def weight(value, index, V, bound, peak, scale):
        return cl.select(
            (index < V) & (_key(value) >= bound),
            cl.exp2((value - peak) * scale),
            0.0,
        )

    @cl.inline
    def counted(logits, base, slot, V, bound, counts):
        count = cl.full((WIDTH,), 0, cl.u32, layout=slots)
        for start in range(0, V, WIDTH):
            index = start + slot
            count = count + cl.select(
                (index < V) & (_key(logit(logits, base, index, V)) >= bound),
                1,
                0,
            )
        return cl.item(cl.sum(count, 0, scratch=counts), scratch=counts)

    @cl.inline
    def weighed(logits, base, slot, V, bound, peak, scale, floats):
        total = cl.full((WIDTH,), 0.0, cl.f32, layout=slots)
        for start in range(0, V, WIDTH):
            index = start + slot
            total = total + weight(
                logit(logits, base, index, V),
                index,
                V,
                bound,
                peak,
                scale,
            )
        return cl.item(cl.sum(total, 0, scratch=floats), scratch=floats)

    @cl.inline
    def draw(
        logits,
        base,
        slot,
        V,
        lowest,
        peak,
        scale,
        seed,
        position,
        attempt,
        floats,
        counts,
    ):
        totals = cl.full((WIDTH,), 0.0, cl.f32, layout=slots)
        for start in range(0, V, WIDTH):
            index = start + slot
            totals = totals + weight(
                logit(logits, base, index, V),
                index,
                V,
                lowest,
                peak,
                scale,
            )
        reached = cl.cumsum(totals, 0, scratch=floats)
        mass = cl.item(cl.max(reached, 0, scratch=floats), scratch=floats)
        target = _uniform(seed, position, V + attempt) * mass
        first = cl.item(
            cl.max(
                cl.select((reached > target) & (totals > 0.0), WIDTH - slot, 0),
                0,
                scratch=counts,
            ),
            scratch=counts,
        )
        last = cl.item(
            cl.max(cl.select(totals > 0.0, slot + 1, 0), 0, scratch=counts),
            scratch=counts,
        )
        owner = cl.select(first > 0, WIDTH - first, last - 1)
        rest = target - (reached - totals)
        running = cl.full((WIDTH,), 0.0, cl.f32, layout=slots)
        found = cl.full((WIDTH,), V, cl.u32, layout=slots)
        picked = cl.full((WIDTH,), -cl.inf, cl.f32, layout=slots)
        final = cl.full((WIDTH,), 0, cl.u32, layout=slots)
        tail = cl.full((WIDTH,), -cl.inf, cl.f32, layout=slots)
        for start in range(0, V, WIDTH):
            index = start + slot
            value = logit(logits, base, index, V)
            share = weight(value, index, V, lowest, peak, scale)
            running = running + share
            held = cl.bitcast(share, cl.u32) != 0
            crossing = (found == V) & held & (_key(running) > _key(rest))
            found = cl.select(crossing, index, found)
            picked = cl.select(crossing, value, picked)
            final = cl.select(held, index, final)
            tail = cl.select(held, value, tail)
        mine = slot == owner
        token = cl.item(
            cl.max(
                cl.select(mine, cl.select(found < V, found, final), 0),
                0,
                scratch=counts,
            ),
            scratch=counts,
        )
        value = cl.item(
            cl.max(
                cl.select(mine, cl.select(found < V, picked, tail), -cl.inf),
                0,
                scratch=floats,
            ),
            scratch=floats,
        )
        return token, value, mass

    @cl.inline
    def lowest(
        logits,
        base,
        slot,
        V,
        best,
        top,
        peak,
        top_k,
        min_p,
        temperature,
        counts,
    ):
        floor = 0
        if (top_k > 0) & (top_k < V):
            lower = ~cl.item(cl.max(~best, 0, scratch=counts), scratch=counts)
            lo, hi = cl.select(top_k <= WIDTH, lower, 0) >> low, (top >> low) + 1
            for step in range(32 - low):
                if hi - lo > 1:
                    middle = lo + (hi - lo) // 2
                    enough = (
                        counted(logits, base, slot, V, middle << low, counts) >= top_k
                    )
                    lo = cl.select(enough, middle, lo)
                    hi = cl.select(enough, hi, middle)
            floor = lo << low
        return cl.select(
            min_p > 0.0,
            cl.maximum(floor, _key(peak + cl.log(min_p) * temperature)),
            floor,
        )

    @cl.inline
    def nucleus(logits, base, slot, V, floor, top, peak, scale, target, floats):
        lo, hi = floor >> low, (top >> low) + 1
        for step in range(32 - low):
            if hi - lo > 1:
                middle = lo + (hi - lo) // 2
                bound = cl.maximum(middle << low, floor)
                enough = (
                    weighed(logits, base, slot, V, bound, peak, scale, floats) >= target
                )
                lo = cl.select(enough, middle, lo)
                hi = cl.select(enough, hi, middle)
        return cl.maximum(lo << low, floor)

    return Vocabulary(slots, logit, weight, weighed, draw, lowest, nucleus)


@cache
def make_sample(dtype=cl.bf16, *, penalty=False):
    vocabulary = _vocabulary(dtype)
    slots, logit, weighed = vocabulary.slots, vocabulary.logit, vocabulary.weighed
    draw, lowest, nucleus = vocabulary.draw, vocabulary.lowest, vocabulary.nucleus

    def penalized(logits, seen, base, words, index, V, factor):
        value = logit(logits, base, index, V)
        if not penalty:
            return value
        marked = (index < V) & (
            (cl.load(seen[0], words + index // 32, index < V, other=0) >> (index % 32))
            & 1
            == 1
        )
        stored = cl.cast(
            cl.select(
                cl.bitcast(value, cl.u32) >> 31 == 1,
                value * factor,
                value / factor,
            ),
            dtype,
        )
        cl.store(logits, base + index, stored, marked)
        return cl.select(marked, cl.cast(stored, cl.f32), value)

    def mark(seen, words, token):
        if penalty:
            word = cl.full((), 0, cl.u32) + words + token // 32
            cl.store(
                seen[0],
                word,
                cl.load(seen[0], word, True, other=0) | (1 << token % 32),
                True,
            )

    @cl.kernel
    def sample(
        logits: cl.buffer(dtype),
        table: cl.buffer(cl.u32),
        lengths: cl.buffer(cl.u32),
        tokens: cl.buffer(cl.u32),
        seen: (cl.buffer(cl.u32),) if penalty else (),
        V: cl.constexpr,
    ):
        row = cl.program_id()
        base, words = row * V, row * ((V + 31) // 32)
        slot = cl.arange(WIDTH, layout=slots)
        seed = cl.load(table, row * SETTINGS, True, other=0)
        top_k = cl.load(table, row * SETTINGS + 1, True, other=0)
        temperature = cl.bitcast(
            cl.load(table, row * SETTINGS + 2, True, other=0),
            cl.f32,
        )
        top_p = cl.bitcast(cl.load(table, row * SETTINGS + 3, True, other=0), cl.f32)
        min_p = cl.bitcast(cl.load(table, row * SETTINGS + 4, True, other=0), cl.f32)
        factor = cl.bitcast(cl.load(table, row * SETTINGS + 5, True, other=0), cl.f32)
        position = cl.load(lengths, row, True, other=0)
        floats, counts = cl.scratch((WIDTH,), cl.f32), cl.scratch((WIDTH,), cl.u32)
        best = cl.full((WIDTH,), 0, cl.u32, layout=slots)
        where = cl.full((WIDTH,), 0, cl.u32, layout=slots)
        for start in range(0, V, WIDTH):
            index = start + slot
            key = cl.select(
                index < V,
                _key(penalized(logits, seen, base, words, index, V, factor)),
                0,
            )
            better = key > best
            best = cl.select(better, key, best)
            where = cl.select(better, index, where)
        top = cl.item(cl.max(best, 0, scratch=counts), scratch=counts)
        token = V - cl.item(
            cl.max(cl.select(best == top, V - where, 0), 0, scratch=counts),
            scratch=counts,
        )
        peak = _value(top)
        if temperature > 0.0:
            scale = LOG2E / temperature
            floor = lowest(
                logits,
                base,
                slot,
                V,
                best,
                top,
                peak,
                top_k,
                min_p,
                temperature,
                counts,
            )
            token, value, mass = draw(
                logits,
                base,
                slot,
                V,
                floor,
                peak,
                scale,
                seed,
                position,
                0,
                floats,
                counts,
            )
            if top_p < 1.0:
                target = mass * top_p
                settled = (
                    weighed(logits, base, slot, V, _key(value) + 1, peak, scale, floats)
                    < target
                )
                for attempt in range(1, ATTEMPTS):
                    if ~settled:
                        token, value, mass = draw(
                            logits,
                            base,
                            slot,
                            V,
                            _key(value) + 1,
                            peak,
                            scale,
                            seed,
                            position,
                            attempt,
                            floats,
                            counts,
                        )
                        settled = (
                            weighed(
                                logits,
                                base,
                                slot,
                                V,
                                _key(value) + 1,
                                peak,
                                scale,
                                floats,
                            )
                            < target
                        )
                if ~settled:
                    token, value, mass = draw(
                        logits,
                        base,
                        slot,
                        V,
                        nucleus(
                            logits,
                            base,
                            slot,
                            V,
                            floor,
                            top,
                            peak,
                            scale,
                            target,
                            floats,
                        ),
                        peak,
                        scale,
                        seed,
                        position,
                        ATTEMPTS,
                        floats,
                        counts,
                    )
        cl.store(tokens, cl.full((), 0, cl.u32) + row, token, True)
        mark(seen, words, token)

    return sample


def sample(
    device,
    logits,
    table,
    lengths,
    tokens,
    *,
    rows,
    vocab,
    seen=None,
    dtype=cl.bf16,
):
    """Dispatch: tokens[r] (u32) = a token of row r of (rows, vocab) logits drawn under
    row r's `settings` at position lengths[r]. With `seen`, a `seen_bits` bitmap, the
    row's repetition penalty applies in place to the logits of the tokens it marks, and
    the chosen token is marked."""
    artifact = make_sample(dtype, penalty=seen is not None).compile(
        constants={"V": vocab},
        threads=THREADS,
        device=device,
        max_grid=rows,
    )
    bound = [] if seen is None else [seen]
    return artifact.dispatch(
        device,
        [logits, table, lengths, tokens, *bound],
        grid=(rows, 1, 1),
    )


@cache
def make_accept(*, history=False, last=False):
    layout = Layout.blocked((SEQUENCES,), threads=SEQUENCES)

    @cl.kernel
    def accept(
        inputs: cl.buffer(cl.u32),
        tokens: cl.buffer(cl.u32),
        lengths: cl.buffer(cl.u32),
        accepted: cl.buffer(cl.u32),
        nexts: cl.buffer(cl.u32),
        kept: (cl.buffer(cl.u32),) if history else (),
        lasts: (cl.buffer(cl.u32),) if last else (),
        sequences: cl.u32,
        rows: cl.u32,
        capacity: cl.u32,
        offset: cl.u32,
    ):
        sequence = cl.program_id() * SEQUENCES + cl.arange(SEQUENCES, layout=layout)
        live = sequence < sequences
        first = sequence * rows
        run = cl.full((SEQUENCES,), 1, cl.u32, layout=layout)
        count = cl.full((SEQUENCES,), 0, cl.u32, layout=layout)
        for row in range(1, rows):
            draft = cl.load(inputs, first + row, live, other=0)
            choice = cl.load(tokens, first + row - 1, live, other=0)
            run = run & cl.select(draft == choice, 1, 0)
            count = count + run
        length = cl.load(lengths, sequence, live, other=0)
        cl.store(accepted, sequence, count + 1, live)
        cl.store(lengths, sequence, length + count + 1, live)
        cl.store(nexts, sequence, cl.load(tokens, first + count, live, other=0), live)
        if history:
            for row in range(rows):
                cl.store(
                    kept[0],
                    sequence * capacity + length + row,
                    cl.load(tokens, first + row, live, other=0),
                    live & (row <= count) & (length + row < capacity),
                )
        if last:
            cl.store(lasts[0], sequence, offset + first + count, live)

    return accept


@cache
def make_distribution(dtype=cl.bf16):
    vocabulary = _vocabulary(dtype)
    slots, logit, weighed = vocabulary.slots, vocabulary.logit, vocabulary.weighed
    lowest, nucleus = vocabulary.lowest, vocabulary.nucleus

    @cl.kernel
    def distribution(
        logits: cl.buffer(dtype),
        table: cl.buffer(cl.u32),
        stats: cl.buffer(cl.u32),
        per: cl.u32,
        stride: cl.u32,
        V: cl.constexpr,
    ):
        row = cl.program_id()
        base = row * V
        setting = (row // per * stride + row % per) * SETTINGS
        slot = cl.arange(WIDTH, layout=slots)
        top_k = cl.load(table, setting + 1, True, other=0)
        temperature = cl.bitcast(cl.load(table, setting + 2, True, other=0), cl.f32)
        top_p = cl.bitcast(cl.load(table, setting + 3, True, other=0), cl.f32)
        min_p = cl.bitcast(cl.load(table, setting + 4, True, other=0), cl.f32)
        floats, counts = cl.scratch((WIDTH,), cl.f32), cl.scratch((WIDTH,), cl.u32)
        best = cl.full((WIDTH,), 0, cl.u32, layout=slots)
        for start in range(0, V, WIDTH):
            index = start + slot
            key = cl.select(index < V, _key(logit(logits, base, index, V)), 0)
            best = cl.maximum(best, key)
        top = cl.item(cl.max(best, 0, scratch=counts), scratch=counts)
        peak = _value(top)
        floor, mass = top, 0.0
        if temperature > 0.0:
            scale = LOG2E / temperature
            floor = lowest(
                logits,
                base,
                slot,
                V,
                best,
                top,
                peak,
                top_k,
                min_p,
                temperature,
                counts,
            )
            mass = weighed(logits, base, slot, V, floor, peak, scale, floats)
            if top_p < 1.0:
                target = mass * top_p
                floor = nucleus(
                    logits,
                    base,
                    slot,
                    V,
                    floor,
                    top,
                    peak,
                    scale,
                    target,
                    floats,
                )
                mass = weighed(logits, base, slot, V, floor, peak, scale, floats)
        entry = cl.full((), 0, cl.u32) + row * 3
        cl.store(stats, entry, floor, True)
        cl.store(stats, entry + 1, cl.bitcast(peak, cl.u32), True)
        cl.store(stats, entry + 2, cl.bitcast(mass, cl.u32), True)

    return distribution


@cache
def make_accept_sampled(dtype=cl.bf16, *, drafted=False, history=False, last=False):
    vocabulary = _vocabulary(dtype)
    slots, logit, weight = vocabulary.slots, vocabulary.logit, vocabulary.weight

    def probability(logits, stats, row, index, V, scale):
        floor = cl.load(stats, row * 3, True, other=0)
        peak = cl.bitcast(cl.load(stats, row * 3 + 1, True, other=0), cl.f32)
        mass = cl.bitcast(cl.load(stats, row * 3 + 2, True, other=0), cl.f32)
        value = logit(logits, row * V, index, V)
        return weight(value, index, V, floor, peak, scale) / mass

    def guessed(drafts, guesses, draft, index, V, scale):
        if drafted:
            return probability(drafts[0], guesses[0], draft, index, V, scale)
        return 1.0

    def excess(logits, stats, drafts, guesses, row, draft, token, index, V, scale):
        p = probability(logits, stats, row, index, V, scale)
        if drafted:
            return cl.maximum(p - guessed(drafts, guesses, draft, index, V, scale), 0.0)
        return cl.select(index == token, 0.0, p)

    @cl.inline
    def redraw(
        logits,
        stats,
        drafts,
        guesses,
        row,
        draft,
        token,
        V,
        slot,
        scale,
        seed,
        position,
        floats,
        counts,
    ):
        totals = cl.full((WIDTH,), 0.0, cl.f32, layout=slots)
        for start in range(0, V, WIDTH):
            index = start + slot
            totals = totals + excess(
                logits,
                stats,
                drafts,
                guesses,
                row,
                draft,
                token,
                index,
                V,
                scale,
            )
        reached = cl.cumsum(totals, 0, scratch=floats)
        mass = cl.item(cl.max(reached, 0, scratch=floats), scratch=floats)
        target = _uniform(seed, position, RESIDUAL_DRAW) * mass
        first = cl.item(
            cl.max(
                cl.select((reached > target) & (totals > 0.0), WIDTH - slot, 0),
                0,
                scratch=counts,
            ),
            scratch=counts,
        )
        past = cl.item(
            cl.max(cl.select(totals > 0.0, slot + 1, 0), 0, scratch=counts),
            scratch=counts,
        )
        owner = cl.select(first > 0, WIDTH - first, past - 1)
        rest = target - (reached - totals)
        running = cl.full((WIDTH,), 0.0, cl.f32, layout=slots)
        found = cl.full((WIDTH,), V, cl.u32, layout=slots)
        final = cl.full((WIDTH,), 0, cl.u32, layout=slots)
        for start in range(0, V, WIDTH):
            index = start + slot
            part = excess(
                logits,
                stats,
                drafts,
                guesses,
                row,
                draft,
                token,
                index,
                V,
                scale,
            )
            running = running + part
            held = cl.bitcast(part, cl.u32) != 0
            crossing = (found == V) & held & (_key(running) > _key(rest))
            found = cl.select(crossing, index, found)
            final = cl.select(held, index, final)
        chosen = cl.select(slot == owner, cl.select(found < V, found, final), 0)
        return cl.item(cl.max(chosen, 0, scratch=counts), scratch=counts)

    @cl.kernel
    def accept_sampled(
        inputs: cl.buffer(cl.u32),
        tokens: cl.buffer(cl.u32),
        lengths: cl.buffer(cl.u32),
        accepted: cl.buffer(cl.u32),
        nexts: cl.buffer(cl.u32),
        kept: (cl.buffer(cl.u32),) if history else (),
        lasts: (cl.buffer(cl.u32),) if last else (),
        logits: cl.buffer(dtype),
        table: cl.buffer(cl.u32),
        stats: cl.buffer(cl.u32),
        drafts: (cl.buffer(dtype),) if drafted else (),
        guesses: (cl.buffer(cl.u32),) if drafted else (),
        rows: cl.u32,
        capacity: cl.u32,
        offset: cl.u32,
        V: cl.constexpr,
    ):
        sequence = cl.program_id()
        first = sequence * rows
        slot = cl.arange(WIDTH, layout=slots)
        floats, counts = cl.scratch((WIDTH,), cl.f32), cl.scratch((WIDTH,), cl.u32)
        entry = cl.full((), 0, cl.u32) + sequence
        length = cl.item(cl.load(lengths, entry, True, other=0), scratch=counts)
        count, going = 0, 1
        for row in range(1, rows):
            if going == 1:
                target = first + row - 1
                seed = cl.load(table, target * SETTINGS, True, other=0)
                temperature = cl.bitcast(
                    cl.load(table, target * SETTINGS + 2, True, other=0),
                    cl.f32,
                )
                draft = cl.load(inputs, first + row, True, other=0)
                stays = draft == cl.load(tokens, target, True, other=0)
                if temperature > 0.0:
                    scale = LOG2E / temperature
                    p = probability(logits, stats, target, draft, V, scale)
                    q = guessed(
                        drafts,
                        guesses,
                        sequence * (rows - 1) + row - 1,
                        draft,
                        V,
                        scale,
                    )
                    stays = _uniform(seed, length + row - 1, ACCEPT_DRAW) * q < p
                going = cl.select(stays, 1, 0)
                count = count + going
        chosen = first + count
        token = cl.load(tokens, chosen, True, other=0)
        temperature = cl.bitcast(
            cl.load(table, chosen * SETTINGS + 2, True, other=0),
            cl.f32,
        )
        if (count + 1 < rows) & (temperature > 0.0):
            token = redraw(
                logits,
                stats,
                drafts,
                guesses,
                chosen,
                sequence * (rows - 1) + count,
                cl.load(inputs, chosen + 1, True, other=0),
                V,
                slot,
                LOG2E / temperature,
                cl.load(table, chosen * SETTINGS, True, other=0),
                length + count,
                floats,
                counts,
            )
        cl.store(accepted, entry, count + 1, True)
        cl.store(lengths, entry, length + count + 1, True)
        cl.store(nexts, entry, token, True)
        if history:
            for row in range(rows):
                cl.store(
                    kept[0],
                    entry * capacity + length + row,
                    cl.select(
                        row < count,
                        cl.load(inputs, first + row + 1, True, other=0),
                        token,
                    ),
                    (row <= count) & (length + row < capacity),
                )
        if last:
            cl.store(lasts[0], entry, offset + chosen, True)

    return accept_sampled


def distribution(device, logits, table, stats, *, rows, per, stride, vocab, dtype):
    artifact = make_distribution(dtype).compile(
        constants={"V": vocab},
        threads=THREADS,
        device=device,
        max_grid=rows,
    )
    return artifact.dispatch(
        device,
        [logits, table, stats],
        grid=(rows, 1, 1),
        scalars={"per": per, "stride": stride},
    )


def accept(
    device,
    inputs,
    tokens,
    lengths,
    accepted,
    nexts,
    *,
    sequences,
    rows,
    history=None,
    capacity=0,
    last=None,
    offset=0,
    logits=None,
    table=None,
    drafts=None,
    vocab=0,
    dtype=cl.bf16,
):
    """Return the dispatches for the acceptance of a verify step, on the GPU.

    Each of `sequences` sequences ran `rows` rows: inputs[s, 0] is the token that the
    step continues from and inputs[s, r] for r >= 1 are drafts; tokens[s, r] is the
    token chosen after row r. Of the drafts, the first n match the tokens chosen just
    before them, and n + 1 tokens stay: those n drafts and tokens[s, n]. Then
    accepted[s] = n + 1, lengths[s] advances by n + 1, so that it is the start
    position of the next step's rows, and nexts[s] = tokens[s, n] is the token the next
    step continues from. All buffers are u32; inputs and tokens hold (sequences, rows)
    values. A sequence with fewer drafts pads them with a token outside the
    vocabulary. With 1 row, this is a plain decode step's advance.

    With `history`, a (sequences, capacity) buffer, the kept tokens tokens[s, 0..n] also
    go to history[s, L..L + n], where L is lengths[s] before the step; positions from
    `capacity` on are not written. With `last`, last[s] = offset + s * rows + n, the
    flat index of the row that chose nexts[s].

    With `logits`, the step's (sequences * rows, vocab) logits of `dtype`, and `table`,
    the `settings` of each row as `sample` takes them, acceptance is sampled at rows
    with a temperature: the draft after row r stays with probability
    min(1, p(x) / q(x)), where p is the distribution that `sample` draws row r from
    and q the drafter's. q is the same settings on row r of `drafts`, the drafter's
    (sequences * (rows - 1), vocab) logits, or with no `drafts` one certain token. At
    the first draft that does not stay, the next token is drawn from max(0, p - q),
    normalized; when all drafts stay, it is tokens[s, rows - 1]. The draws are keyed
    by the row's seed and position, as in `sample`, so the output tokens follow p.
    Rows at temperature 0 keep greedy acceptance."""
    if rows < 1 or sequences < 1 or sequences * rows >= 2**32:
        raise ValueError("accept takes sequences and rows, under 2^32 tokens")
    if history is not None and not 1 <= capacity <= (2**32 - 1) // sequences:
        raise ValueError("history needs a capacity, under 2^32 tokens in all")
    flags = {"history": history is not None, "last": last is not None}
    outputs = [buffer for buffer in (history, last) if buffer is not None]
    scalars = {"rows": rows, "capacity": capacity, "offset": offset}
    if logits is None:
        artifact = make_accept(**flags).compile(
            constants={},
            threads=SEQUENCES,
            device=device,
            max_grid=(2**32 - 1) // SEQUENCES,
        )
        return [
            artifact.dispatch(
                device,
                [inputs, tokens, lengths, accepted, nexts, *outputs],
                grid=(-(-sequences // SEQUENCES), 1, 1),
                scalars={"sequences": sequences, **scalars},
            ),
        ]
    if table is None or not 1 <= vocab or sequences * rows * vocab >= 2**32:
        raise ValueError(
            "sampled acceptance takes the settings table and a vocabulary, under 2^32 "
            "logits",
        )
    stats = device.buffer(sequences * rows * 12)
    steps = [
        distribution(
            device,
            logits,
            table,
            stats,
            rows=sequences * rows,
            per=rows,
            stride=rows,
            vocab=vocab,
            dtype=dtype,
        ),
    ]
    guessing = []
    if drafts is not None:
        guesses = device.buffer(max(16, sequences * (rows - 1) * 12))
        guessing = [drafts, guesses]
        if rows > 1:
            steps.append(
                distribution(
                    device,
                    drafts,
                    table,
                    guesses,
                    rows=sequences * (rows - 1),
                    per=rows - 1,
                    stride=rows,
                    vocab=vocab,
                    dtype=dtype,
                ),
            )
    artifact = make_accept_sampled(dtype, drafted=drafts is not None, **flags).compile(
        constants={"V": vocab},
        threads=THREADS,
        device=device,
        max_grid=sequences,
    )
    steps.append(
        artifact.dispatch(
            device,
            [
                inputs,
                tokens,
                lengths,
                accepted,
                nexts,
                *outputs,
                logits,
                table,
                stats,
                *guessing,
            ],
            grid=(sequences, 1, 1),
            scalars=scalars,
        ),
    )
    return steps
