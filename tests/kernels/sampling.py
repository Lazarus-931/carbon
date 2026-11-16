import numpy as np
import pytest

from carbon import language as cl
from carbon.kernels.ops import sampling
from testbed import device, storage, upload, widen


def submit(metal, steps):
    metal.prepare(steps if isinstance(steps, list) else [steps]).submit().wait()


def bf16(metal, values):
    return upload(metal, storage(values, cl.bf16))


def test_sample_greedy_rows_pick_the_maximum():
    metal, rows, vocab = device(), 6, 3000
    rng = np.random.default_rng(11)
    logits = rng.standard_normal((rows, vocab)) * 2
    winners = rng.integers(0, vocab, rows)
    logits[np.arange(rows), winners] = logits.max() + 1
    table = sampling.settings(
        range(1, rows + 1),
        [0.0, 0.0, 1.0, 1.0, 0.5, 0.0],
        top_k=[0, 0, 1, 1, 1, 0],
    )
    tokens = metal.buffer(rows * 4)
    submit(
        metal,
        sampling.sample(
            metal,
            bf16(metal, logits),
            upload(metal, table),
            upload(metal, np.zeros(rows, np.uint32)),
            tokens,
            rows=rows,
            vocab=vocab,
        ),
    )
    assert np.array_equal(np.frombuffer(tokens.read(), np.uint32), winners)


def test_accept_keeps_matching_drafts_and_advances_lengths():
    metal, rows = device(), 5
    rng = np.random.default_rng(4)
    matches = [0, 1, 4, 2, 0, 3] * 50
    sequences = len(matches)
    tokens = rng.integers(0, 1000, (sequences, rows)).astype(np.uint32)
    inputs = rng.integers(1000, 2000, (sequences, rows)).astype(np.uint32)
    for sequence, count in enumerate(matches):
        inputs[sequence, 1 : count + 1] = tokens[sequence, :count]
    inputs[4, 2:] = tokens[4, 1:-1]
    lengths = rng.integers(0, 4000, sequences).astype(np.uint32)
    buffers = [upload(metal, array) for array in (inputs, tokens, lengths)]
    accepted, nexts = metal.buffer(sequences * 4), metal.buffer(sequences * 4)
    submit(
        metal,
        sampling.accept(
            metal,
            *buffers,
            accepted,
            nexts,
            sequences=sequences,
            rows=rows,
        ),
    )
    kept = np.array(matches, np.uint32) + 1
    assert np.array_equal(np.frombuffer(accepted.read(), np.uint32), kept)
    assert np.array_equal(np.frombuffer(buffers[2].read(), np.uint32), lengths + kept)
    chosen = tokens[np.arange(sequences), np.array(matches)]
    assert np.array_equal(np.frombuffer(nexts.read(), np.uint32), chosen)


def test_accept_one_row_is_a_decode_step():
    metal, sequences = device(), 3
    tokens = np.array([[7], [8], [9]], np.uint32)
    lengths = np.array([5, 0, 99], np.uint32)
    buffers = [upload(metal, array) for array in (tokens, tokens, lengths)]
    accepted, nexts = metal.buffer(sequences * 4), metal.buffer(sequences * 4)
    submit(
        metal,
        sampling.accept(
            metal,
            *buffers,
            accepted,
            nexts,
            sequences=sequences,
            rows=1,
        ),
    )
    assert list(np.frombuffer(accepted.read(), np.uint32)) == [1, 1, 1]
    assert list(np.frombuffer(buffers[2].read(), np.uint32)) == [6, 1, 100]
    assert list(np.frombuffer(nexts.read(), np.uint32)) == [7, 8, 9]


@pytest.mark.parametrize("rows", [1, 4])
def test_accept_writes_history_and_last_rows(rows):
    metal, capacity = device(), 64
    matches = [0, 3, 3, 2] if rows > 1 else [0, 0, 0, 0]
    sequences = len(matches)
    rng = np.random.default_rng(rows)
    tokens = rng.integers(0, 1000, (sequences, rows)).astype(np.uint32)
    inputs = rng.integers(1000, 2000, (sequences, rows)).astype(np.uint32)
    for sequence, count in enumerate(matches):
        inputs[sequence, 1 : count + 1] = tokens[sequence, :count]
    lengths = np.array([5, 0, 62, 30], np.uint32)
    history = np.full((sequences, capacity), 7777, np.uint32)
    buffers = [upload(metal, array) for array in (inputs, tokens, lengths, history)]
    accepted, nexts, last = (metal.buffer(sequences * 4) for _ in range(3))
    submit(
        metal,
        sampling.accept(
            metal,
            *buffers[:3],
            accepted,
            nexts,
            sequences=sequences,
            rows=rows,
            history=buffers[3],
            capacity=capacity,
            last=last,
            offset=100,
        ),
    )
    expected = history.copy()
    for sequence, count in enumerate(matches):
        for row in range(count + 1):
            if lengths[sequence] + row < capacity:
                expected[sequence, lengths[sequence] + row] = tokens[sequence, row]
    assert np.array_equal(
        np.frombuffer(buffers[3].read(), np.uint32).reshape(sequences, capacity),
        expected,
    )
    assert list(np.frombuffer(last.read(), np.uint32)) == [
        100 + sequence * rows + count for sequence, count in enumerate(matches)
    ]
    assert np.array_equal(
        np.frombuffer(buffers[2].read(), np.uint32),
        lengths + np.array(matches, np.uint32) + 1,
    )


def speculate(metal, logits, inputs, table, starts, drafts=None):
    """Sample every verify row, then accept with sampling; returns accepted, nexts."""
    sequences, rows, vocab = logits.shape
    positions = (starts[:, None] + np.arange(rows)).astype(np.uint32)
    target = bf16(metal, logits.reshape(-1, vocab))
    settings = upload(metal, table)
    tokens = metal.buffer(sequences * rows * 4)
    accepted, nexts = metal.buffer(sequences * 4), metal.buffer(sequences * 4)
    guesses = None if drafts is None else bf16(metal, drafts.reshape(-1, vocab))
    submit(
        metal,
        [
            sampling.sample(
                metal,
                target,
                settings,
                upload(metal, positions.reshape(-1)),
                tokens,
                rows=sequences * rows,
                vocab=vocab,
            ),
            *sampling.accept(
                metal,
                upload(metal, inputs),
                tokens,
                upload(metal, starts.astype(np.uint32)),
                accepted,
                nexts,
                sequences=sequences,
                rows=rows,
                logits=target,
                table=settings,
                drafts=guesses,
                vocab=vocab,
            ),
        ],
    )
    return (
        np.frombuffer(accepted.read(), np.uint32),
        np.frombuffer(nexts.read(), np.uint32),
        np.frombuffer(tokens.read(), np.uint32).reshape(sequences, rows),
    )


def kept_distribution(logits, temperature, top_k=0, top_p=1.0, min_p=0.0):
    """The distribution `sample` draws from, for distinct bf16 logits."""
    values = widen(storage(logits, cl.bf16), cl.bf16).astype(np.float64)
    order = np.argsort(-values)
    weights = np.exp((values - values.max()) / temperature)
    keep = np.ones(len(values), bool)
    if top_k:
        keep[order[top_k:]] = False
    if min_p:
        keep &= weights >= min_p
    if top_p < 1.0:
        mass = weights[keep].sum()
        running, nucleus = 0.0, np.zeros(len(values), bool)
        for token in order:
            if keep[token] and running < top_p * mass:
                nucleus[token] = True
                running += weights[token]
        keep = nucleus
    weights = np.where(keep, weights, 0.0)
    return weights / weights.sum()


def assert_follows(outputs, p):
    frequency = np.bincount(outputs, minlength=len(p)) / len(outputs)
    sigma = np.sqrt(p * (1 - p) / len(outputs))
    assert np.all(np.abs(frequency - p) <= 5 * sigma + 1e-3), (frequency, p)


SAMPLED = [
    ("drafter", 2, {}),
    ("drafter", 3, {}),
    ("certain", 2, {}),
    ("drafter", 2, {"top_k": 6, "top_p": 0.8, "min_p": 0.02}),
    ("certain", 3, {"top_k": 6, "top_p": 0.8, "min_p": 0.02}),
]


@pytest.mark.parametrize(
    ("kind", "rows", "filters"),
    SAMPLED,
    ids=[f"{kind}-rows{rows}-{len(filters)}filters" for kind, rows, filters in SAMPLED],
)
def test_sampled_accept_keeps_the_target_distribution(kind, rows, filters):
    metal, sequences, vocab, temperature = device(), 4096, 16, 0.8
    rng = np.random.default_rng(rows + len(filters))
    shape = (sequences, rows, vocab)
    logits = np.broadcast_to(rng.permutation(vocab) * 0.25 - 2.0, shape).copy()
    drafted = np.broadcast_to(rng.permutation(vocab) * 0.25 - 2.0, shape).copy()
    p = kept_distribution(logits[0, 0], temperature, **filters)
    q = kept_distribution(drafted[0, 0], temperature, **filters)
    inputs = rng.integers(0, vocab, (sequences, rows)).astype(np.uint32)
    if kind == "drafter":
        inputs[:, 1:] = rng.choice(vocab, (sequences, rows - 1), p=q)
    else:
        inputs[:, 1:] = np.argsort(-p)[1]
    table = sampling.settings(
        np.arange(sequences * rows) // rows + 1,
        [temperature] * (sequences * rows),
        top_k=filters.get("top_k"),
        top_p=None
        if "top_p" not in filters
        else [filters["top_p"]] * (sequences * rows),
        min_p=None
        if "min_p" not in filters
        else [filters["min_p"]] * (sequences * rows),
    )
    starts = rng.integers(0, 1000, sequences)
    accepted, nexts, _ = speculate(
        metal,
        logits,
        inputs,
        table,
        starts,
        drafts=drafted[:, 1:] if kind == "drafter" else None,
    )
    first = np.where(accepted > 1, inputs[:, 1], nexts)
    assert_follows(first, p)
    assert 0 < np.mean(accepted > 1) < 1


def test_sampled_accept_keeps_every_draft_of_the_same_distribution():
    metal, sequences, rows, vocab = device(), 64, 4, 300
    rng = np.random.default_rng(9)
    logits = rng.standard_normal((sequences, rows, vocab)) * 2
    table = sampling.settings(
        np.arange(sequences * rows) + 1,
        [1.0] * (sequences * rows),
    )
    inputs = rng.integers(0, vocab, (sequences, rows)).astype(np.uint32)
    accepted, nexts, tokens = speculate(
        metal,
        logits,
        inputs,
        table,
        rng.integers(0, 1000, sequences),
        drafts=logits[:, :-1],
    )
    assert np.all(accepted == rows)
    assert np.array_equal(nexts, tokens[:, -1])


def test_sampled_accept_at_temperature_zero_is_greedy():
    metal, sequences, rows, vocab = device(), 40, 5, 500
    rng = np.random.default_rng(12)
    logits = rng.standard_normal((sequences, rows, vocab)) * 2
    winners = widen(storage(logits, cl.bf16), cl.bf16).argmax(-1)
    matches = rng.integers(0, rows, sequences)
    inputs = rng.integers(0, vocab, (sequences, rows)).astype(np.uint32)
    for sequence, count in enumerate(matches):
        inputs[sequence, 1 : count + 1] = winners[sequence, :count]
    table = sampling.settings(
        np.arange(sequences * rows) + 1,
        [0.0] * (sequences * rows),
    )
    accepted, nexts, _ = speculate(
        metal,
        logits,
        inputs,
        table,
        rng.integers(0, 1000, sequences),
    )
    expected = np.array(
        [
            next(
                (
                    row
                    for row in range(1, rows)
                    if inputs[s, row] != winners[s, row - 1]
                ),
                rows,
            )
            for s in range(sequences)
        ],
    )
    assert np.array_equal(accepted, expected)
    assert np.array_equal(nexts, winners[np.arange(sequences), expected - 1])
