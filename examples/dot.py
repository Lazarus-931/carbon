from carbon import language as cl


@cl.kernel
def dot_rows(
    left: cl.buffer(cl.f16),
    right: cl.buffer(cl.f16),
    output: cl.buffer(cl.f32),
    columns: cl.u32,
    SCHEDULE: cl.constexpr,
):
    """output[r] = row r of `left` dotted with row r of `right`, in f32, one row per
    threadgroup."""
    shared = cl.scratch(SCHEDULE.shape, cl.f32)
    local = cl.arange(SCHEDULE.size, layout=SCHEDULE)
    accumulated = cl.full(SCHEDULE.shape, 0.0, cl.f32, layout=SCHEDULE)
    for start in range(0, columns, SCHEDULE.size):
        column = start + local
        index = cl.program_id() * columns + column
        active = local < columns - start
        a = cl.cast(cl.load(left, index, active, other=0.0), cl.f32)
        b = cl.cast(cl.load(right, index, active, other=0.0), cl.f32)
        accumulated = accumulated + a * b
    total = cl.sum(accumulated, 0, scratch=shared)
    row = cl.full((), cl.program_id(), cl.u32)
    cl.store(output, row, total, True)
