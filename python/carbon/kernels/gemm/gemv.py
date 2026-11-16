from functools import cache

from carbon import language as cl
from carbon.compiler import Layout
from carbon.kernels.common import operand_buffers


@cache
def make_cooperative_gemv(dtype=cl.f32, *, splits=1, epilogue=None):
    """Divide K across tile rows and optional f32 partial-output partitions. `epilogue`
    maps each output tile before the store, and its operands bind after `output`; a
    split product's epilogue goes to make_sum_partials."""
    if dtype not in (cl.f16, cl.bf16, cl.f32):
        raise ValueError("storage must be f16, bf16, or f32")
    if type(splits) is not int or not 1 <= splits < 2**32:
        raise ValueError("splits must be a positive u32")
    if splits > 1 and epilogue is not None:
        raise ValueError("a split product's epilogue belongs to make_sum_partials")
    output_type = cl.f32 if splits > 1 else dtype
    epilogue_buffers = operand_buffers(epilogue)

    def configure(left, right, output, schedule):
        """(K, N, schedule rows, schedule columns, splits) for LEFT (1, K), RIGHT (K, N)
        and OUTPUT (splits, N)."""
        if len(left.shape) != 2 or left.shape[0] != 1 or len(right.shape) != 2:
            raise ValueError("GEMV requires shapes (1, K) and (K, N)")
        k, n = right.shape
        if left.shape[1] != k or output.shape != (splits, n):
            raise ValueError("incompatible GEMV or partial-output dimensions")
        if len(schedule.shape) != 2:
            raise ValueError("GEMV schedule needs reduction and column axes")
        if max(left.size, right.size, output.size) >= 2**32:
            raise ValueError("GEMV views must fit u32 indexing")
        return k, n, *schedule.shape, splits

    def result_layout(schedule):
        return Layout.striped((schedule.shape[1],), threads=schedule.threads)

    @cl.kernel
    def cooperative_gemv(
        left: cl.buffer(dtype),
        right: cl.buffer(dtype),
        output: cl.buffer(output_type),
        operands: epilogue_buffers,
        LEFT: cl.constexpr,
        RIGHT: cl.constexpr,
        OUTPUT: cl.constexpr,
        SCHEDULE: cl.constexpr,
    ):
        K, N, ROWS, COLUMNS, SPLITS = configure(LEFT, RIGHT, OUTPUT, SCHEDULE)
        a = cl.view(left, LEFT)
        b = cl.view(right, RIGHT)
        c = cl.view(output, OUTPUT)
        if N > 0:
            BLOCKS = (N + COLUMNS - 1) // COLUMNS
            partition = cl.program_id() // BLOCKS
            column = cl.program_id() % BLOCKS
            index = cl.arange(SCHEDULE.size, layout=SCHEDULE)
            shared = cl.scratch(SCHEDULE.shape, cl.f32)
            accumulated = cl.full(SCHEDULE.shape, 0.0, cl.f32, layout=SCHEDULE)
            for start in range(0, K, ROWS * SPLITS):
                offset = partition * ROWS + index // COLUMNS
                x = cl.load(a, start + offset, offset < K - start, other=0.0)
                window = cl.partition(
                    b,
                    (ROWS, COLUMNS),
                    (start // ROWS + partition) * BLOCKS + column,
                )
                y = cl.load(window, index, True, other=0.0)
                accumulated = accumulated + cl.cast(x, cl.f32) * cl.cast(y, cl.f32)
            total = cl.sum(
                accumulated,
                0,
                layout=result_layout(SCHEDULE),
                scratch=shared,
            )
            destination = cl.partition(c, (1, COLUMNS), cl.program_id())
            output_index = cl.arange(COLUMNS, layout=result_layout(SCHEDULE))
            if epilogue is not None:
                first_row = cl.full(
                    (COLUMNS,),
                    0,
                    cl.u32,
                    layout=result_layout(SCHEDULE),
                )
                total = epilogue(
                    total,
                    first_row,
                    column * COLUMNS + output_index,
                    operands,
                )
            cl.store(destination, output_index, cl.cast(total, output_type), True)

    return cooperative_gemv
