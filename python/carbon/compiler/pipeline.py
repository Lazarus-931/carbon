from .. import language as cl
from .builder import Value, current_builder


@cl.inline
def _double_buffer(count, buffers, state, produce, consume, args):
    if count > 0:
        produce(0, buffers[0], args)
        cl.barrier()
        for first in range(0, count - 1, 2):
            produce(first + 1, buffers[1], args)
            state = consume(first, buffers[0], state, args)
            cl.barrier()
            if first + 2 < count:
                produce(first + 2, buffers[0], args)
            state = consume(first + 1, buffers[1], state, args)
            cl.barrier()
        if count % 2 != 0:
            state = consume(count - 1, buffers[0], state, args)
            cl.barrier()
    return state


def double_buffer(count, *, buffers, state, produce, consume, args):
    builder = current_builder()
    if isinstance(count, Value):
        builder.check(count)
        if count.view or count.width or count.element != "i32":
            raise TypeError("double-buffer counts must be u32 scalars")
    elif type(count) is not int or not 0 <= count < 2**32:
        raise ValueError("double-buffer counts must be nonnegative u32 integers")
    if type(buffers) is not tuple or any(type(stage) is not tuple for stage in buffers):
        raise TypeError(
            "double buffering takes a tuple of two tuples of scratch buffers",
        )
    if not callable(produce) or not callable(consume) or type(args) is not tuple:
        raise TypeError(
            "double buffering requires producer/consumer callables and a tuple of "
            "arguments",
        )
    for stage in buffers:
        for buffer in stage:
            if not isinstance(buffer, Value):
                raise TypeError("double buffering requires scratch buffers")
            builder.check(buffer)
    builder.native.check_stages(
        [[buffer.handle for buffer in stage] for stage in buffers],
    )
    return _double_buffer(count, buffers, state, produce, consume, args)
