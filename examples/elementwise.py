from carbon import language as cl


@cl.kernel
def multiply(
    left: cl.buffer(cl.f32, rank=2),
    right: cl.buffer(cl.f32, rank=2),
    output: cl.buffer(cl.f32, rank=2),
    scale: cl.f32,
    BM: cl.constexpr,
    BN: cl.constexpr,
):
    """output = left * right * scale on runtime-shaped views, one (BM, BN) tile per
    threadgroup."""
    index = cl.arange(BM * BN)
    a = cl.partition(left, (BM, BN), cl.program_id())
    b = cl.partition(right, (BM, BN), cl.program_id())
    destination = cl.partition(output, (BM, BN), cl.program_id())
    x = cl.load(a, index, True, other=0.0)
    y = cl.load(b, index, True, other=0.0)
    cl.store(destination, index, x * y * scale, True)
