from functools import cache

from carbon import language as cl
from carbon.compiler import TensorMatmul
from carbon.kernels.common import operand_buffers


@cache
def make_tensor_matmul(dtype=cl.f16, *, tile=(64, 64), epilogue=None):
    """output = left @ right on the GPU's matrix units: the Neural Accelerators on Apple
    family 10 (M5).

    One `tile` of output per threadgroup of four SIMD groups; the hardware op walks K
    and clips the edges itself. `epilogue` maps each output tile before the store, and
    its operands bind after `output`; the product then comes back to registers instead
    of the matrix unit writing the output itself.
    """
    operation = TensorMatmul(dtype)
    epilogue_buffers = operand_buffers(epilogue)

    @cl.kernel
    def matmul(
        left: cl.buffer(dtype),
        right: cl.buffer(dtype),
        output: cl.buffer(dtype),
        operands: epilogue_buffers,
        LEFT: cl.constexpr,
        RIGHT: cl.constexpr,
        OUTPUT: cl.constexpr,
    ):
        BLOCK_M, BLOCK_N = tile
        K, N = RIGHT.shape
        BLOCKS = (N + BLOCK_N - 1) // BLOCK_N
        a = cl.partition(cl.view(left, LEFT), (BLOCK_M, K), cl.program_id() // BLOCKS)
        b = cl.partition(cl.view(right, RIGHT), (K, BLOCK_N), cl.program_id() % BLOCKS)
        c = cl.partition(cl.view(output, OUTPUT), (BLOCK_M, BLOCK_N), cl.program_id())
        total = operation.run(a, b, operation.zeros((BLOCK_M, BLOCK_N)))
        index = cl.arange(BLOCK_M * BLOCK_N, layout=total.layout)
        if epilogue is not None:
            row = cl.program_id() // BLOCKS * BLOCK_M + index // BLOCK_N
            column = cl.program_id() % BLOCKS * BLOCK_N + index % BLOCK_N
            total = epilogue(total, row, column, operands)
        cl.store(c, index, cl.cast(total, dtype), True)

    return matmul
