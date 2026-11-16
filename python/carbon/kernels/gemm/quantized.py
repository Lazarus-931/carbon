"""f32 matrix products against packed (N, K) weights in MLX's row-major layout.

Codes occupy u32 words low bits first, with one scale and optional bias per `group`.
Formats supply `scale_bias` and `value`. Epilogues run before the output store,
with their operand buffers bound after the output.
"""

from dataclasses import dataclass
from functools import cache
from math import gcd
from typing import Callable

from carbon import language as cl, quantization as quant
from carbon.compiler import Atom, CopyAtom, Layout, Mode, TensorMatmul, TiledAtom, View
from carbon.kernels.common import operand_buffers
from carbon.kernels.gemm.epilogue import paired


@dataclass(frozen=True)
class Affine:
    """MLX's format: weight = scale * code + bias, codes unsigned, scales and biases in
    the activation type."""

    bits: int = 4
    group: int = 64

    def scale_type(self, dtype):
        return dtype

    def scale_bias(self, scales, biases, group, mask):
        scale = cl.cast(cl.load(scales, group, mask, other=0.0), cl.f32)
        return scale, cl.cast(cl.load(biases, group, mask, other=0.0), cl.f32)

    def value(self, code):
        return cl.cast(code, cl.f32)


@dataclass(frozen=True)
class Scaled:
    """weight = decode(code) * scale, e.g. FP8 or FP4 codes; MX formats use E8M0 scale
    bytes packed four per u32."""

    decode: Callable
    bits: int = 4
    group: int = 32
    microscaled: bool = True

    def scale_type(self, dtype):
        """E8M0 scale bytes in u32 words when microscaled, otherwise the activation
        type."""
        return cl.u32 if self.microscaled else dtype

    def scale_bias(self, scales, biases, group, mask):
        """The f32 scale of `group`, and no bias."""
        if self.microscaled:
            return quant.e8m0(quant.unpack(scales, group, 8, mask)), None
        return cl.cast(cl.load(scales, group, mask, other=0.0), cl.f32), None

    def value(self, code):
        return self.decode(code)


MXFP4 = Scaled(quant.fp4_e2m1, bits=4, group=32)
MXFP8 = Scaled(quant.fp8_e4m3fn, bits=8, group=32)


def flat(dtype, count):
    return View(dtype.name, (count,), (1,), count * dtype.itemsize)


def scale_view(fmt, dtype, n, k):
    groups = n * k // fmt.group
    return (
        flat(cl.u32, -(-groups // 4))
        if fmt.scale_type(dtype) == cl.u32
        else flat(dtype, groups)
    )


def per_word(fmt):
    """Codes per u32 word when whole words share one group; 0 for codes that straddle
    words (3, 5, 6, 7 bits)."""
    return (
        32 // fmt.bits
        if 32 % fmt.bits == 0 and fmt.group % (32 // fmt.bits) == 0
        else 0
    )


def packing(fmt):
    words = fmt.bits // gcd(fmt.bits, 32)
    return words, 32 * words // fmt.bits


def group_codes(words, index, fmt):
    word, shift = divmod(index * fmt.bits, 32)
    code = words[word] >> shift
    if shift + fmt.bits > 32:
        code = code | (words[word + 1] << (32 - shift))
    return code & ((1 << fmt.bits) - 1)


def vector_words(fmt, k):
    step = 32 * (per_word(fmt) or 8)
    return 1 if k % (2 * step) and not k % step else 2


def dequantize(packed, scales, biases, row, column, columns, mask, fmt):
    code = quant.unpack(packed, row * columns + column, fmt.bits, mask)
    scale, bias = fmt.scale_bias(
        scales,
        biases,
        row * (columns // fmt.group) + column // fmt.group,
        mask,
    )
    value = fmt.value(code)
    return value * scale if bias is None else cl.fma(value, scale, bias)


def word_codes(packed, scales, biases, row, word, columns, mask, codes, fmt):
    """Decode each word once, returning values in `codes` and scale/bias per word.
    `row`, `word` and `mask` share a layout; `codes` adds a thread-local last axis,
    so widening requires no communication."""
    count = 32 // fmt.bits
    bits = cl.load(packed, row * (columns // count) + word, mask, other=0)
    scale, bias = fmt.scale_bias(
        scales,
        biases,
        row * (columns // fmt.group) + word * count // fmt.group,
        mask,
    )
    rank = len(codes.shape)
    widened = cl.broadcast_to(
        bits,
        codes.shape,
        axes=tuple(range(rank - 3, rank - 1)),
        layout=codes,
    )
    shift = cl.arange(codes.size, layout=codes) % count * fmt.bits
    return fmt.value((widened >> shift) & ((1 << fmt.bits) - 1)), scale, bias


@cache
def make_dequantize(dtype, fmt, *, interleave=False, columns=32, threads=256):
    count = per_word(fmt)
    width = count or 8
    depth = threads // columns * width
    lanes = Layout.blocked((1, threads), threads=threads)
    codes = lanes.stack(width, 2)
    rows = CopyAtom(dtype, 4).layout((depth, columns), threads=threads)

    @cl.kernel
    def dequantize_weight(
        packed: cl.buffer(cl.u32),
        scales: cl.buffer(fmt.scale_type(dtype)),
        biases: cl.buffer(fmt.scale_type(dtype)),
        output: cl.buffer(dtype),
        N: cl.constexpr,
        K: cl.constexpr,
    ):
        packed = cl.view(packed, flat(cl.u32, -(-N * K * fmt.bits // 32)))
        output = cl.view(output, flat(dtype, N * K))
        BLOCKS = (N + columns - 1) // columns
        first_column = cl.program_id() % BLOCKS * columns
        first_depth = cl.program_id() // BLOCKS * depth
        thread = cl.arange(threads, layout=lanes)
        column = first_column + thread // (threads // columns)
        row = column % 2 * (N // 2) + column // 2 if interleave else column
        start = first_depth + thread % (threads // columns) * width
        live = (column < N) & (start < K)
        if count:
            value, scale, bias = word_codes(
                packed,
                scales,
                biases,
                row,
                start // count,
                K,
                live,
                codes,
                fmt,
            )
            scale = cl.broadcast_to(scale, codes.shape, axes=(0, 1), layout=codes)
            value = (
                value * scale
                if bias is None
                else cl.fma(
                    value,
                    scale,
                    cl.broadcast_to(bias, codes.shape, axes=(0, 1), layout=codes),
                )
            )
        index = cl.arange(codes.size, layout=codes)
        offset = index // width % (threads // columns) * width + index % width
        if not count:
            depths = first_depth + offset
            value = dequantize(
                packed,
                scales,
                biases,
                cl.broadcast_to(row, codes.shape, axes=(0, 1), layout=codes),
                depths,
                K,
                cl.broadcast_to(live, codes.shape, axes=(0, 1), layout=codes)
                & (depths < K),
                fmt,
            )
        block = cl.scratch((depth, columns), dtype)
        cl.store(
            block,
            offset * columns + index // width // (threads // columns),
            cl.cast(value, dtype),
            True,
        )
        cl.barrier()
        element = cl.arange(depth * columns, layout=rows)
        target, place = (
            first_depth + element // columns,
            first_column + element % columns,
        )
        cl.store(
            output,
            target * N + place,
            cl.load(block, element, True, other=0.0),
            (target < K) & (place < N),
        )

    return dequantize_weight


@cache
def make_qmv(
    dtype,
    fmt,
    *,
    outputs=4,
    words=2,
    rows=1,
    glu=None,
    epilogue=None,
    stride=None,
):
    """Small-row products: a SIMD group per feature, lanes over K words.
    Replicated x tiles avoid communication between groups. Each affine word
    contributes scale * sum(code * x) + bias * sum(x)."""
    scale_type = fmt.scale_type(dtype)
    epilogue_buffers = operand_buffers(epilogue)
    count = per_word(fmt)
    width = 32 * words
    step = width * (count or 8)
    if count:
        weights = Layout(
            (outputs, width),
            Mode(outputs, width),
            Mode(32, 1),
            Mode(words, 32),
        )
        codes = weights.stack(count, 2)
        inputs = Layout(
            (width, count),
            Mode(1, 0),
            Mode(32, count),
            Mode((words, count), (32 * count, 1)),
        )
    else:
        blocked = Layout.blocked((step,), threads=32)
        weights = codes = Layout(
            (outputs, step),
            Mode(outputs, step),
            blocked.lanes,
            blocked.slots,
        )
        inputs = Layout((step,), Mode(1, 0), blocked.lanes, blocked.slots)
    inputs = inputs.across_groups(outputs)
    features = weights.reduced(1)

    def storage(N, H, K):
        S = H if stride is None else stride
        return (
            flat(dtype, rows * K),
            flat(cl.u32, -(-N * K * fmt.bits // 32)),
            flat(dtype, (rows - 1) * S + H),
        )

    def activations(x, start, K, origin):
        column = start + cl.arange(inputs.size, layout=inputs)
        value = cl.load(
            x,
            origin + column,
            True if K % step == 0 else column < K,
            other=0.0,
        )
        return cl.broadcast_to(
            cl.cast(value, cl.f32),
            codes.shape,
            axes=tuple(range(1, len(codes.shape))),
            layout=codes,
        )

    def unpacked(packed, scales, biases, first, start, N, K, whole):
        index = cl.arange(weights.size, layout=weights)
        feature = first + index // weights.shape[1]
        if not count:
            column = start + index % step
            return (
                dequantize(
                    packed,
                    scales,
                    biases,
                    feature,
                    column,
                    K,
                    (feature < N) & (column < K),
                    fmt,
                ),
                None,
                None,
            )
        word = start // count + index % width
        mask = True if whole else (feature < N) & (word < K // count)
        return word_codes(packed, scales, biases, feature, word, K, mask, codes, fmt)

    def product(weight, value):
        code, scale, bias = weight
        if not count:
            return code * value
        result = scale * cl.sum(code * value, 2, layout=weights)
        return (
            result
            if bias is None
            else cl.fma(bias, cl.sum(value, 2, layout=weights), result)
        )

    def zeros():
        halves = 1 if glu is None else 2
        return tuple(
            cl.full(weights.shape, 0.0, cl.f32, layout=weights)
            for _ in range(rows * halves)
        )

    def accumulate(packed, scales, biases, x, firsts, totals, start, N, K, whole):
        values = [activations(x, start, K, row * K) for row in range(rows)]
        parts = [
            unpacked(packed, scales, biases, first, start, N, K, whole)
            for first in firsts
        ]
        pairs = [(part, value) for part in parts for value in values]
        return tuple(
            total + product(part, value) for total, (part, value) in zip(totals, pairs)
        )

    def finish(output, operands, totals, first, H):
        sums = [cl.sum(total, 1, layout=features) for total in totals]
        column = first + cl.arange(outputs, layout=features)
        for row in range(rows):
            value = sums[row] if glu is None else glu(sums[row], sums[rows + row])
            if epilogue is not None:
                value = epilogue(
                    value,
                    cl.full((outputs,), row, cl.u32, layout=features),
                    column,
                    operands,
                )
            cl.store(
                output,
                row * (H if stride is None else stride) + column,
                cl.cast(value, dtype),
                column < H,
            )

    @cl.kernel
    def qmv(
        x: cl.buffer(dtype),
        packed: cl.buffer(cl.u32),
        scales: cl.buffer(scale_type),
        biases: cl.buffer(scale_type),
        output: cl.buffer(dtype),
        operands: epilogue_buffers,
        N: cl.constexpr,
        K: cl.constexpr,
    ):
        H = N if glu is None else N // 2
        X, PACKED, OUTPUT = storage(N, H, K)
        x, packed, output = (
            cl.view(x, X),
            cl.view(packed, PACKED),
            cl.view(output, OUTPUT),
        )
        first = cl.program_id() * outputs
        firsts = (first,) if glu is None else (first, first + H)
        totals = zeros()
        for start in range(0, K, step):
            totals = accumulate(
                packed,
                scales,
                biases,
                x,
                firsts,
                totals,
                start,
                N,
                K,
                K % step == 0 and H % outputs == 0,
            )
        finish(output, operands, totals, first, H)

    return qmv


@cache
def make_qmv_rows(
    dtype,
    fmt,
    *,
    rows,
    features=4,
    simds=2,
    glu=None,
    epilogue=None,
    stride=None,
):
    """Few-row products over whole-word codes, using fused multiply-adds.
    Each SIMD group owns `features` outputs, with lanes over K words. Rows share
    unpacked weights; features share x loads."""
    scale_type = fmt.scale_type(dtype)
    epilogue_buffers = operand_buffers(epilogue)
    count = per_word(fmt)
    step = 32 * count
    block = simds * features
    weights = Layout(
        (block, 32),
        Mode(simds, features * 32),
        Mode(32, 1),
        Mode(features, 32),
    )
    lanes = Layout.striped((32,), threads=32).across_groups(simds)
    outputs = weights.reduced(1)
    halves = 1 if glu is None else 2

    def storage(N, H, K):
        S = H if stride is None else stride
        return (
            flat(dtype, rows * K),
            flat(cl.u32, -(-N * K * fmt.bits // 32)),
            flat(dtype, (rows - 1) * S + H),
        )

    def unpacked(packed, scales, biases, first, start, N, K, whole):
        index = cl.arange(weights.size, layout=weights)
        feature = first + index // 32
        word = start // count + index % 32
        mask = True if whole else (feature < N) & (word < K // count)
        bits = cl.load(packed, feature * (K // count) + word, mask, other=0)
        scale, bias = fmt.scale_bias(
            scales,
            biases,
            feature * (K // fmt.group) + word * count // fmt.group,
            mask,
        )
        values = [
            fmt.value((bits >> (fmt.bits * offset)) & ((1 << fmt.bits) - 1))
            for offset in range(count)
        ]
        if bias is None:
            return [value * scale for value in values]
        return [cl.fma(value, scale, bias) for value in values]

    def accumulate(packed, scales, biases, x, firsts, totals, start, N, K, whole):
        parts = [
            unpacked(packed, scales, biases, first, start, N, K, whole)
            for first in firsts
        ]
        lane = cl.arange(32, layout=lanes)
        column = (start // count + lane) * count
        live = True if K % step == 0 else column < K
        totals = list(totals)
        for row in range(rows):
            loaded = [
                cl.load(x, row * K + column + offset, live, other=0.0)
                for offset in range(count)
            ]
            values = [
                cl.broadcast_to(
                    cl.cast(value, cl.f32),
                    weights.shape,
                    axes=(1,),
                    layout=weights,
                )
                for value in loaded
            ]
            for half, weight in enumerate(parts):
                at = half * rows + row
                for offset in range(count):
                    totals[at] = cl.fma(weight[offset], values[offset], totals[at])
        return tuple(totals)

    def finish(output, operands, totals, first, H):
        sums = [cl.sum(total, 1, layout=outputs) for total in totals]
        column = first + cl.arange(block, layout=outputs)
        for row in range(rows):
            value = sums[row] if glu is None else glu(sums[row], sums[rows + row])
            if epilogue is not None:
                value = epilogue(
                    value,
                    cl.full((block,), row, cl.u32, layout=outputs),
                    column,
                    operands,
                )
            cl.store(
                output,
                row * (H if stride is None else stride) + column,
                cl.cast(value, dtype),
                column < H,
            )

    @cl.kernel
    def qmv_rows(
        x: cl.buffer(dtype),
        packed: cl.buffer(cl.u32),
        scales: cl.buffer(scale_type),
        biases: cl.buffer(scale_type),
        output: cl.buffer(dtype),
        operands: epilogue_buffers,
        N: cl.constexpr,
        K: cl.constexpr,
    ):
        H = N if glu is None else N // 2
        X, PACKED, OUTPUT = storage(N, H, K)
        x, packed, output = (
            cl.view(x, X),
            cl.view(packed, PACKED),
            cl.view(output, OUTPUT),
        )
        first = cl.program_id() * block
        firsts = (first,) if glu is None else (first, first + H)
        totals = tuple(
            cl.full(weights.shape, 0.0, cl.f32, layout=weights)
            for _ in range(rows * halves)
        )
        for start in range(0, K, step):
            totals = accumulate(
                packed,
                scales,
                biases,
                x,
                firsts,
                totals,
                start,
                N,
                K,
                K % step == 0 and H % block == 0,
            )
        finish(output, operands, totals, first, H)

    return qmv_rows


@cache
def make_qmm_registers(
    dtype,
    fmt,
    *,
    tiles=1,
    extra=0,
    split=4,
    glu=None,
    epilogue=None,
    stride=None,
):
    """8-row matrix tiles with register-only operands: output^T = W x^T.
    Each lane unpacks neighbouring W codes into operand A; x^T is operand B.
    `extra` rows reuse those weights through fused multiply-adds."""
    atom = Atom(cl.f32)
    threads = 32 * split
    weights, inputs = (
        atom.layout("a", threads=threads),
        atom.layout("b", threads=threads),
    )
    scale_type = fmt.scale_type(dtype)
    epilogue_buffers = operand_buffers(epilogue)
    words, count = packing(fmt)
    block = 8 if glu is not None else 16

    def multiply(
        packed,
        scales,
        biases,
        xs,
        spares,
        feature,
        position,
        depth,
        N,
        K,
        totals,
        sums,
    ):
        place = depth + count * position
        scale, bias = fmt.scale_bias(
            scales,
            biases,
            feature * (K // fmt.group) + place // fmt.group,
            feature < N,
        )
        first = (feature * (K // count) + place // count) * words
        loaded = [
            cl.load(packed, first + word, feature < N, other=0) for word in range(words)
        ]
        for offset in range(count):
            value = fmt.value(group_codes(loaded, offset, fmt))
            weight = value * scale if bias is None else cl.fma(value, scale, bias)
            totals = tuple(
                atom.mma(weight, values[offset], total)
                for values, total in zip(xs, totals)
            )
            sums = tuple(
                cl.fma(weight, values[offset], total)
                for values, total in zip(spares, sums)
            )
        return totals + sums

    def step(
        x,
        packed,
        scales,
        biases,
        blocks,
        position,
        column,
        rows,
        spare_rows,
        depth,
        x_depth,
        N,
        K,
        state,
    ):
        loaded = [
            [
                cl.load(x, (row, x_depth + count * column + offset), True, other=0.0)
                for offset in range(count)
            ]
            for row in rows
        ]
        xs = [[cl.cast(value, cl.f32) for value in line] for line in loaded]
        spared = [
            [
                cl.load(x, (row, depth + count * position + offset), True, other=0.0)
                for offset in range(count)
            ]
            for row in spare_rows
        ]
        spares = [[cl.cast(value, cl.f32) for value in line] for line in spared]
        width = tiles + extra
        return tuple(
            value
            for index, feature in enumerate(blocks)
            for value in multiply(
                packed,
                scales,
                biases,
                xs,
                spares,
                feature,
                position,
                depth,
                N,
                K,
                state[index * width : index * width + tiles],
                state[index * width + tiles : (index + 1) * width],
            )
        )

    def zeros():
        return tuple(
            atom.zeros()
            if index < tiles
            else cl.full(weights.shape, 0.0, cl.f32, layout=weights)
            for _ in range(2)
            for index in range(tiles + extra)
        )

    def gathered(totals):
        stacked = cl.stack(list(totals))
        if split == 1:
            return stacked
        parts = cl.reshape(stacked, (len(totals), split, 8, 8))
        return cl.sum(parts, 1, scratch=cl.scratch((len(totals), split, 8, 8), cl.f32))

    def features(total):
        lines = cl.sum(total, 1, layout=weights.reduced(1))
        if split == 1:
            return lines
        parts = cl.reshape(lines, (split, 8))
        return cl.sum(parts, 0, scratch=cl.scratch((split, 8), cl.f32))

    def store(output, operands, value, row, column, rows, H):
        if epilogue is not None:
            value = epilogue(value, row, column, operands)
        cl.store(
            output,
            row * (H if stride is None else stride) + column,
            cl.cast(value, dtype),
            (row < rows) & (column < H),
        )

    def finish(output, operands, state, rows, first, H):
        width = tiles + extra
        totals = state[:tiles] + state[width : width + tiles]
        value = (
            gathered(totals)
            if glu is None
            else glu(gathered(totals[:tiles]), gathered(totals[tiles:]))
        )
        e = cl.arange(value.layout.size, layout=value.layout)
        store(
            output,
            operands,
            value,
            e // 64 % tiles * 8 + e % 8,
            first + e // 64 // tiles * 8 + e % 64 // 8,
            rows,
            H,
        )
        for row in range(extra):
            gate = features(state[tiles + row])
            up = features(state[width + tiles + row])
            lanes = gate.layout
            line = cl.full((8,), 8 * tiles + row, cl.u32, layout=lanes)
            column = first + cl.arange(8, layout=lanes)
            if glu is None:
                store(output, operands, gate, line, column, rows, H)
                store(output, operands, up, line, column + 8, rows, H)
            else:
                store(output, operands, glu(gate, up), line, column, rows, H)

    @cl.kernel
    def qmm_registers(
        x: cl.buffer(dtype, rank=2),
        packed: cl.buffer(cl.u32),
        scales: cl.buffer(scale_type),
        biases: cl.buffer(scale_type),
        output: cl.buffer(dtype),
        operands: epilogue_buffers,
        N: cl.constexpr,
        K: cl.constexpr,
    ):
        rows = cl.dim(x, 0)
        H = N if glu is None else N // 2
        packed = cl.view(packed, flat(cl.u32, N * K // count * words))
        scales = cl.view(scales, scale_view(fmt, dtype, N, K))
        biases = (
            cl.view(biases, scale_view(fmt, dtype, N, K))
            if isinstance(fmt, Affine)
            else biases
        )
        first = cl.program_id() * block
        span = K // split
        a = cl.arange(weights.size, layout=weights)
        position, part = a % 8, a // 64
        blocks = (first + a // 8 % 8, first + (8 if glu is None else H) + a // 8 % 8)
        b = cl.arange(inputs.size, layout=inputs)
        column = b % 64 // 8
        lines = tuple(cl.minimum(b % 8 + 8 * tile, rows - 1) for tile in range(tiles))
        spare_rows = tuple(
            cl.minimum(rows - 1, 8 * tiles + row) for row in range(extra)
        )
        state = zeros()
        for depth in range(0, span, 8 * count):
            state = step(
                x,
                packed,
                scales,
                biases,
                blocks,
                position,
                column,
                lines,
                spare_rows,
                depth + part * span,
                depth + b // 64 * span,
                N,
                K,
                state,
            )
        finish(output, operands, state, rows, first, H)

    return qmm_registers


@cache
def make_qmm(
    dtype,
    fmt,
    *,
    block=(32, 32),
    block_k=32,
    tensor=False,
    glu=None,
    epilogue=None,
    stride=None,
):
    """Double-buffer dequantized weights and, for SIMD-group atoms, x tiles.
    Weights use n-major storage viewed as (K, N). The default uses 2 x 2 SIMD groups;
    `tensor=True` reads ranked x directly with tensor ops. Both accumulate in f32.
    The shared double-buffer schedule publishes writes and completes readers before
    either stage is reused."""
    tiled = TiledAtom(Atom(dtype), (2, 2), (block[0] // 16, block[1] // 16))
    scale_type = fmt.scale_type(dtype)
    block_m, block_n = block
    threads = tiled.threads
    count = per_word(fmt)
    left_layout = CopyAtom(dtype, 4).layout((block_m, block_k), threads=threads)
    if count:
        weights = Layout.striped((block_n, block_k // count), threads=threads)
        codes = weights.stack(count, 2)
    else:
        weights = codes = Layout.striped((block_n, block_k), threads=threads)
    right_view = View(
        dtype.name,
        (block_k, block_n),
        (1, block_k),
        block_k * block_n * dtype.itemsize,
        address_space="threadgroup",
    )

    operation = TensorMatmul(dtype)
    epilogue_buffers = operand_buffers(epilogue)
    width = block_n if glu is None else block_n // 2

    def words(N, K):
        return flat(cl.u32, -(-N * K * fmt.bits // 32))

    def stage_right(packed, scales, biases, right, first_column, start, N, K):
        index = cl.arange(weights.size, layout=weights)
        local = index // weights.shape[1]
        if glu is None:
            feature = live = first_column + local
            bound = N
        else:
            live = first_column + local // 2
            feature, bound = local % 2 * (N // 2) + live, N // 2
        if count:
            word = start // count + index % weights.shape[1]
            code, scale, bias = word_codes(
                packed,
                scales,
                biases,
                feature,
                word,
                K,
                (live < bound) & (word < K // count),
                codes,
                fmt,
            )
            scale = cl.broadcast_to(scale, codes.shape, axes=(0, 1), layout=codes)
            weight = (
                code * scale
                if bias is None
                else cl.fma(
                    code,
                    scale,
                    cl.broadcast_to(bias, codes.shape, axes=(0, 1), layout=codes),
                )
            )
        else:
            column = start + index % block_k
            weight = dequantize(
                packed,
                scales,
                biases,
                feature,
                column,
                K,
                (live < bound) & (column < K),
                fmt,
            )
        cl.store(
            right,
            cl.arange(codes.size, layout=codes),
            cl.cast(weight, dtype),
            True,
        )

    def stage(
        x,
        packed,
        scales,
        biases,
        left,
        right,
        first_row,
        first_column,
        start,
        rows,
        N,
        K,
    ):
        a = cl.arange(left_layout.size, layout=left_layout)
        a_row, a_column = first_row + a // block_k, a % block_k
        value = cl.load(
            x,
            a_row * K + start + a_column,
            (a_row < rows) & (start + a_column < K),
            other=0.0,
        )
        cl.store(left, a, value, True)
        stage_right(packed, scales, biases, right, first_column, start, N, K)

    def multiply(left, right, accumulators):
        group_row, group_column = (
            cl.simd_id() // tiled.groups[1],
            cl.simd_id() % tiled.groups[1],
        )
        weights = cl.view(right, right_view)
        for step in range(block_k // 8):
            left_fragments = tiled.load_a(left, group_row * tiled.fragments[0], step)
            right_fragments = tiled.load_b(
                weights,
                group_column * tiled.fragments[1],
                step,
            )
            accumulators = tiled.mma(left_fragments, right_fragments, accumulators)
        return accumulators

    def produce(block, buffers, arguments):
        x, packed, scales, biases, first_row, first_column, rows, n, k = arguments
        if tensor:
            stage_right(
                packed,
                scales,
                biases,
                buffers[0],
                first_column,
                block * block_k,
                n,
                k,
            )
        else:
            stage(
                x,
                packed,
                scales,
                biases,
                buffers[0],
                buffers[1],
                first_row,
                first_column,
                block * block_k,
                rows,
                n,
                k,
            )

    def consume(block, buffers, total, arguments):
        if tensor:
            x, packed, scales, biases, first_row, first_column, rows, n, k = arguments
            blocks = (
                (n + block_n - 1) // block_n
                if glu is None
                else (n // 2 + width - 1) // width
            )
            steps = (k + block_k - 1) // block_k
            tile = cl.program_id() // blocks * steps + block
            return operation.run(
                cl.partition(x, (block_m, block_k), tile),
                cl.view(buffers[0], right_view),
                total,
            )
        return multiply(buffers[0], buffers[1], total)

    def finish(output, operands, result, first_row, first_column, rows, N):
        H = N if glu is None else N // 2
        result = (
            result
            if glu is None
            else paired(glu, cl.reshape(result, (block_m, block_n)))
        )
        index = cl.arange(block_m * width, layout=result.layout)
        out_row, out_column = first_row + index // width, first_column + index % width
        if epilogue is not None:
            result = epilogue(result, out_row, out_column, operands)
        cl.store(
            output,
            out_row * (H if stride is None else stride) + out_column,
            cl.cast(result, dtype),
            (out_row < rows) & (out_column < H),
        )

    @cl.kernel
    def qmm(
        x: cl.buffer(dtype),
        packed: cl.buffer(cl.u32),
        scales: cl.buffer(scale_type),
        biases: cl.buffer(scale_type),
        output: cl.buffer(dtype),
        operands: epilogue_buffers,
        rows: cl.u32,
        N: cl.constexpr,
        K: cl.constexpr,
    ):
        packed = cl.view(packed, words(N, K))
        BLOCKS = ((N if glu is None else N // 2) + width - 1) // width
        first_row = cl.program_id() // BLOCKS * block_m
        first_column = cl.program_id() % BLOCKS * width
        lefts = (
            cl.scratch((block_m, block_k), dtype),
            cl.scratch((block_m, block_k), dtype),
        )
        rights = (
            cl.scratch((block_n * block_k,), dtype),
            cl.scratch((block_n * block_k,), dtype),
        )
        accumulators = tiled.zeros()
        accumulators = cl.double_buffer(
            (K + block_k - 1) // block_k,
            buffers=((lefts[0], rights[0]), (lefts[1], rights[1])),
            state=accumulators,
            produce=produce,
            consume=consume,
            args=(x, packed, scales, biases, first_row, first_column, rows, N, K),
        )
        finish(
            output,
            operands,
            tiled.unpack(accumulators),
            first_row,
            first_column,
            rows,
            N,
        )

    @cl.kernel
    def tensor_qmm(
        x: cl.buffer(dtype, rank=2),
        packed: cl.buffer(cl.u32),
        scales: cl.buffer(scale_type),
        biases: cl.buffer(scale_type),
        output: cl.buffer(dtype),
        operands: epilogue_buffers,
        rows: cl.u32,
        N: cl.constexpr,
        K: cl.constexpr,
    ):
        packed = cl.view(packed, words(N, K))
        BLOCKS = ((N if glu is None else N // 2) + width - 1) // width
        first_row = cl.program_id() // BLOCKS * block_m
        first_column = cl.program_id() % BLOCKS * width
        rights = (
            cl.scratch((block_n * block_k,), dtype),
            cl.scratch((block_n * block_k,), dtype),
        )
        total = operation.zeros((block_m, block_n))
        total = cl.double_buffer(
            (K + block_k - 1) // block_k,
            buffers=((rights[0],), (rights[1],)),
            state=total,
            produce=produce,
            consume=consume,
            args=(x, packed, scales, biases, first_row, first_column, rows, N, K),
        )
        finish(output, operands, total, first_row, first_column, rows, N)

    return tensor_qmm if tensor else qmm
