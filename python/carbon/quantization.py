"""Packed weight decoding through ordinary Carbon operations, with f32 results."""

from . import language as cl


def unpack(storage, index, bits, mask=True):
    """Read 1–8-bit codes from a low-bit-first u32 stream; mask logical padding."""
    if type(bits) is not int or not 1 <= bits <= 8:
        raise ValueError("packed codes require 1 to 8 bits")
    if 32 % bits == 0:
        per_word = 32 // bits
        word = cl.load(storage, index // per_word, mask, other=0)
        return (word >> ((index % per_word) * bits)) & ((1 << bits) - 1)
    offset = (index % 32) * bits
    word = (index // 32) * bits + offset // 32
    shift = offset % 32
    low = cl.load(storage, word, mask, other=0)
    high = cl.load(storage, word + 1, mask & (shift > 32 - bits), other=0)
    return ((low >> shift) | (high << (32 - shift))) & ((1 << bits) - 1)


def decode_int(code, bits, *, signed=False):
    """Decode unsigned or two's-complement 1–8-bit codes; scales stay explicit."""
    if type(bits) is not int or not 1 <= bits <= 8 or type(signed) is not bool:
        raise ValueError("integer codes require 1 to 8 bits and a boolean signed flag")
    code = code & ((1 << bits) - 1)
    if signed:
        sign = 1 << (bits - 1)
        return cl.cast(code & (sign - 1), cl.f32) - cl.cast(code & sign, cl.f32)
    return cl.cast(code, cl.f32)


def _float(code, exponent_bits, mantissa_bits, bias, special):
    mantissa_mask = (1 << mantissa_bits) - 1
    exponent_mask = (1 << exponent_bits) - 1
    mantissa = code & mantissa_mask
    exponent = (code >> mantissa_bits) & exponent_mask
    normal = ((exponent + (127 - bias)) << 23) | (mantissa << (23 - mantissa_bits))
    subnormal = cl.cast(mantissa, cl.f32) * (2.0 ** (1 - bias - mantissa_bits))
    magnitude = cl.select(exponent == 0, cl.bitcast(subnormal, cl.u32), normal)
    if special == "ieee":
        exceptional = cl.select(mantissa == 0, 0x7F800000, 0x7FC00000)
        magnitude = cl.select(exponent == exponent_mask, exceptional, magnitude)
    elif special == "nan":
        magnitude = cl.select(
            (exponent == exponent_mask) & (mantissa == mantissa_mask),
            0x7FC00000,
            magnitude,
        )
    sign = (code >> (exponent_bits + mantissa_bits)) & 1
    return cl.bitcast(magnitude | (sign << 31), cl.f32)


def fp8_e4m3fn(code):
    """Decode E4M3FN, including signed zero and NaN; all other codes are finite."""
    return _float(code, 4, 3, 7, "nan")


def fp8_e5m2(code):
    """Decode E5M2, including signed zero, infinities, and NaNs."""
    return _float(code, 5, 2, 15, "ieee")


def fp6_e3m2(code):
    """Decode finite E3M2 with bias 3 and signed zero."""
    return _float(code, 3, 2, 3, "finite")


def fp6_e2m3(code):
    """Decode finite E2M3 with bias 1 and signed zero."""
    return _float(code, 2, 3, 1, "finite")


def fp4_e2m1(code):
    """Decode finite E2M1 with bias 1 and signed zero."""
    return _float(code, 2, 1, 1, "finite")


def e8m0(code):
    """Decode an unsigned MX scale; 255 is NaN and zero encodes 2**-127."""
    code = code & 255
    bits = cl.select(code == 0, 0x00400000, code << 23)
    return cl.bitcast(cl.select(code == 255, 0x7FC00000, bits), cl.f32)
