from dataclasses import dataclass

from carbon import language as cl
from carbon.compiler import View
from carbon.kernels.common import Hook

LOG2E = 1.4426950408889634
KEYS = 16

ScoreMod = Hook
OutputMod = Hook


@dataclass(frozen=True, eq=False)
class Call:
    q: object
    keys: object
    values: object
    table: object
    lengths: object
    output: object
    rows: int
    heads: int
    kv_heads: int
    head_dim: int
    page_size: int
    max_pages: int
    context: int
    causal: bool
    window: int
    segments: object
    sequences: int
    longest: int
    score_mod: object
    score_operands: tuple
    output_mod: object
    output_operands: tuple
    factor: float
    dtype: object
    bounds: object = None

    @property
    def group(self):
        return self.heads // self.kv_heads

    @property
    def paged(self):
        return self.table is not None

    @property
    def segmented(self):
        return self.segments is not None

    @property
    def packed(self):
        return self.segmented and bool(self.max_pages)

    @property
    def tables(self):
        return (self.table,) if self.paged else ()

    @property
    def cuts(self):
        return (self.bounds,) if self.segmented else ()

    @property
    def cache(self):
        count = self.keys.size // self.dtype.itemsize
        return View(self.dtype.name, (count,), (1,), count * self.dtype.itemsize)

    @property
    def options(self):
        return {
            "head_dim": self.head_dim,
            "group": self.group,
            "page_size": self.page_size,
            "paged": self.paged,
            "causal": self.causal,
            "window": self.window,
            "segmented": self.segmented,
        }


def span(row, position, start, rows, bounds, *, causal, window):
    low, end = 0, position + 1 if causal else start + rows
    if bounds:
        low = start + cl.load(bounds[0], row * 2, True, other=0)
        if not causal:
            end = start + cl.load(bounds[0], row * 2 + 1, True, other=0)
    if window and causal:
        low = cl.maximum(low, cl.maximum(position + 1, window) - window)
    elif window:
        low = cl.maximum(low, cl.maximum(position, window) - window)
        end = cl.minimum(end, position + window + 1)
    return low, end


def scored(score_mod, product, factor, row, head, key, operands):
    if score_mod is None:
        return product * factor
    return score_mod(product * (factor / LOG2E), row, head, key, operands) * LOG2E


def ceil_div(value, divisor):
    return (value - 1) // divisor + 1
