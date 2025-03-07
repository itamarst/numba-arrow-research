from time import time

import polars as pl
import awkward.numba
import awkward as ak
from numba import jit


def timeit(prefix, f, count=1000):
    start = time()
    for _ in range(count):
        f()
    print(f"{prefix}:", (time() - start) / count)


def time_from_arrow(size):
    series = pl.Series(list(range(size)))
    assert len(series) == size
    timeit(f"Array of size {size}", lambda: ak.from_arrow(series.to_arrow()))


time_from_arrow(1000)
time_from_arrow(10_000)
time_from_arrow(100_000)


@jit
def do_some_math(array):
    result = 0.0
    for value in array:
        if value is not None:
            result += value * 0.7
    return result


def time_do_some_math(size):
    series = pl.Series(list(range(size)))
    assert len(series) == size
    timeit(
        f"Math on Series of size {size}",
        lambda: do_some_math(ak.from_arrow(series.to_arrow())),
    )


print(do_some_math(ak.from_arrow(pl.Series([1, 2]).to_arrow())))
time_do_some_math(1000)
time_do_some_math(10_000)
time_do_some_math(100_000)
time_do_some_math(1_000_000)

series = pl.Series(list(range(1_000_000)))
timeit("series.sum():", series.sum)
