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


@jit
def add_one(array, builder):
    for i in range(len(array)):
        value = array[i]
        if value is None:
            builder.null()
        else:
            builder.integer(value + 1)
    return builder


def add_one_e2e(series: pl.Series) -> pl.Series:
    ak_arr = ak.from_arrow(series.to_arrow())
    builder = ak.ArrayBuilder(initial=len(series) // 8)
    add_one(ak_arr, builder)
    new_ak_array = builder.snapshot()
    arrow_result = ak.to_arrow(new_ak_array, extensionarray=False)
    return pl.from_arrow(arrow_result)


print(add_one_e2e(series))
timeit("add_one(series):", lambda: add_one_e2e(series), count=100)
timeit("series + 1:", lambda: series + 1)
timeit(
    "Pure Python:",
    lambda: pl.Series((value if value is None else value + 1 for value in series)),
    count=10,
)
