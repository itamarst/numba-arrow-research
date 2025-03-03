import timeit

import polars as pl
import awkward.numba
import awkward as ak
from numba import jit

large_series = pl.Series(range(1000))

series = pl.Series([1, 2, None, 4])
ak_arr = ak.from_arrow(series.to_arrow())
print("Converted to awkward:", ak_arr)

@jit
def my_sum(array):
    result = 0
    for value in array:
        if value is not None:
            result += value
    return result

@jit
def my_sum2(array):
    result = 0
    for value in array:
        result += value
    return result

print("Numba sum:", my_sum(ak_arr))
large_ak_arr = ak.from_arrow(large_series.to_arrow())

large_numpy_arr = large_series.to_numpy()
my_sum2(large_numpy_arr)

print("TIMING")
print("Polars:", timeit.timeit("large_series.sum()", globals=locals()))
print("Numba Numpy:", timeit.timeit("my_sum2(large_numpy_arr)", globals=locals()))
print("Numba Numpy w/conversion cost:", timeit.timeit("my_sum2(large_series.to_numpy())", globals=locals()))
print("Numba Arrow: ", timeit.timeit("my_sum(large_ak_arr)", globals=locals()))
print("Numba w/conversion cost: ", timeit.timeit("my_sum(ak.from_arrow(large_series.to_arrow()))", globals=locals()))


@jit
def add_one(builder, array):
    for i in range(len(array)):
        value = array[i]
        if value is None:
            builder.null()
        else:
            builder.integer(value + 1)
    return builder

builder = ak.ArrayBuilder()
add_one(builder, ak_arr)
new_ak_array = builder.snapshot()
print("New array:", new_ak_array)
arrow_result = ak.to_arrow(new_ak_array, extensionarray=False)
print("As arrow:", arrow_result, type(arrow_result))
print("New array as series:", pl.from_arrow(arrow_result))


def add_one_ak_e2e(series: pl.Series) -> pl.Series:
    ak_arr = ak.from_arrow(series.to_arrow())
    builder = ak.ArrayBuilder()
    add_one(builder, ak_arr)
    new_ak_array = builder.snapshot()
    arrow_result = ak.to_arrow(new_ak_array, extensionarray=False)
    return pl.from_arrow(arrow_result)

def add_one_pl(series: pl.Series) -> pl.Series:
    return series + 1

assert add_one_ak_e2e(large_series).to_list() == add_one_pl(large_series).to_list()

import timeit

timeit.timeit("add_one_ak_e2e(large_series)", globals=locals())
timeit.timeit("add_one_pl(large_series)", globals=locals())
