import polars as pl
import awkward.numba
import awkward as ak
from numba import jit

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

print("Numba sum:", my_sum(ak_arr))
