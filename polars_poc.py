"""
Proof-of-concept Polars integration.
"""

from functools import wraps
from typing import Callable, TypeVar

import polars as pl
from polars._typing import PolarsDataType
import awkward as ak
import awkward.numba  # register with Numba
from numba import jit

_ReturnType = TypeVar("_ReturnType")


def arrow_jit(
    return_dtype: PolarsDataType, returns_scalar: bool
) -> Callable[[Callable[[pl.Series], _ReturnType]], Callable[[pl.Expr], pl.Expr]]:
    """
    Decorator factory that wrap a function with Numba.

    The final wrapped function takes a ``pl.Expr`` and calls ``map_batches()``
    on it with the original function accelerated by Numba.Expr()

    This would be slightly nicer if it were built-in to Polars, but probably
    not there yet.
    """

    def decorator(
        func: Callable[[pl.Series], _ReturnType]
    ) -> Callable[[pl.Expr], pl.Expr]:
        # Convert to Numba function:
        jit_func = jit(func)

        if returns_scalar:

            def run_func(series: pl.Series) -> _ReturnType:
                ak_arr = ak.from_arrow(series.to_arrow())
                return jit_func(ak_arr)

        else:

            def run_func(series: pl.Series) -> _ReturnType:
                builder = ak.ArrayBuilder()
                ak_arr = ak.from_arrow(series.to_arrow())
                jit_func(ak_arr, builder)
                result = pl.from_arrow(
                    ak.to_arrow(builder.snapshot(), extensionarray=False)
                )
                # TODO do something with return_dtype?
                return result

        @wraps(func)
        def wrapped(expr: pl.Expr) -> pl.Expr:
            return expr.map_batches(
                run_func, return_dtype=return_dtype, returns_scalar=returns_scalar
            )

        return wrapped

    return decorator


if __name__ == "__main__":

    @arrow_jit(return_dtype=pl.Float64(), returns_scalar=True)
    def make_scalar(arr):
        result = 0.0
        for i in range(len(arr)):
            value = arr[i]
            if value is not None:
                result += value * 0.7
        return result

    @arrow_jit(return_dtype=pl.Int64(), returns_scalar=False)
    def make_series(arr, array_builder):
        for i in range(len(arr)):
            value = arr[i]
            if value is None:
                array_builder.null()
            else:
                array_builder.integer(value + 1)

    df = pl.DataFrame({"values": [17, 2, None, 5]})
    print(df.select(make_scalar(pl.col("values"))))
    print(df.select(make_series(pl.col("values"))))
