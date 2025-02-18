# Research and maybe experiments in adding Arrow support to Numba

## Why?

* Numba only supports NumPy arrays out of the box.
* Lots of projects now use Arrow (Polars, Pandas, if used PyArrow directly, no doubt others).
* Numba is a nice way to write _fast_ extensions without switching languages.
* Current Numba usage involves converting to NumPy arrays and then back, which is a problem because it loses information about missing data.
