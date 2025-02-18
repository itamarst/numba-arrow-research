# Research and maybe experiments in adding Arrow support to Numba

## Why?

* Numba only supports NumPy arrays out of the box.
* Lots of projects now use Arrow (Polars, Pandas, if used PyArrow directly, no doubt others).
* Numba is a nice way to write _fast_ extensions without switching languages.
* Current Numba usage involves converting to NumPy arrays and then back, which is a problem because it loses information about missing data.

## Existing attempts/conversations

[Apparently](https://numba.discourse.group/t/feature-request-about-supporting-arrow-in-numba/1668/2) the Awkward Array library uses the same data representation as Arrow for its columns, and can therefore convert to/from Arrow with zero-copy.
And Awkward Array has a Numba integration provided.
So this may just be a matter of documentation rather than coding.

Next step, then:

1. Validate that awkward array is indeed zero-copy from Arrow.
2. Play around with the Numba integration and see if it works.
3. In particular, do a proof-of-concept with Polars.

Docs: https://awkward-array.org/doc/main/user-guide/how-to-use-in-numba-features.html

## What I've learned so far

* Pandas is a required dependency for the Awkard Arrow integration to work.
  Annoying but not the end of the world.
* Not necessarily much in the way of APIs to access?
