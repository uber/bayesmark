# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from hypothesis import given
from hypothesis.strategies import floats, integers, just, tuples
from hypothesis_gufunc.gufunc import gufunc_args


def identity(x):
    """When one needs a default mapping that does nothing."""
    return x


def seeds():
    return integers(min_value=0, max_value=(2 ** 32) - 1)


def probs():
    return floats(min_value=1e-3, max_value=1 - 1e-3)


def mfloats():
    return floats(min_value=-1e3, max_value=1e3)


def gufunc_floats(signature, min_side=0, max_side=5, unique=False, **kwargs):
    elements = floats(**kwargs)
    S = gufunc_args(signature, dtype=np.float_, elements=elements, unique=unique, min_side=min_side, max_side=max_side)
    return S


def close_enough(x, y, equal_nan=False, rtol=1e-5, atol=1e-8):
    # Might want to adjust rtol and atol for lower precision floats
    x, y = np.asarray(x), np.asarray(y)

    if x.shape != y.shape:
        return False

    if x.dtype != y.dtype:
        return False

    if x.dtype.kind == "f":
        assert y.dtype.kind == "f"
        # Note: equal_nan only considered in both float case!
        return np.allclose(x, y, equal_nan=equal_nan, rtol=rtol, atol=atol)

    return np.all(x == y)


def broadcasted(
    f, signature, itypes, otypes, elements, unique=False, excluded=(), min_side=0, max_side=5, max_dims_extra=2
):
    """Strategy that makes it easy to test the broadcasting semantics of a
    function against the 'ground-truth' broadcasting convention provided by
    :obj:`numpy.vectorize`.

    Parameters
    ----------
    f : callable
        This is the original function handles broadcasting itself. It must
        return an `ndarray` or multiple `ndarray` (which Python treats as a
        `tuple`) if returning 2-or-more output arguments.
    signature : str
        Signature for shapes to be compatible with. Expects string in format
        of numpy generalized universal function signature, e.g.,
        `'(m,n),(n)->(m)'` for vectorized matrix-vector multiplication.
        Officially, only supporting ascii characters.
    itypes : list-like of dtype
        List of numpy `dtype` for each argument. These can be either strings
        (``'int64'``), type (``np.int64``), or numpy `dtype`
        (``np.dtype('int64')``). A single `dtype` can be supplied for all
        arguments.
    otypes : list of dtype
        The dtype for the the outputs of `f`. It must be a list with one dtype
        for each output argument of `f`. It must be a singleton list if `f`
        only returns a single output. It can also be set to `None` to leave it
        to be inferred, but this can create issues with empty arrays, so it is
        not officially supported here.
    elements : list-like of strategy
        Strategies to fill in array elements on a per argument basis. One can
        also specify a single strategy
        (e.g., :func:`hypothesis.strategies.floats`)
        and have it applied to all arguments.
    unique : list-like of bool
        Boolean flag to specify if all elements in an array must be unique.
        One can also specify a single boolean to apply it to all arguments.
    excluded : list-like of integers
        Set of integers representing the positional for which the function will
        not be vectorized. Uses same format as :obj:`numpy.vectorize`.
    min_side : int or dict
        Minimum size of any side of the arrays. It is good to test the corner
        cases of 0 or 1 sized dimensions when applicable, but if not, a min
        size can be supplied here. Minimums can be provided on a per-dimension
        basis using a dict, e.g. ``min_side={'n': 2}``. One can use, e.g.,
        ``min_side={hypothesis.extra.gufunc.BCAST_DIM: 2}`` to limit the size
        of the broadcasted dimensions.
    max_side : int or dict
        Maximum size of any side of the arrays. This can usually be kept small
        and still find most corner cases in testing. Dictionaries can be
        supplied as with `min_side`.
    max_dims_extra : int
        Maximum number of extra dimensions that can be appended on left of
        arrays for broadcasting. This should be kept small as the memory used
        grows exponentially with extra dimensions.

    Returns
    -------
    f : callable
        This is the original function handles broadcasting itself.
    f_vec : callable
        Function that should be functionaly equivalent to `f` but broadcasting
        is handled by :obj:`numpy.vectorize`.
    res : tuple of ndarrays
        Resulting ndarrays with shapes consistent with `signature`. Extra
        dimensions for broadcasting will be present.

    Examples
    --------

    .. code-block:: pycon

      >>> import numpy as np
      >>> from hypothesis.strategies import integers, booleans
      >>> broadcasted(np.add, '(),()->()', ['int64'], ['int64', 'bool'],
                      elements=[integers(0,9), booleans()],
                      unique=[True, False]).example()
      (<ufunc 'add'>,
       <numpy.lib.function_base.vectorize at 0x11a777690>,
       (array([5, 6]), array([ True], dtype=bool)))
      >>> broadcasted(np.add, '(),()->()', ['int64'], ['int64', 'bool'],
                      elements=[integers(0,9), booleans()],
                      excluded=(1,)).example()
      (<ufunc 'add'>,
       <numpy.lib.function_base.vectorize at 0x11a715b10>,
       (array([9]), array(True, dtype=bool)))
      >>> f, fv, args = broadcasted(np.add, '(),()->()', ['int64'],
                                    ['int64', 'bool'],
                                    elements=[integers(0,9), booleans()],
                                    min_side=1, max_side=3,
                                    max_dims_extra=1).example()
      >>> f is np.add
      True
      >>> f(*args)
      7
      >>> fv(*args)
      array(7)
    """
    # cache and doc not needed for property testing, excluded not actually
    # needed here because we don't generate extra dims for the excluded args.
    # Using the excluded argument in np.vectorize only seems to confuse it in
    # corner cases.
    f_vec = np.vectorize(f, signature=signature, otypes=otypes)

    broadcasted_args = gufunc_args(
        signature,
        itypes,
        elements,
        unique=unique,
        excluded=excluded,
        min_side=min_side,
        max_side=max_side,
        max_dims_extra=max_dims_extra,
    )
    funcs_and_args = tuples(just(f), just(f_vec), broadcasted_args)
    return funcs_and_args


def broadcast_tester(
    f,
    signature,
    otype,
    excluded=(),
    dtype=np.float_,
    elements=None,
    unique=False,
    map_=identity,
    min_side=0,
    max_side=5,
    max_dims_extra=2,
    **kwargs,  # This still confuses flake8
):
    # Build the test for broadcasting with random dimensions
    elements = floats(**kwargs) if elements is None else elements

    @given(
        broadcasted(
            f,
            signature,
            otypes=[otype],
            excluded=excluded,
            itypes=dtype,
            elements=elements,
            unique=unique,
            min_side=min_side,
            max_side=max_side,
            max_dims_extra=max_dims_extra,
        )
    )
    def test_f(bargs):
        f0, f_vec, args = bargs
        args = map_(args)

        R1 = f0(*args)
        R2 = f_vec(*args)

        kind = np.dtype(otype).kind
        if kind in "US":  # Same kind ok for str and unicode dtypes
            assert R1.dtype.kind == kind
            assert R2.dtype.kind == kind
        elif otype is not None:
            assert R1.dtype == otype
            assert R2.dtype == otype
        assert close_enough(R1, R2, equal_nan=True)

    # Call the test
    test_f()


def multi_broadcast_tester(
    f,
    signature,
    otypes,
    excluded=(),
    dtype=np.float_,
    elements=None,
    unique=False,
    map_=identity,
    min_side=0,
    max_side=5,
    max_dims_extra=2,
    **kwargs,
):
    elements = floats(**kwargs) if elements is None else elements

    @given(
        broadcasted(
            f,
            signature,
            otypes=otypes,
            excluded=excluded,
            itypes=dtype,
            elements=elements,
            unique=unique,
            min_side=min_side,
            max_side=max_side,
            max_dims_extra=max_dims_extra,
        )
    )
    def test_f(bargs):
        f0, f_vec, args = bargs
        args = map_(args)

        R1 = f0(*args)
        R2 = f_vec(*args)
        for rr1, rr2, ot in zip(R1, R2, otypes):
            kind = np.dtype(ot).kind
            if kind in "US":  # Same kind ok for str and unicode dtypes
                assert R1.dtype.kind == kind
                assert R2.dtype.kind == kind
            else:
                assert rr1.dtype == ot
                assert rr2.dtype == ot
            assert close_enough(rr1, rr2, equal_nan=True)

    # Call the test
    test_f()
