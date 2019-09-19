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
from hypothesis import assume, given
from hypothesis.extra.numpy import from_dtype
from hypothesis.strategies import booleans, floats, integers, just, lists, sampled_from
from hypothesis_gufunc.gufunc import gufunc_args as gufunc
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelBinarizer

import bo_benchmark.space as sp
from bo_benchmark.np_util import linear_rescale
from bo_benchmark.space import CAT_DTYPE, CAT_KIND, CAT_NATIVE_DTYPE
from hypothesis_util import broadcast_tester, close_enough, gufunc_floats
from util import space_configs

INT_MIN = np.iinfo("i").min
INT_MAX = np.iinfo("i").max

WARPS = ("logit", "linear", "bilog", "log")
ENCODER_DTYPES = ("bool", "int", "float")


def encoder_gen(args):
    X, labels, assume_sorted, dtype, assume_valid = args

    if assume_sorted:
        labels = np.sort(labels)
    X = labels[X % len(labels)]
    dtype = dtype.item()  # np.array does not like np.array(dtype)
    return X, labels, assume_sorted, dtype, assume_valid


def decoder_gen(args):
    Y, labels, assume_sorted, dtype, assume_valid = args

    if assume_sorted:
        labels = np.sort(labels)
    dtype = dtype.item()
    return Y, labels, assume_sorted, dtype, assume_valid


def decoder_gen_broadcast(args):
    Y, labels, assume_sorted = args

    if assume_sorted:
        labels = np.sort(labels)
    return Y, labels, assume_sorted


@given(
    gufunc(
        "(),(n),(),(),()->(n)",
        dtype=[np.int_, CAT_DTYPE, np.bool_, str, np.bool_],
        elements=[
            integers(0, INT_MAX),
            from_dtype(np.dtype(CAT_DTYPE)),
            booleans(),
            sampled_from(ENCODER_DTYPES),
            booleans(),
        ],
        unique=[False, True, False, False, False],
        min_side={"n": 1},
    ).map(encoder_gen)
)
def test_encode_decode(args):
    X, labels, assume_sorted, dtype, assume_valid = args

    Y = sp.encode(X, labels, assume_sorted=assume_sorted, dtype=dtype, assume_valid=assume_valid)
    if assume_sorted:  # otherwise labels will be re-arranged
        (idx,), = np.where(Y > 0)
        assert np.asarray(labels[idx]) == X
    assert Y.dtype == dtype

    X2 = sp.decode(Y, labels, assume_sorted=assume_sorted)

    assert close_enough(X, X2)


@given(
    gufunc(
        "(m),(n),(),(),()->(n)",
        dtype=[np.int_, CAT_DTYPE, np.bool_, str, np.bool_],
        elements=[
            integers(0, INT_MAX),
            from_dtype(np.dtype(CAT_DTYPE)),
            booleans(),
            sampled_from(ENCODER_DTYPES),
            booleans(),
        ],
        unique=[False, True, False, False, False],
        min_side={"m": 1, "n": 3},
    ).map(encoder_gen)
)
def test_encoder_to_sklearn(args):
    # sklearn cannot handle this correctly unless n >= 3
    X, labels, assume_sorted, dtype, assume_valid = args

    Y = sp.encode(X, labels, assume_sorted=assume_sorted, dtype=dtype, assume_valid=assume_valid)

    enc = LabelBinarizer()
    enc.fit(labels)
    Y2 = enc.transform(X)

    assert close_enough(Y, Y2.astype(dtype))


@given(
    gufunc(
        "(m,n),(n),(),(),()->(n)",
        dtype=[np.float_, CAT_DTYPE, np.bool_, str, np.bool_],
        elements=[floats(), from_dtype(np.dtype(CAT_DTYPE)), booleans(), sampled_from(ENCODER_DTYPES), booleans()],
        unique=[False, True, False, False, False],
        min_side={"n": 1},
    ).map(decoder_gen)
)
def test_decode_encode(args):
    Y, labels, assume_sorted, dtype, assume_valid = args

    assert Y.ndim >= 1 and Y.shape[-1] == len(labels)

    X = sp.decode(Y, labels, assume_sorted=assume_sorted)
    Y2 = sp.encode(X, labels, assume_sorted=assume_sorted, dtype=dtype, assume_valid=assume_valid)

    # The encoding is defined as the argmax
    assert np.all(Y.argmax(axis=1) == Y2.argmax(axis=1))
    assert np.all(np.sum(Y2 != 0, axis=1) == 1)
    assert np.all(np.sum(Y2 == 1, axis=1) == 1)


@given(
    gufunc(
        "(m,n),(n),(),(),()->(n)",
        dtype=[np.float_, CAT_DTYPE, np.bool_, str, np.bool_],
        elements=[floats(), from_dtype(np.dtype(CAT_DTYPE)), booleans(), sampled_from(ENCODER_DTYPES), booleans()],
        unique=[False, True, False, False, False],
        min_side={"m": 1, "n": 3},
    ).map(decoder_gen)
)
def test_decode_to_sklearn(args):
    Y, labels, assume_sorted, dtype, assume_valid = args

    assert Y.ndim >= 1 and Y.shape[-1] == len(labels)

    X = sp.decode(Y, labels, assume_sorted=assume_sorted)

    enc = LabelBinarizer()
    enc.fit(labels)
    X2 = enc.inverse_transform(Y)

    assert X.dtype.kind == CAT_KIND
    assert close_enough(X, X2.astype(X.dtype))


def test_encode_broadcast_bool():
    broadcast_tester(
        sp.encode,
        "(),(n),(),(),()->(n)",
        otype=bool,
        excluded=(1, 2, 3, 4),
        dtype=[np.int_, CAT_DTYPE, np.bool_, object, np.bool_],
        elements=[integers(0, INT_MAX), from_dtype(np.dtype(CAT_DTYPE)), booleans(), just("bool"), booleans()],
        unique=[False, True, False, False, False],
        min_side={"n": 1},
        map_=encoder_gen,
    )


def test_encode_broadcast_int():
    broadcast_tester(
        sp.encode,
        "(),(n),(),(),()->(n)",
        otype=int,
        excluded=(1, 2, 3, 4),
        dtype=[np.int_, CAT_DTYPE, np.bool_, object, np.bool_],
        elements=[integers(0, INT_MAX), from_dtype(np.dtype(CAT_DTYPE)), booleans(), just("int"), booleans()],
        unique=[False, True, False, False, False],
        min_side={"n": 1},
        map_=encoder_gen,
    )


def test_encode_broadcast_float():
    broadcast_tester(
        sp.encode,
        "(),(n),(),(),()->(n)",
        otype=float,
        excluded=(1, 2, 3, 4),
        dtype=[np.int_, CAT_DTYPE, np.bool_, object, np.bool_],
        elements=[integers(0, INT_MAX), from_dtype(np.dtype(CAT_DTYPE)), booleans(), just("float"), booleans()],
        unique=[False, True, False, False, False],
        min_side={"n": 1},
        map_=encoder_gen,
    )


def test_decode_broadcast_bool():
    broadcast_tester(
        sp.decode,
        "(m,n),(n),()->(m)",
        otype=CAT_DTYPE,
        excluded=(1, 2),
        dtype=[np.bool_, CAT_DTYPE, np.bool_],
        elements=[booleans(), from_dtype(np.dtype(CAT_DTYPE)), booleans()],
        unique=[False, True, False],
        min_side={"n": 1},
        map_=decoder_gen_broadcast,
    )


def test_decode_broadcast_int():
    broadcast_tester(
        sp.decode,
        "(m,n),(n),()->(m)",
        otype=CAT_DTYPE,
        excluded=(1, 2),
        dtype=[np.int_, CAT_DTYPE, np.bool_],
        elements=[integers(INT_MIN, INT_MAX), from_dtype(np.dtype(CAT_DTYPE)), booleans()],
        unique=[False, True, False],
        min_side={"n": 1},
        map_=decoder_gen_broadcast,
    )


def test_decode_broadcast_float():
    broadcast_tester(
        sp.decode,
        "(m,n),(n),()->(m)",
        otype=CAT_DTYPE,
        excluded=(1, 2),
        dtype=[np.float_, CAT_DTYPE, np.bool_],
        elements=[floats(), from_dtype(np.dtype(CAT_DTYPE)), booleans()],
        unique=[False, True, False],
        min_side={"n": 1},
        map_=decoder_gen_broadcast,
    )


@given(gufunc("()->()", dtype=np.float_, elements=floats()))
def test_bilog_props(args):
    x, = args

    y = sp.bilog(x)

    assert sp.bilog(0) == 0  # This could be its own test
    assert close_enough(y, -sp.bilog(-x), equal_nan=True)
    assert np.isfinite(y) == np.isfinite(x)


@given(gufunc_floats("(2)->(2)", allow_infinity=False, allow_nan=False))
def test_bilog_monotonic(args):
    x, = args

    x1, x2 = sorted(np.abs(x))

    assert sp.bilog(x1) < sp.bilog((1 + 1e-6) * x2 + 1e-6)


@given(gufunc("()->()", dtype=np.float_, elements=floats()))
def test_bilog_biexp(args):
    x, = args

    assert close_enough(sp.biexp(sp.bilog(x)), x, equal_nan=True)


def test_bilog_broadcast():
    broadcast_tester(sp.bilog, "()->()", otype=float)


def test_biexp_broadcast():
    broadcast_tester(sp.biexp, "()->()", otype=float, min_value=-10, max_value=10)


@given(sampled_from(WARPS), gufunc_floats("(n),(m)->(n)", allow_infinity=False, allow_nan=False))
def test_real_values_warp_unwarp(warp, args):
    x, values = args

    if warp == "log":
        values = values[values > 0]
    if warp == "logit":
        values = values[(0 < values) & (values < 1)]

    # We could eliminate need for this if we split out test for log and logit
    # cases and specify unique flag, but works as is
    v = np.unique(values)
    assume(len(v) >= 2)

    f = interp1d(v, v, kind="nearest", fill_value="extrapolate")
    x = f(x)
    assert x.ndim == 1  # make sure interp1d did not mess it up

    S = sp.Real(warp=warp, values=values)
    y = S.warp(x)
    assert y.shape == x.shape + (1,)
    assert y.dtype == sp.WARPED_DTYPE

    # Test bounds
    lower, upper = S.get_bounds().T
    assert np.all(lower <= y)
    assert np.all(y <= upper)

    y2 = S.validate_warped(y)
    assert close_enough(y, y2)

    x2 = S.unwarp(y)
    assert x2.shape == x.shape
    x3 = S.validate(x2)
    assert close_enough(x2, x3)

    assert close_enough(x, x2)


@given(sampled_from(WARPS), gufunc_floats("(n),(2)->(n)", allow_infinity=False, allow_nan=False))
def test_real_range_warp_unwarp(warp, args):
    x, range_ = args

    if warp == "log":
        range_ = range_[range_ > 0]
    if warp == "logit":
        range_ = range_[(0 < range_) & (range_ < 1)]

    range_ = np.sort(range_)
    assume(len(range_) == 2 and range_[0] < range_[1])

    x = np.clip(x, range_[0], range_[1])

    S = sp.Real(warp=warp, range_=range_)

    y = S.warp(x)
    assert y.shape == x.shape + (1,)
    assert y.dtype == sp.WARPED_DTYPE

    # Test bounds
    lower, upper = S.get_bounds().T
    assert np.all(lower <= y)
    assert np.all(y <= upper)

    y2 = S.validate_warped(y)
    assert close_enough(y, y2)

    x2 = S.unwarp(y)
    assert x2.shape == x.shape
    x3 = S.validate(x2)
    assert close_enough(x2, x3)

    assert close_enough(x, x2)


# Note to really stress test this we should elim min and max val, but that
# requires that we split out a diff test func for log and logit
@given(sampled_from(WARPS), gufunc_floats("(n,1),(2)->(n)", min_value=-1000, max_value=1000))
def test_real_range_unwarp_warp(warp, args):
    x_w, range_ = args

    if warp == "log":
        range_ = range_[range_ > 0]
    if warp == "logit":
        range_ = range_[(0 < range_) & (range_ < 1)]

    range_ = np.sort(range_)
    assume(len(range_) == 2 and range_[0] < range_[1])

    range_warped = sp.WARP_DICT[warp](range_)

    x_w = np.clip(x_w, range_warped[0], range_warped[1])

    S = sp.Real(warp=warp, range_=range_)

    # Test bounds
    lower, upper = S.get_bounds().T
    x_w = linear_rescale(x_w, lb0=-1000, ub0=1000, lb1=lower, ub1=upper)

    x = S.unwarp(x_w)
    assert x_w.shape == x.shape + (1,)
    assert x.dtype == range_.dtype
    assert x.dtype == S.dtype

    x2 = S.validate(x)
    assert close_enough(x, x2)

    x_w2 = S.warp(x)
    assert x_w2.shape == x_w.shape
    x_w3 = S.validate_warped(x_w2)
    assert close_enough(x_w2, x_w3)

    assert close_enough(x_w, x_w2)


@given(
    sampled_from(("linear", "bilog")),
    gufunc("(n),(m)->(n)", dtype=np.int_, elements=integers(INT_MIN, INT_MAX), unique=[False, True], min_side={"m": 2}),
)
def test_int_values_warp_unwarp(warp, args):
    x, values = args

    v = np.unique(values)  # Also sort
    assert len(v) >= 2
    f = interp1d(v, v, kind="nearest", fill_value="extrapolate")
    x = f(x).astype(values.dtype)
    assert x.ndim == 1  # make sure interp1d did not mess it up

    S = sp.Integer(warp=warp, values=values)
    y = S.warp(x)
    assert y.shape == x.shape + (1,)
    assert y.dtype == sp.WARPED_DTYPE

    # Test bounds
    lower, upper = S.get_bounds().T
    assert np.all(lower <= y)
    assert np.all(y <= upper)

    y2 = S.validate_warped(y)
    assert close_enough(y, y2)

    x2 = S.unwarp(y)
    assert x2.shape == x.shape
    x3 = S.validate(x2)
    assert close_enough(x2, x3)

    assert close_enough(x, x2)


@given(gufunc("(n),(m)->(n)", dtype=np.int_, elements=integers(1, INT_MAX), unique=[False, True], min_side={"m": 2}))
def test_log_int_values_warp_unwarp(args):
    x, values = args

    warp = "log"

    v = np.unique(values)  # Also sort
    assert len(v) >= 2
    f = interp1d(v, v, kind="nearest", fill_value="extrapolate")
    x = f(x).astype(values.dtype)
    assert x.ndim == 1  # make sure interp1d did not mess it up

    S = sp.Integer(warp=warp, values=values)
    y = S.warp(x)
    assert y.shape == x.shape + (1,)
    assert y.dtype == sp.WARPED_DTYPE

    # Test bounds
    lower, upper = S.get_bounds().T
    assert np.all(lower <= y)
    assert np.all(y <= upper)

    y2 = S.validate_warped(y)
    assert close_enough(y, y2)

    x2 = S.unwarp(y)
    assert x2.shape == x.shape
    x3 = S.validate(x2)
    assert close_enough(x2, x3)

    assert close_enough(x, x2)


@given(sampled_from(("linear", "bilog", "log")), gufunc("(n),(2)->(n)", dtype=np.int_, elements=integers(-1000, 1000)))
def test_int_range_warp_unwarp(warp, args):
    """Warning: this explicitly ignores issues with min max if going to int
    limit, since
    >>> np.array(INT_MAX).astype(np.float32).astype(np.int32)
    array(-2147483648, dtype=int32)
    Without any warning from numpy.
    """
    x, range_ = args

    # We could split out log into diff function without this pruning if we
    # start failing hypothesis health check.
    if warp == "log":
        range_ = range_[range_ > 0]

    range_ = np.sort(range_)
    assume(len(range_) == 2 and range_[0] < range_[1])

    x = np.clip(x, range_[0], range_[1]).astype(range_.dtype)

    S = sp.Integer(warp=warp, range_=range_)
    y = S.warp(x)
    assert y.shape == x.shape + (1,)
    assert y.dtype == sp.WARPED_DTYPE

    # Test bounds
    lower, upper = S.get_bounds().T
    assert np.all(lower <= y)
    assert np.all(y <= upper)

    y2 = S.validate_warped(y)
    assert close_enough(y, y2)

    x2 = S.unwarp(y)
    assert x2.shape == x.shape
    x3 = S.validate(x2)
    assert close_enough(x2, x3)

    assert x.dtype == x2.dtype
    # Close enough when evaluated as floats
    assert close_enough(x.astype("f"), x2.astype("f"))


@given(gufunc("(n)->(n)", dtype=np.bool_, elements=booleans()))
def test_bool_warp_unwarp(args):
    x, = args

    S = sp.Boolean()
    y = S.warp(x)
    assert y.shape == x.shape + (1,)
    assert y.dtype == sp.WARPED_DTYPE

    # Test bounds
    lower, upper = S.get_bounds().T
    assert np.all(lower <= y)
    assert np.all(y <= upper)

    y2 = S.validate_warped(y)
    assert close_enough(y, y2)

    x2 = S.unwarp(y)
    assert x2.shape == x.shape
    x3 = S.validate(x2)
    assert close_enough(x2, x3)

    assert close_enough(x, x2)


@given(
    gufunc(
        "(n),(m)->(n)",
        dtype=[np.int_, CAT_DTYPE],
        elements=[integers(0, INT_MAX), from_dtype(np.dtype(CAT_DTYPE))],
        unique=[False, True],
        min_side={"m": 2},
    )
)
def test_cat_warp_unwarp(args):
    x, values = args

    assert len(set(values)) >= 2

    x = values[x % len(values)]
    assert x.ndim == 1

    S = sp.Categorical(values=values)
    y = S.warp(x)
    assert y.shape == x.shape + (len(values),)
    assert y.dtype == sp.WARPED_DTYPE

    # Test bounds
    lower, upper = S.get_bounds().T
    assert np.all(lower <= y)
    assert np.all(y <= upper)

    y2 = S.validate_warped(y)
    assert close_enough(y, y2)

    x2 = S.unwarp(y)
    assert x2.shape == x.shape
    x3 = S.validate(x2)
    assert close_enough(x2, x3)

    assert close_enough(x, x2)


@given(space_configs())
def test_joint_space_unwarp_warp(args):
    meta, X, _, _ = args

    S = sp.JointSpace(meta)
    S.validate(X)

    X_w2 = S.warp(X)
    assert X_w2.dtype == sp.WARPED_DTYPE

    # Test bounds
    lower, upper = S.get_bounds().T
    assert np.all(lower <= X_w2)
    assert np.all(X_w2 <= upper)

    X2 = S.unwarp(X_w2)

    assert all(all(close_enough(X[ii][vv], X2[ii][vv]) for vv in X[ii]) for ii in range(len(X)))
    S.validate(X2)


@given(space_configs())
def test_joint_space_warp_missing(args):
    meta, X, _, fixed_vars = args

    S = sp.JointSpace(meta)

    X_w = S.warp([fixed_vars])
    assert X_w.dtype == sp.WARPED_DTYPE

    # Test bounds
    lower, upper = S.get_bounds().T
    assert np.all((lower <= X_w) | np.isnan(X_w))
    assert np.all((X_w <= upper) | np.isnan(X_w))

    for param, xx in zip(S.param_list, np.hsplit(X_w, S.blocks[:-1])):
        xx, = xx
        if param in fixed_vars:
            x_orig = S.spaces[param].unwarp(xx).item()
            S.spaces[param].validate(x_orig)
            assert close_enough(x_orig, fixed_vars[param])

            # check other direction
            x_w2 = S.spaces[param].warp(fixed_vars[param])
            assert close_enough(xx, x_w2)
        else:
            assert np.all(np.isnan(xx))


@given(space_configs())
def test_joint_space_warp_fixed_vars(args):
    meta, X, _, fixed_vars = args

    # set X vals equal to fixed_vars
    for xx in X:
        for param in fixed_vars:
            xx[param] = fixed_vars[param]

    S = sp.JointSpace(meta)
    lower, upper = S.get_bounds().T

    X_w = S.warp(X)
    assert X_w.dtype == sp.WARPED_DTYPE

    # Test bounds
    lower, upper = S.get_bounds().T
    assert np.all(lower <= X_w)
    assert np.all(X_w <= upper)

    X2 = S.unwarp(X_w, fixed_vals=fixed_vars)

    # Make sure we get == not just close in unwarp for fixed vars
    for xx in X2:
        for param in fixed_vars:
            assert xx[param] == fixed_vars[param]


@given(space_configs(), integers(min_value=0, max_value=10))
def test_joint_grid(args, max_interp):
    meta, _, _, _ = args

    type_whitelist = (bool, int, float, CAT_NATIVE_DTYPE)

    S = sp.JointSpace(meta)

    lower, upper = S.get_bounds().T

    G = S.grid(max_interp=max_interp)
    assert sorted(G.keys()) == sorted(meta.keys())

    for var, grid in G.items():
        curr_space = S.spaces[var]

        # Make sure same as calling direct
        grid2 = curr_space.grid(max_interp)
        assert grid == grid2

        if len(grid) == 0:
            assert grid == []
            assert max_interp == 0 if curr_space.values is None else len(curr_space.values) == 0
            continue

        # Make sure native type
        assert all(type(xx) in type_whitelist for xx in grid)
        tt = type(grid[0])
        assert all(type(xx) == tt for xx in grid)

        assert np.all(np.array(grid) == np.unique(grid))

        if max_interp >= 2:
            assert curr_space.lower is None or close_enough(curr_space.lower, grid[0])
            assert curr_space.upper is None or close_enough(curr_space.upper, grid[-1])

        if curr_space.values is not None:
            assert np.all(curr_space.values == grid)
        else:
            assert len(grid) <= max_interp
            # Could else, check approx linear in warped space, but good enough
            # for now.


def test_unravel_index_empty():
    assert sp.unravel_index(()) == ()


@given(lists(integers(0, 2), min_size=1))
def test_unravel_index_empty_2(dims):
    if np.prod(dims) > 0:
        dims[0] = 0
    dims = tuple(dims)

    assert sp.unravel_index(dims) == ()
