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
from hypothesis.strategies import floats, integers, lists

from bo_benchmark import np_util
from hypothesis_util import broadcast_tester, close_enough, gufunc_floats, seeds


@given(seeds())
def test_random_seed(seed):
    random = np.random.RandomState(seed)
    seed = np_util.random_seed(random)


@given(lists(lists(floats())), seeds())
def test_shuffle_2d(X, seed):
    random = np.random.RandomState(seed)
    np_util.shuffle_2d(X, random)


@given(gufunc_floats("(n,m)->()"), integers(1, 5), seeds())
def test_strat_split(X, n_splits, seed):
    X, = X

    random = np.random.RandomState(seed)
    np_util.strat_split(X, n_splits, inplace=False, random=random)

    random = np.random.RandomState(seed)
    np_util.strat_split(X, n_splits, inplace=True, random=random)


@given(gufunc_floats("(),()->()", allow_nan=False))
def test_isclose_lte_pass(args):
    x, y = args
    x = np.minimum(x, y + 1e-10)
    assert np_util.isclose_lte(x, y)


@given(gufunc_floats("(),()->()", allow_nan=False))
def test_isclose_lte_fail(args):
    x, y = args
    fac = 1e20
    if np.ndim(x) == 0:
        y = np.nan_to_num(y)
        x = y + fac * np.spacing(np.abs(y)) + 1
    else:
        y.flat[0] = np.nan_to_num(y.flat[0])
        x.flat[0] = y.flat[0] + fac * np.spacing(np.abs(y.flat[0])) + 1
    assert not np_util.isclose_lte(x, y)


def test_isclose_broadcast():
    broadcast_tester(np_util.isclose_lte, "(),()->()", otype="bool", min_value=-1000, max_value=1000)


@given(gufunc_floats("(),(),()->()", allow_nan=False))
def test_clip_chk_pass(args):
    x, lb, ub = args

    assume(lb <= ub)

    x = np.clip(x, lb - 1e-10, ub + 1e-10)
    x_clip = np_util.clip_chk(x, lb=lb, ub=ub)
    assert np.all(x_clip == np.clip(x_clip, lb, ub))


@given(gufunc_floats("(),(),()->()", allow_nan=True))
def test_clip_chk_pass_nan(args):
    x, lb, ub = args

    assume(lb <= ub)

    x = np.clip(x, lb - 1e-10, ub + 1e-10)
    x_clip = np_util.clip_chk(x, lb=lb, ub=ub, allow_nan=True)
    assert np.all((np.isnan(x) & np.isnan(x_clip)) | (x_clip == np.clip(x_clip, lb, ub)))


@given(gufunc_floats("(n),()->()", allow_nan=False))
def test_snap_to_pass(args):
    x, val = args

    x = np.clip(x, val - 1e-10, val + 1e-10)
    x_snap = np_util.snap_to(x, val)
    assert np.all(x_snap == val)


@given(gufunc_floats("(),(),(),()->()", min_value=-1000, max_value=1000))
def test_linear_rescale_bounds(args):
    lb0, ub0, lb1, ub1 = args

    # Use sorted because hypothesis doesn't like using assume too often
    lb0, ub0 = sorted([lb0, ub0])
    lb1, ub1 = sorted([lb1, ub1])

    assume(lb0 < ub0)
    assume(lb1 <= ub1)

    lb1_ = np_util.linear_rescale(lb0, lb0, ub0, lb1, ub1)
    assert close_enough(lb1, lb1_)

    ub1_ = np_util.linear_rescale(ub0, lb0, ub0, lb1, ub1)
    assert close_enough(ub1, ub1_)


@given(gufunc_floats("(),(),(),(),()->()", min_value=-1000, max_value=1000))
def test_linear_rescale_inner(args):
    X, lb0, ub0, lb1, ub1 = args

    # Use sorted because hypothesis doesn't like using assume too often
    lb0, ub0 = sorted([lb0, ub0])
    lb1, ub1 = sorted([lb1, ub1])

    assume(lb0 < ub0)
    assume(lb1 <= ub1)

    X = np.clip(X, lb0, ub0)

    X = np_util.linear_rescale(X, lb0, ub0, lb1, ub1)

    assert np.all(X <= ub1)
    assert np.all(lb1 <= X)


@given(gufunc_floats("(),(),(),(),(),()->()", min_value=-1000, max_value=1000))
def test_linear_rescale_inverse(args):
    X, lb0, ub0, lb1, ub1, enforce_bounds = args
    enforce_bounds = enforce_bounds >= 0

    # Use sorted because hypothesis doesn't like using assume too often
    lb0, ub0 = sorted([lb0, ub0])
    lb1, ub1 = sorted([lb1, ub1])

    assume(lb0 < ub0)
    assume(lb1 < ub1)
    # Can't expect numerics to work well in these extreme cases:
    assume((ub0 - lb0) < 1e3 * (ub1 - lb1))

    if enforce_bounds:
        X = np.clip(X, lb0, ub0)

    X_ = np_util.linear_rescale(X, lb0, ub0, lb1, ub1, enforce_bounds=enforce_bounds)
    X_ = np_util.linear_rescale(X_, lb1, ub1, lb0, ub0, enforce_bounds=enforce_bounds)

    assert close_enough(X_, X)


@given(gufunc_floats("(),(),(),(),()->()", min_value=-1000, max_value=1000))
def test_linear_rescale_bound_modes(args):
    X, lb0, ub0, lb1, ub1 = args

    # Use sorted because hypothesis doesn't like using assume too often
    lb0, ub0 = sorted([lb0, ub0])
    lb1, ub1 = sorted([lb1, ub1])

    assume(lb0 < ub0)
    assume(lb1 <= ub1)

    X = np.clip(X, lb0, ub0)

    Y1 = np_util.linear_rescale(X, lb0, ub0, lb1, ub1, enforce_bounds=False)
    Y2 = np_util.linear_rescale(X, lb0, ub0, lb1, ub1, enforce_bounds=True)

    assert close_enough(Y1, Y2)


def pair_sort(X, Y):
    X, Y = np.broadcast_arrays(X, Y)
    Z = [X, Y]
    Z = np.sort(Z, axis=0)
    X, Y = Z
    return X, Y


def test_linear_rescale_broadcast():
    def clean_up(args):
        X, lb0, ub0, lb1, ub1, enforce_bounds = args

        enforce_bounds = enforce_bounds >= 0

        # Ideally, hypothesis should be able to handle constraints like this
        lb0, ub0 = pair_sort(lb0, ub0)
        lb1, ub1 = pair_sort(lb1, ub1)

        assume(np.all(lb0 < ub0))
        assume(np.all(lb1 <= ub1))

        if enforce_bounds:
            X = np.clip(X, lb0, ub0)

        return X, lb0, ub0, lb1, ub1, enforce_bounds

    broadcast_tester(
        np_util.linear_rescale,
        "(),(),(),(),(),()->()",
        "float64",
        excluded=(5,),
        map_=clean_up,
        min_value=-1000,
        max_value=1000,
    )


@given(floats(), floats())
def test_isclose(x, y):
    """Test numpy version new enough to avoid broadcasting bug in `np.isclose`.

    See numpy issue 'inconsistency in np.isclose #7014'. We could bump up numpy requirement version and
    eliminate this wrapper.

    See:
    https://github.com/numpy/numpy/issues/7014
    """
    z = np.isclose(x, y)
    assert type(z) == np.bool_
    assert z == np.squeeze(z)
    assert np.isscalar(z)
    assert np.shape(z) == ()

    z = np.isclose(np.asarray(x), y)
    assert type(z) == np.bool_
    assert z == np.squeeze(z)
    assert np.isscalar(z)
    assert np.shape(z) == ()

    z = np.isclose(x, np.asarray(y))
    assert type(z) == np.bool_
    assert z == np.squeeze(z)
    assert np.isscalar(z)
    assert np.shape(z) == ()

    z = np.isclose(np.asarray(x), np.asarray(y))
    assert type(z) == np.bool_
    assert z == np.squeeze(z)
    assert np.isscalar(z)
    assert np.shape(z) == ()


def test_isclose_2():
    """Make sure we are running numpy version where numpy issue 'inconsistency in np.isclose #7014' has been fixed.
    """
    y = np.isclose(0, 1)

    assert np.ndim(y) == 0
    assert np.isscalar(y)
