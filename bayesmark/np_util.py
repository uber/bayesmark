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
"""Utilities to that could be included in `numpy` but aren't.
"""
import numpy as np

# np seed must be in [0, 2**32 - 1] = [0, uint32 max]
SEED_MAX_INCL = np.iinfo(np.uint32).max

# Access default numpy rng in way that is short and sphinx friendly
random = np.random.random.__self__


def random_seed(random=random):
    """Draw a random seed compatible with :class:`numpy:numpy.random.RandomState`.

    Parameters
    ----------
    random : :class:`numpy:numpy.random.RandomState`
        Random stream to use to draw the random seed.

    Returns
    -------
    seed : int
        Seed for a new random stream in ``[0, 2**32-1)``.
    """
    # np randint is exclusive on the high value, py randint is inclusive. We
    # must use inclusive limit here to work with both. We are missing one
    # possibility here (2**32-1), but I don't think that matters.
    seed = random.randint(0, SEED_MAX_INCL)
    return seed


def shuffle_2d(X, random=random):
    """Generalization of :func:`numpy:numpy.random.shuffle` of 2D array.

    Performs in-place shuffling of `X`. So, it has no return value.

    Parameters
    ----------
    X : :class:`numpy:numpy.ndarray` of shape (n, m)
        Array-like 2D data to shuffle in place. Shuffles order of rows and order of elements within a row.
    random : :class:`numpy:numpy.random.RandomState`
        Random stream to use to draw the random seed.
    """
    random.shuffle(X)
    for rr in X:
        random.shuffle(rr)


def strat_split(X, n_splits, inplace=False, random=random):
    """Make a stratified random split of items.

    Parameters
    ----------
    X : :class:`numpy:numpy.ndarray` of shape (n, m)
        Data we would like to split randomly into groups. We should get the same number +/-1 of elements from each row
        in each group.
    n_splits : int
        How many groups we want to split into.
    inplace : bool
        If true, this function will cause in place modifications to `X`.
    random : :class:`numpy:numpy.random.RandomState`
        Random stream to use for reproducibility.

    Returns
    -------
    Y : list(:class:`numpy:numpy.ndarray`)
        Stratified split of `X` where each row of `Y` contains the same number +/-1 of elements from each row of `X`.
        Must be a list of arrays since each row may have a different length.
    """
    # Arguably, this function could go in stats
    assert np.ndim(X) == 2
    assert n_splits > 0

    if not inplace:
        X = np.array(X, copy=True)

    shuffle_2d(X, random=random)
    # Note this is like X.T.ravel()
    Y = np.array_split(np.ravel(X, order="F"), n_splits)

    # Just for good measure make sure this is shuffled too, prob not needed.
    shuffle_2d(Y, random=random)
    return Y


def isclose_lte(x, y):
    """Check that less than or equal to (lte, ``x <= y``) is approximately true between all elements of `x` and `y`.

    This is similar to :func:`numpy:numpy.allclose` for equality. Shapes of all input variables must be broadcast
    compatible.

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray`
        Lower limit in ``<=`` check.
    y : :class:`numpy:numpy.ndarray`
        Upper limit in ``<=`` check.

    Returns
    -------
    lte : bool
        True if ``x <= y`` is approximately true element-wise.
    """
    # Use np.less_equal to ensure always np type consistently
    lte = np.less_equal(x, y) | np.isclose(x, y)
    return lte


def clip_chk(x, lb, ub, allow_nan=False):
    """Clip all element of `x` to be between `lb` and `ub` like :func:`numpy:numpy.clip`, but also check
    :func:`numpy:numpy.isclose`.

    Shapes of all input variables must be broadcast compatible.

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray`
        Array containing elements to clip.
    lb : :class:`numpy:numpy.ndarray`
        Lower limit in clip.
    ub : :class:`numpy:numpy.ndarray`
        Upper limit in clip.
    allow_nan : bool
        If true, we allow ``nan`` to be present in `x` without out raising an error.

    Returns
    -------
    x : :class:`numpy:numpy.ndarray`
        An array with the elements of `x`, but where values < `lb` are replaced with `lb`, and those > `ub` with `ub`.
    """
    assert np.all(lb <= ub)  # np.clip does not do this check

    x = np.asarray(x)

    # These are asserts not exceptions since clip_chk most used internally.
    if allow_nan:
        assert np.all(isclose_lte(lb, x) | np.isnan(x))
        assert np.all(isclose_lte(x, ub) | np.isnan(x))
    else:
        assert np.all(isclose_lte(lb, x))
        assert np.all(isclose_lte(x, ub))
    x = np.clip(x, lb, ub)
    return x


def snap_to(x, fixed_val=None):
    """Snap input `x` to the `fixed_val` unless `fixed_val` is `None`, where `x` is returned.

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray`
        Array containing elements to snap.
    fixed_val : :class:`numpy:numpy.ndarray` or None
        Values to be returned if `x` is close, otherwise an error is raised. If `fixed_val` is `None`, `x` is returned.

    Returns
    -------
    fixed_val : :class:`numpy:numpy.ndarray`
        Snapped to value of `x`.
    """
    if fixed_val is None:
        return x

    # Include == for discrete types where allclose doesn't work
    if not (np.all(x == fixed_val) or np.allclose(x, fixed_val)):
        raise ValueError("Expected fixed value %s, got %s." % (repr(fixed_val), repr(x)))

    assert np.all(x == fixed_val) or np.allclose(x, fixed_val)
    fixed_val = np.broadcast_to(fixed_val, np.shape(x))
    return fixed_val


def linear_rescale(X, lb0, ub0, lb1, ub1, enforce_bounds=True):
    """Linearly transform all elements of `X`, bounded between `lb0` and `ub0`, to be between `lb1` and `ub1`.

    Shapes of all input variables must be broadcast compatible.

    Parameters
    ----------
    X : :class:`numpy:numpy.ndarray`
        Array containing elements to rescale.
    lb0 : :class:`numpy:numpy.ndarray`
        Current lower bound of `X`.
    ub0 : :class:`numpy:numpy.ndarray`
        Current upper bound of `X`.
    lb1 : :class:`numpy:numpy.ndarray`
        Desired lower bound of `X`.
    ub1 : :class:`numpy:numpy.ndarray`
        Desired upper bound of `X`.
    enforce_bounds : bool
        If True, perform input bounds check (and clipping if slight violation) on the input `X` and again on the
        output. This argument is not meant to be vectorized like the other input variables.

    Returns
    -------
    X : :class:`numpy:numpy.ndarray`
        Elements of input `X` after linear rescaling.
    """
    assert np.all(np.isfinite(lb0))
    assert np.all(np.isfinite(lb1))
    assert np.all(np.isfinite(ub0))
    assert np.all(np.isfinite(ub1))
    assert np.all(lb0 < ub0)
    assert np.all(lb1 <= ub1)

    m = np.true_divide(ub1 - lb1, ub0 - lb0)
    assert np.all(m >= 0)

    if enforce_bounds:
        X = clip_chk(X, lb0, ub0)  # This will flag any non-finite X input.
        X = clip_chk(m * (X - lb0) + lb1, lb1, ub1)
    else:
        X = m * (X - lb0) + lb1
    return X


def argmin_2d(X):
    """Take the arg minimum of a 2D array."""
    assert X.size > 0, "argmin of empty array not defined"

    ii, jj = np.unravel_index(X.argmin(), X.shape)
    return ii, jj


def cummin(x_val, x_key):
    """Get the cumulative minimum of `x_val` when ranked according to `x_key`.

    Parameters
    ----------
    x_val : :class:`numpy:numpy.ndarray` of shape (n, d)
        The array to get the cumulative minimum of along axis 0.
    x_key : :class:`numpy:numpy.ndarray` of shape (n, d)
        The array for ranking elements as to what is the minimum.

    Returns
    -------
    c_min : :class:`numpy:numpy.ndarray` of shape (n, d)
        The cumulative minimum array.
    """
    assert x_val.shape == x_key.shape
    assert x_val.ndim == 2
    assert not np.any(np.isnan(x_key)), "cummin not defined for nan key"

    n, _ = x_val.shape

    xm = np.minimum.accumulate(x_key, axis=0)
    idx = np.maximum.accumulate((x_key <= xm) * np.arange(n)[:, None])
    c_min = np.take_along_axis(x_val, idx, axis=0)
    return c_min
