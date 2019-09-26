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
"""Compute quantiles and confidence intervals.
"""
import numpy as np
import scipy.stats as ss

from bayesmark.np_util import isclose_lte


def ensure_shape(x, y):
    """Util to broadcast on var to another but only when shape is different.

    This way we don't convert scalar into array type unnecessarily.
    """
    shape_y = np.shape(y)
    if np.shape(x) == shape_y:
        return x
    return np.broadcast_to(x, shape_y)


def order_stats(X):
    """Compute order statistics on sample `X`.

    Follows convention that order statistic 1 is minimum and statistic n is maximum. Therefore, array elements ``0``
    and ``n+1`` are ``-inf`` and ``+inf``.

    Parameters
    ----------
    X : :class:`numpy:numpy.ndarray` of shape (n,)
        Data for order statistics. Can be vectorized. Must be sortable data type (which is almost everything).

    Returns
    -------
    o_stats : :class:`numpy:numpy.ndarray` of shape (n+2,)
        Order statistics on `X`.
    """
    assert np.ndim(X) >= 1
    # NaN is not allowed since it does not have well defined order.
    assert not np.any(np.isnan(X))

    X_shape = np.shape(X)
    inf_pad = np.full(X_shape[:-1] + (1,), np.inf)

    o_stats = np.concatenate((-inf_pad, np.sort(X, axis=-1), inf_pad), axis=-1)
    return o_stats


def _quantile(n, q):
    idx = np.ceil(n * q).astype(int)
    return idx


def quantile(X, q):
    """Computes `q` th quantile of `X`.

    Similar to :func:`numpy:numpy.percentile` except that it matches the mathematical definition of a quantile *and*
    `q` is scaled in (0,1) rather than (0,100).

    Parameters
    ----------
    X : :class:`numpy:numpy.ndarray` of shape (n,)
        Data for quantile estimation. Can be vectorized. Must be sortable data type (which is almost everything).
    q : float
        Quantile to compute, must be in (0, 1). Can be vectorized.

    Returns
    -------
    estimate : dtype of `X`, scalar
        Empirical `q` quantile from sample `X`.
    """
    assert np.ndim(X) >= 1
    # We could robustify things to allow the edge cases, but maybe later
    assert np.all(0 < q) and np.all(q < 1)
    # Currently don't support broadcasting both at same time
    assert np.ndim(X) == 1 or np.ndim(q) == 0

    n = X.shape[-1]
    idx = _quantile(n, q)

    o_stats = order_stats(X)
    estimate = o_stats[..., idx]
    return estimate


def _quantile_CI(n, q, alpha):
    # Use in case there is -inf case from being at extreme of distn
    idx_lower = np.fmax(0, ss.binom.ppf(alpha / 2.0, n, q)).astype(int)
    assert np.all(isclose_lte(ss.binom.cdf(idx_lower - 1, n, q), alpha / 2.0))
    assert np.all(isclose_lte(alpha / 2.0, ss.binom.cdf(idx_lower, n, q)))
    assert np.all(0 <= idx_lower) and np.all(idx_lower <= n + 1)

    idx_upper = np.fmax(0, ss.binom.isf(alpha / 2.0, n, q)).astype(int) + 1
    assert np.all(isclose_lte(ss.binom.sf(idx_upper - 1, n, q), alpha / 2.0))
    assert np.all(isclose_lte(alpha / 2.0, ss.binom.sf(idx_upper - 2, n, q)))
    assert np.all(isclose_lte(1 - (alpha / 2.0), ss.binom.cdf(idx_upper - 1, n, q)))
    assert np.all(isclose_lte(ss.binom.cdf(idx_upper - 2, n, q), 1 - (alpha / 2.0)))
    assert np.all(0 <= idx_upper) and np.all(idx_upper <= n + 1)

    C = ss.binom.cdf(idx_upper - 1, n, q) - ss.binom.cdf(idx_lower - 1, n, q)
    assert np.all(isclose_lte(1.0 - alpha, C))

    return idx_lower, idx_upper


def quantile_CI(X, q, alpha=0.05):
    """Calculate CI on `q` quantile from same `X` using nonparametric estimation from order statistics.

    This will have alpha level of at most `alpha` due to the discrete nature of order statistics.

    Parameters
    ----------
    X : :class:`numpy:numpy.ndarray` of shape (n,)
        Data for quantile estimation. Can be vectorized. Must be sortable data type (which is almost everything).
    q : float
        Quantile to compute, must be in (0, 1). Can be vectorized.
    alpha : float
        False positive rate we allow for CI, must be in (0, 1). Can be vectorized.

    Returns
    -------
    LB : dtype of `X`, scalar
        Lower end on CI
    UB : dtype of `X`, scalar
        Upper end on CI
    """
    assert np.ndim(X) >= 1
    # We could robustify things to allow the edge cases, but maybe later
    assert np.all(0 < q) and np.all(q < 1)
    assert np.all(0 < alpha) and np.all(alpha < 1)
    # Currently don't support broadcasting both at same time
    assert np.ndim(X) == 1 or (np.ndim(q) == 0 and np.ndim(alpha) == 0)

    n = X.shape[-1]
    idx_lower, idx_upper = _quantile_CI(n, q, alpha)

    o_stats = order_stats(X)
    LB, UB = o_stats[..., idx_lower], o_stats[..., idx_upper]
    return LB, UB


def max_quantile_CI(X, q, m, alpha=0.05):
    """Calculate CI on `q` quantile of distribution on max of `m` iid samples using a data set `X`.

    This uses nonparametric estimation from order statistics and will have alpha level of at most `alpha` due to the
    discrete nature of order statistics.

    Parameters
    ----------
    X : :class:`numpy:numpy.ndarray` of shape (n,)
        Data for quantile estimation. Can be vectorized. Must be sortable data type (which is almost everything).
    q : float
        Quantile to compute, must be in (0, 1). Can be vectorized.
    m : int
        Compute statistics for distribution on max over `m` samples. Must be ``>= 1``. Can be vectorized.
    alpha : float
        False positive rate we allow for CI, must be in (0, 1). Can be vectorized.

    Returns
    -------
    estimate : dtype of `X`, scalar
        Best estimate on `q` quantile on max over `m` iid samples.
    LB : dtype of `X`, scalar
        Lower end on CI
    UB : dtype of `X`, scalar
        Upper end on CI
    """
    # X and alpha used/checked below in quantile_CI routine.
    # We could robustify things to allow the edge cases, but maybe later
    assert np.all(0 < q) and np.all(q < 1)
    # Could check int but if someone wants to interpolate, we will let them.
    assert np.all(m >= 1)
    # Currently don't support broadcasting both at same time
    assert np.ndim(X) == 1 or (np.ndim(q) == 0 and np.ndim(q) == 0 and np.ndim(alpha) == 0)

    q = q ** (1.0 / m)
    o_stats = order_stats(X)

    n = X.shape[-1]
    idx = _quantile(n, q)
    idx_lower, idx_upper = _quantile_CI(n, q, alpha=alpha)

    LB, UB = o_stats[..., idx_lower], o_stats[..., idx_upper]
    # Might need to broadcast estimate out if vectorization is in alpha
    estimate = ensure_shape(o_stats[..., idx], LB)
    return estimate, LB, UB


def min_quantile_CI(X, q, m, alpha=0.05):
    """Calculate confidence interval on `q` quantile of distribution on min of `m` iid samples using a data set `X`.

    This uses nonparametric estimation from order statistics and will have alpha level of at most `alpha` due to the
    discrete nature of order statistics.

    Parameters
    ----------
    X : :class:`numpy:numpy.ndarray` of shape (n,)
        Data for quantile estimation. Can be vectorized. Must be sortable data type (which is almost everything).
    q : float
        Quantile to compute, must be in (0, 1). Can be vectorized.
    m : int
        Compute statistics for distribution on min over `m` samples. Must be ``>= 1``. Can be vectorized.
    alpha : float
        False positive rate we allow for CI, must be in (0, 1). Can be vectorized.

    Returns
    -------
    estimate : dtype of `X`, scalar
        Best estimate on `q` quantile on min over `m` iid samples.
    LB : dtype of `X`, scalar
        Lower end on CI
    UB : dtype of `X`, scalar
        Upper end on CI
    """
    # X and alpha used/checked below in quantile_CI routine.
    # We could robustify things to allow the edge cases, but maybe later
    assert np.all(0 < q) and np.all(q < 1)
    # Could check int but if someone wants to interp, we will let them.
    assert np.all(m >= 1)
    # Currently don't support broadcasting both at same time
    assert np.ndim(X) == 1 or (np.ndim(q) == 0 and np.ndim(q) == 0 and np.ndim(alpha) == 0)

    # This might have numerics issues for small q
    q = 1.0 - (1.0 - q) ** (1.0 / m)
    o_stats = order_stats(X)

    n = X.shape[-1]
    idx = _quantile(n, q)
    idx_lower, idx_upper = _quantile_CI(n, q, alpha=alpha)

    LB, UB = o_stats[..., idx_lower], o_stats[..., idx_upper]
    # Might need to broadcast estimate out if vectorization is in alpha
    estimate = ensure_shape(o_stats[..., idx], LB)
    return estimate, LB, UB


def quantile_and_CI(X, q, alpha=0.05):
    """Calculate CI on `q` quantile from same `X` using nonparametric estimation from order statistics.

    This will have alpha level of at most `alpha` due to the discrete nature of order statistics.

    Parameters
    ----------
    X : :class:`numpy:numpy.ndarray` of shape (n,)
        Data for quantile estimation. Can be vectorized. Must be sortable data type (which is almost everything).
    q : float
        Quantile to compute, must be in (0, 1). Can be vectorized.
    alpha : float
        False positive rate we allow for CI, must be in (0, 1). Can be vectorized.

    Returns
    -------
    estimate : dtype of `X`, scalar
        Empirical `q` quantile from sample `X`.
    LB : dtype of `X`, scalar
        Lower end on CI
    UB : dtype of `X`, scalar
        Upper end on CI
    """
    # This routine is mostly just a wrapper routine
    estimate, LB, UB = max_quantile_CI(X, q=q, m=1, alpha=alpha)
    return estimate, LB, UB
