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
"""General statistic tools useful in the benchmark.
"""
import numpy as np
import scipy.stats as sst


def robust_standardize(X, q_level=0.5):
    """Perform robust standardization of data matrix `X` over axis 0.

    Similar to :func:`sklearn:sklearn.preprocessing.robust_scale` except also does a Gaussian
    adjustment rescaling so that if Gaussian data is passed in the transformed
    data will, in large `n`, be distributed as N(0,1). See sklearn feature
    request #10139 on github.

    Parameters
    ----------
    X : :class:`numpy:numpy.ndarray` of shape (n, ...)
        Array containing elements standardize. Require ``n >= 2``.
    q_level : scalar
        Must be in [0, 1]. Inter-quartile range to use for scale estimation.

    Returns
    -------
    X : :class:`numpy:numpy.ndarray` of shape (n, ...)
        Elements of input `X` standardization.
    """
    X = np.asarray(X)
    assert X.ndim in (1, 2)
    assert np.all(np.isfinite(X))
    assert 0.0 < q_level and q_level <= 1.0
    assert X.shape[0] >= 2

    mu = np.median(X, axis=0)

    q0, q1 = 0.5 * (1.0 - q_level), 0.5 * (1.0 + q_level)
    v = np.percentile(X, 100 * q1, axis=0) - np.percentile(X, 100 * q0, axis=0)
    v = np.asarray(v)
    v[v == 0.0] = 1.0

    X_ss = (X - mu) / v
    # Rescale to match scale of N(0,1)
    X_ss = X_ss * (sst.norm.ppf(q1) - sst.norm.ppf(q0))
    assert X.shape == X_ss.shape
    return X_ss


def t_EB(x, alpha=0.05, axis=-1):
    """Get t-statistic based error bars on mean of `x`.

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray` of shape (n_samples,)
        Data points to estimate mean. Must not be empty or contain ``NaN``.
    alpha : float
        The alpha level (``1-confidence``) probability (in (0, 1)) to construct confidence interval from t-statistic.
    axis : int
        The axis on `x` where we compute the t-statistics. The function is vectorized over all other dimensions.

    Returns
    -------
    EB : float
        Size of error bar on mean (``>= 0``). The confidence interval is ``[mean(x) - EB, mean(x) + EB]``. `EB` is
        ``inf`` when ``len(x) <= 1``. Will be ``NaN`` if there are any infinite values in `x`.
    """
    assert np.ndim(x) >= 1 and (not np.any(np.isnan(x)))
    assert np.ndim(alpha) == 0
    assert 0.0 < alpha and alpha < 1.0

    N = np.shape(x)[axis]
    if N <= 1:
        return np.full(np.sum(x, axis=axis).shape, fill_value=np.inf)

    confidence = 1 - alpha
    # loc cancels out when we just want EB anyway
    LB, UB = sst.t.interval(confidence, N - 1, loc=0.0, scale=1.0)
    assert not (LB > UB)
    # Just multiplying scale=ss.sem(x) is better for when scale=0
    EB = 0.5 * sst.sem(x, axis=axis) * (UB - LB)
    return EB
