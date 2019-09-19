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
"""Compute expected maximum or minimum from iid samples.
"""
import numpy as np
from scipy.special import gammaln, logsumexp


def get_expected_max_weights(n, m):
    """Get the L-estimator weights for computing unbiased estimator of expected ``max(x[1:m])`` on a data set.

    Parameters
    ----------
    n : int
        Number of data points in data set ``len(x)``. Must be ``>= 1``.
    m : `int` or :class:`numpy:numpy.ndarray` with dtype `int`
        This function is for estimating the expected maximum over `m` iid draws. Require ``m >= 1``. This can be
        broadcasted. If ``m > n``, the weights will be nan, because there is no way to get unbiased estimate in that
        case.

    Returns
    -------
    pdf : :class:`numpy:numpy.ndarray`, shape (n,)
        The weights for L-estimator. Will be positive and sum to one.
    """
    assert np.ndim(n) == 0
    assert n >= 1  # otherwise makes no sense

    m = np.asarray(m)  # Must be np type for broadcasting
    # We could also check dtype is int, but not bothering here
    assert np.all(m >= 1)  # otherwise makes no sense
    m = m[..., None]

    kk = 1 + np.arange(n)
    lpdf = gammaln(kk) - gammaln(kk - (m - 1))
    pdf = np.exp(lpdf - logsumexp(lpdf, axis=-1, keepdims=True))
    # expect nan for m > n
    assert np.all((m > n) | np.isclose(np.sum(pdf, axis=-1, keepdims=True), 1.0))
    return pdf


def expected_max(x, m):
    """Compute unbiased estimator of expected ``max(x[1:m])`` on a data set.

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray` of shape (n,)
        Data set we would like expected ``max(x[1:m])`` on.
    m : `int` or :class:`numpy:numpy.ndarray` with dtype `int`
        This function is for estimating the expected maximum over `m` iid draws. Require ``m >= 1``. This can be
        broadcasted. If ``m > n``, the weights will be nan, because there is no way to get unbiased estimate in that
        case.

    Returns
    -------
    E_max_x : float
        Unbiased estimate of mean max of `m` draws from distribution on `x`.
    """
    assert np.ndim(x) == 1
    # m is validated by get_expected_max_weights

    # Get order stats for L-estimator
    x = np.array(x, copy=True)  # we will modify in place
    x.sort()  # in place!!

    # Now get estimator weights
    n, = x.shape
    if n == 0:
        return np.full(np.shape(m), np.nan)
    pdf = get_expected_max_weights(n, m)

    # Compute L-estimator
    E_max_x = np.sum(x * pdf, axis=-1)
    return E_max_x


def expected_min(x, m):
    """Compute unbiased estimator of expected ``min(x[1:m])`` on a data set.

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray` of shape (n,)
        Data set we would like expected ``min(x[1:m])`` on. Require ``len(x) >= 1``.
    m : `int` or :class:`numpy:numpy.ndarray` with dtype `int`
        This function is for estimating the expected minimum over `m` iid draws. Require ``m >= 1``. This can be
        broadcasted. If ``m > n``, the weights will be nan, because there is no way to get unbiased estimate in that
        case.

    Returns
    -------
    E_min_x : float
        Unbiased estimate of mean min of `m` draws from distribution on `x`.
    """
    x = np.asarray(x)
    E_min_x = -expected_max(-x, m)
    return E_min_x
