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
import scipy.stats as sst
from hypothesis import assume, given
from hypothesis.strategies import integers, lists, sampled_from
from hypothesis_gufunc.gufunc import gufunc_args
from sklearn.preprocessing import robust_scale

from bayesmark import stats
from hypothesis_util import close_enough, mfloats, probs, seeds


def t_test_(x):
    """Perform a standard t-test to test if the values in `x` are sampled from
    a distribution with a zero mean.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        array of data points to test.

    Returns
    -------
    pval : float
        p-value (in [0,1]) from t-test on `x`.
    """
    assert np.ndim(x) == 1 and (not np.any(np.isnan(x)))

    if (len(x) <= 1) or (not np.all(np.isfinite(x))):
        return 1.0  # Can't say anything about scale => p=1

    _, pval = sst.ttest_1samp(x, 0.0)
    if np.isnan(pval):
        # Should only be possible if scale underflowed to zero:
        assert np.var(x, ddof=1) <= 1e-100
        # It is debatable if the condition should be ``np.mean(x) == 0.0`` or
        # ``np.all(x == 0.0)``. Should not matter in practice.
        pval = np.float(np.mean(x) == 0.0)
    assert 0.0 <= pval and pval <= 1.0
    return pval


@given(gufunc_args("(n),()->(n)", dtype=np.float_, elements=[mfloats(), probs()], min_side=2))
def test_robust_standardize_to_sklearn(args):
    X, q_level = args

    q0, q1 = 0.5 * (1.0 - q_level), 0.5 * (1.0 + q_level)
    assert close_enough(q1 - q0, q_level)

    X_bo = stats.robust_standardize(X, q_level=q_level)

    X = X[:, None]
    X_skl = robust_scale(X, axis=0, with_centering=True, with_scaling=True, quantile_range=[100.0 * q0, 100.0 * q1])
    X_skl = X_skl[:, 0] * (sst.norm.ppf(q1) - sst.norm.ppf(q0))

    assert close_enough(X_bo, X_skl, equal_nan=True)


def test_robust_standardize_broadcast():
    """Need to do things different here since standardize broadcasts over the
    wrong dimension (0 instead of -1).
    """
    # Build vectorize version, this is just loop inside.
    f_vec = np.vectorize(stats.robust_standardize, signature="(n),()->(n)", otypes=["float64"])

    @given(gufunc_args("(n,m),()->(n,m)", dtype=np.float_, min_side={"n": 2}, elements=[mfloats(), probs()]))
    def test_f(args):
        X, q_level = args

        R1 = stats.robust_standardize(X, q_level)
        R2 = f_vec(X.T, q_level).T
        assert R1.dtype == "float64"
        assert R2.dtype == "float64"
        assert close_enough(R1, R2, equal_nan=True)

    # Call the test
    test_f()


@given(integers(0, 10), mfloats(), probs())
def test_t_EB_zero_var(N, val, alpha):
    x = val + np.zeros(N)
    EB = stats.t_EB(x, alpha=alpha)
    if N <= 1:
        assert EB == np.inf
    else:
        assert np.allclose(EB, 0.0)


@given(integers(1, 10), sampled_from([np.inf, -np.inf]), probs())
def test_t_EB_inf(N, val, alpha):
    x = np.zeros(N)
    x[0] = val

    EB = stats.t_EB(x, alpha=alpha)
    if N <= 1:
        assert EB == np.inf
    else:
        assert np.isnan(EB)


@given(seeds(), probs(), integers(2, 10))
def test_t_EB_coverage(seed, alpha, N):
    trials = 100

    random_st = np.random.RandomState(seed)

    fail = 0
    for tt in range(trials):
        x = random_st.randn(N)

        EB = stats.t_EB(x, alpha=alpha)
        mu = np.nanmean(x)
        LB, UB = mu - EB, mu + EB
        assert np.isfinite(LB) and np.isfinite(UB)
        fail += (0.0 < LB) or (UB < 0.0)
    pval = sst.binom_test(fail, trials, alpha)

    assert pval >= 0.05 / 100  # Assume we run 100 times


@given(lists(mfloats(), min_size=2))
def test_t_test_to_EB(x):
    pval = t_test_(x)
    assume(0.0 < pval and pval < 1.0)

    EB = stats.t_EB(x, alpha=pval)
    assert np.allclose(np.abs(np.mean(x)), EB)
