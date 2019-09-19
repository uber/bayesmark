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
import scipy.stats as ss
from hypothesis import assume, given
from hypothesis.strategies import floats, integers
from hypothesis_gufunc.gufunc import gufunc_args as gufunc

import bo_benchmark.quantiles as qt
from hypothesis_util import broadcast_tester, gufunc_floats, multi_broadcast_tester

# We could use nextafter to get closer to limits, but still creates numerics
# issues.
ABOVE0 = 1e-6
BELOW1 = 1 - 1e-6
GLOBAL_FPR = 0.05

# Will need to generalize these out to limits before upstreaming:
floats_ = floats(allow_infinity=False, allow_nan=False)
counts = integers(1, 1000)
probs = floats(ABOVE0, BELOW1)


def order_stats_trim(X):
    Y = qt.order_stats(X)

    X_ss = np.array(X.shape)
    X_ss[-1] = X_ss[-1] + 2
    assert Y.shape == tuple(X_ss)
    assert np.all(Y[..., 0] == -np.inf)
    assert np.all(Y[..., -1] == np.inf)

    Y = Y[..., 1:-1]  # Trim out infs
    assert Y.shape == tuple(X.shape)
    return Y


@given(gufunc_floats("(n)->(m)", allow_nan=False))
def test_order_stats(args):
    X, = args

    o_stats = qt.order_stats(X)

    assert len(o_stats) == len(X) + 2

    # test is sorted
    assert not np.any(np.diff(o_stats) < 0)

    # limit elements
    assert o_stats[0] == -np.inf
    assert o_stats[-1] == np.inf

    # equal to equiv versions with lists
    assert [-np.inf] + sorted(X) + [np.inf] == list(o_stats)


@given(gufunc_floats("(n)->()", allow_nan=False))
def test_quantile(args):
    X, = args

    ll = qt.quantile(X, np.nextafter(0, 1))
    assert ll == -np.inf if len(X) == 0 else ll == np.min(X)

    uu = qt.quantile(X, np.nextafter(1, 0))
    assert uu == -np.inf if len(X) == 0 else uu == np.max(X)

    if len(X) % 2 == 1:
        mm = qt.quantile(X, 0.5)
        assert mm == np.median(X)


@given(gufunc("(n),()->()", dtype=np.float_, elements=[floats_, probs]))
def test_quantile_to_np(args):
    X, q = args

    estimate = qt.quantile(X, q)

    # Correct the off-by-1 error in numpy percentile. This might still have
    # issues due to round off error by multiplying by 100 since powers of 10
    # are not very fp friendly.
    estimate_np = np.percentile(np.concatenate(([-np.inf], X)), 100 * q, interpolation="higher")

    assert estimate == estimate_np


@given(gufunc("(n),(),()->(),()", dtype=np.float_, elements=[floats_, probs, probs]))
def test_quantile_CI(args):
    X, q, alpha = args

    idx_q = qt._quantile(len(X), q)
    idx_l, idx_u = qt._quantile_CI(len(X), q, alpha)
    assert idx_l <= idx_q
    assert idx_q <= idx_u

    # Lot's of checks already inside quantile_CI
    LB, UB = qt.quantile_CI(X, q, alpha)
    assert LB <= UB

    estimate = qt.quantile(X, q)
    assert LB <= estimate
    assert estimate <= UB


@given(gufunc("(n),(),()->(),()", dtype=np.float_, elements=[floats_, probs, probs]))
def test_quantile_CI_monotone_x(args):
    X, q, alpha = args

    assume(len(X) >= 1)

    LB1, UB1 = qt.quantile_CI(X, q, alpha)

    X2 = np.copy(X)
    X2[0] = -np.inf
    LB2, UB2 = qt.quantile_CI(X2, q, alpha)
    assert LB1 >= LB2
    assert UB1 >= UB2

    X2 = np.copy(X)
    X2[0] = np.inf
    LB2, UB2 = qt.quantile_CI(X2, q, alpha)
    assert LB1 <= LB2
    assert UB1 <= UB2


@given(gufunc("(n),(2),()->(),()", dtype=np.float_, elements=[floats_, probs, probs]))
def test_quantile_CI_monotone_q(args):
    X, q, alpha = args

    q1, q2 = sorted(q)

    # Lot's of checks already inside quantile_CI
    LB1, UB1 = qt.quantile_CI(X, q1, alpha)
    LB2, UB2 = qt.quantile_CI(X, q2, alpha)
    assert LB1 <= LB2
    assert UB1 <= UB2


@given(gufunc("(n),(),(2)->(),()", dtype=np.float_, elements=[floats_, probs, probs]))
def test_quantile_CI_monotone_alpha(args):
    X, q, alpha = args

    alpha1, alpha2 = sorted(alpha)

    # Lot's of checks already inside quantile_CI
    LB1, UB1 = qt.quantile_CI(X, q, alpha1)  # This CI should be larger
    LB2, UB2 = qt.quantile_CI(X, q, alpha2)
    assert LB1 <= LB2
    assert UB1 >= UB2


@given(
    gufunc(
        "(n),(),(),()->()", dtype=[np.float_, np.float_, np.int_, np.float_], elements=[floats_, probs, counts, probs]
    )
)
def test_max_quantile_CI(args):
    X, q, m, alpha = args

    estimate0, LB0, UB0 = qt.max_quantile_CI(X, q, m, alpha)
    assert LB0 <= estimate0
    assert estimate0 <= UB0

    # Recompute without using _ internal funcs
    q = q ** (1.0 / m)
    LB, UB = qt.quantile_CI(X, q, alpha=alpha)
    estimate = qt.quantile(X, q)

    assert estimate0 == estimate
    assert LB0 == LB
    assert UB0 == UB


@given(
    gufunc(
        "(n),(),(),()->()", dtype=[np.float_, np.float_, np.int_, np.float_], elements=[floats_, probs, counts, probs]
    )
)
def test_min_quantile_CI(args):
    X, q, m, alpha = args

    estimate0, LB0, UB0 = qt.min_quantile_CI(X, q, m, alpha)
    assert LB0 <= estimate0
    assert estimate0 <= UB0

    # Recompute without using _ internal funcs
    q = 1.0 - (1.0 - q) ** (1.0 / m)
    LB, UB = qt.quantile_CI(X, q, alpha=alpha)
    estimate = qt.quantile(X, q)

    assert estimate0 == estimate
    assert LB0 == LB
    assert UB0 == UB


@given(
    gufunc(
        "(n),(),(),()->()", dtype=[np.float_, np.float_, np.int_, np.float_], elements=[floats_, probs, counts, probs]
    )
)
def test_min_quantile_CI_to_max(args):
    X, q, m, alpha = args

    epsilon = 1e-8  # Small allowance for numerics

    estimate0, LB0, UB0 = qt.min_quantile_CI(X, q, m, alpha)

    # Try just above and below to allow for numerics error in case we are
    # just on the boundary.
    estimate1, LB1, UB1 = qt.max_quantile_CI(-X, (1.0 - q) - epsilon, m, alpha)
    estimate2, LB2, UB2 = qt.max_quantile_CI(-X, 1.0 - q, m, alpha)
    estimate3, LB3, UB3 = qt.max_quantile_CI(-X, (1.0 - q) + epsilon, m, alpha)

    if len(X) == 0:
        assert estimate0 == -np.inf  # quantile spec rounds down if n=0
    else:
        assert -estimate0 in (estimate1, estimate2, estimate3)

    assert -LB0 in (UB1, UB2, UB3)
    assert -UB0 in (LB1, LB2, LB3)


@given(gufunc("(n),(),()->()", dtype=np.float_, elements=[floats_, probs, probs]))
def test_quantile_and_CI(args):
    X, q, alpha = args

    estimate0, LB0, UB0 = qt.quantile_and_CI(X, q, alpha)
    assert LB0 <= estimate0
    assert estimate0 <= UB0

    # Recompute without using _ internal funcs
    LB, UB = qt.quantile_CI(X, q, alpha=alpha)
    estimate = qt.quantile(X, q)

    assert estimate0 == estimate
    assert LB0 == LB
    assert UB0 == UB


def test_order_stats_broadcast():
    broadcast_tester(order_stats_trim, "(n)->(n)", otype="float64", dtype=np.float_, elements=floats_)


def test_quantile_broadcast_0():
    broadcast_tester(
        qt.quantile, "(n),()->()", otype="float64", excluded=(0,), dtype=np.float_, elements=[floats_, probs]
    )


def test_quantile_broadcast_1():
    broadcast_tester(
        qt.quantile, "(n),()->()", otype="float64", excluded=(1,), dtype=np.float_, elements=[floats_, probs]
    )


def test_quantile_CI_broadcast_0():
    multi_broadcast_tester(
        qt.quantile_CI,
        "(n),(),()->(),()",
        otypes=["float64", "float64"],
        excluded=(0,),
        dtype=np.float_,
        elements=[floats_, probs, probs],
    )


def test_quantile_CI_broadcast_1():
    multi_broadcast_tester(
        qt.quantile_CI,
        "(n),(),()->(),()",
        excluded=(1, 2),
        otypes=["float64", "float64"],
        dtype=np.float_,
        elements=[floats_, probs, probs],
    )


def test_max_quantile_CI_broadcast_0():
    multi_broadcast_tester(
        qt.max_quantile_CI,
        "(n),(),(),()->(),(),()",
        otypes=["float64", "float64", "float64"],
        excluded=(0,),
        dtype=[np.float_, np.float_, np.int_, np.float_],
        elements=[floats_, probs, counts, probs],
    )


def test_max_quantile_CI_broadcast_1():
    multi_broadcast_tester(
        qt.max_quantile_CI,
        "(n),(),(),()->(),(),()",
        otypes=["float64", "float64", "float64"],
        excluded=(1, 2, 3),
        dtype=[np.float_, np.float_, np.int_, np.float_],
        elements=[floats_, probs, counts, probs],
    )


def test_min_quantile_CI_broadcast_0():
    multi_broadcast_tester(
        qt.min_quantile_CI,
        "(n),(),(),()->(),(),()",
        otypes=["float64", "float64", "float64"],
        excluded=(0,),
        dtype=[np.float_, np.float_, np.int_, np.float_],
        elements=[floats_, probs, counts, probs],
    )


def test_min_quantile_CI_broadcast_1():
    multi_broadcast_tester(
        qt.min_quantile_CI,
        "(n),(),(),()->(),(),()",
        otypes=["float64", "float64", "float64"],
        excluded=(1, 2, 3),
        dtype=[np.float_, np.float_, np.int_, np.float_],
        elements=[floats_, probs, counts, probs],
    )


def test_quantile_and_CI_broadcast_0():
    multi_broadcast_tester(
        qt.quantile_and_CI,
        "(n),(),()->(),(),()",
        otypes=["float64", "float64", "float64"],
        excluded=(0,),
        dtype=np.float_,
        elements=[floats_, probs, probs],
    )


def test_quantile_and_CI_broadcast_1():
    multi_broadcast_tester(
        qt.quantile_and_CI,
        "(n),(),()->(),(),()",
        otypes=["float64", "float64", "float64"],
        excluded=(1, 2),
        dtype=np.float_,
        elements=[floats_, probs, probs],
    )


def mc_test_quantile_CI(mc_runs=1000, n=2000, q=0.5, alpha=0.05, random=np.random):
    q0 = ss.norm.ppf(q)

    X = random.randn(mc_runs, n)
    R = np.array([qt.quantile_CI(xx, q) for xx in X])
    LB, UB = R[:, 0], R[:, 1]

    n_pass = np.sum((LB <= q0) & (q0 <= UB))
    # This is only a one-sided test
    pval = ss.binom.cdf(n_pass, mc_runs, 1 - alpha)
    return pval


def mc_test_max_quantile_CI(mc_runs=1000, n=2000, q=0.5, m=100, alpha=0.05, random=np.random):
    qq_level = q ** (1.0 / m)
    q0 = ss.norm.ppf(qq_level)

    X = random.randn(mc_runs, n)
    R = np.array([qt.max_quantile_CI(xx, q, m, alpha) for xx in X])
    LB, UB = R[:, 1], R[:, 2]

    n_pass = np.sum((LB <= q0) & (q0 <= UB))
    # This is only a one-sided test
    pval = ss.binom.cdf(n_pass, mc_runs, 1 - alpha)
    return pval


def mc_test_min_quantile_CI(mc_runs=1000, n=2000, q=0.5, m=100, alpha=0.05, random=np.random):
    qq_level = 1.0 - (1.0 - q) ** (1.0 / m)
    q0 = ss.norm.ppf(qq_level)

    X = random.randn(mc_runs, n)
    R = np.array([qt.min_quantile_CI(xx, q, m, alpha) for xx in X])
    LB, UB = R[:, 1], R[:, 2]

    n_pass = np.sum((LB <= q0) & (q0 <= UB))
    # This is only a one-sided test
    pval = ss.binom.cdf(n_pass, mc_runs, 1 - alpha)
    return pval


def test_all_mc():
    random = np.random.RandomState(8623)

    pvals = []
    pvals.append(mc_test_quantile_CI(q=0.3, random=random))
    pvals.append(mc_test_quantile_CI(q=0.5, random=random))
    pvals.append(mc_test_quantile_CI(q=0.99, random=random))
    pvals.append(mc_test_max_quantile_CI(q=0.3, random=random))
    pvals.append(mc_test_max_quantile_CI(q=0.5, random=random))
    pvals.append(mc_test_max_quantile_CI(q=0.99, random=random))
    pvals.append(mc_test_min_quantile_CI(q=0.3, random=random))
    pvals.append(mc_test_min_quantile_CI(q=0.5, random=random))
    pvals.append(mc_test_min_quantile_CI(q=0.99, random=random))

    SIDAK_FPR = 1.0 - (1.0 - GLOBAL_FPR) ** (1.0 / len(pvals))
    assert np.min(pvals) >= SIDAK_FPR

    return pvals
