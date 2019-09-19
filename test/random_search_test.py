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
from hypothesis import given, settings
from hypothesis.strategies import integers

import bo_benchmark.space as sp
from bo_benchmark.np_util import linear_rescale
from bo_benchmark.random_search import suggest_dict
from hypothesis_util import close_enough, gufunc_floats, seeds
from util import space_configs


@given(space_configs(allow_missing=True), integers(min_value=1, max_value=8), seeds())
@settings(deadline=None)
def test_random_search_suggest_sanity(api_args, n_suggest, seed):
    meta, X, y, _ = api_args

    # Get the unwarped X
    S = sp.JointSpace(meta)
    lower, upper = S.get_bounds().T
    S.validate(X)

    N = len(X)
    # Split history and call twice with diff histories but same seed
    M = N // 2
    X1, X2 = X[:M], X[M:]
    y1, y2 = y[:M], y[M:]

    x_guess = suggest_dict(X1, y1, meta, n_suggest, random=np.random.RandomState(seed))
    x_guess2 = suggest_dict(X2, y2, meta, n_suggest, random=np.random.RandomState(seed))

    # Check types too
    assert len(x_guess) == n_suggest
    assert all(all(close_enough(x_guess[nn][k], x_guess2[nn][k]) for k in x_guess[nn]) for nn in range(len(x_guess)))
    assert np.all(x_guess == x_guess2)
    # Make sure validated
    S.validate(x_guess)
    S.validate(x_guess2)

    # Test sanity of output
    D, = lower.shape
    x_guess_w = S.warp(x_guess)
    assert type(x_guess_w) == np.ndarray
    assert x_guess_w.dtype.kind == "f"
    assert x_guess_w.shape == (n_suggest, D)
    assert x_guess_w.shape == (n_suggest, D)
    assert np.all(x_guess_w <= upper)


@given(
    gufunc_floats("(n,D),(n)->()", min_value=0.0, max_value=1.0, min_side={"D": 1}),
    integers(min_value=1, max_value=10),
    seeds(),
)
@settings(deadline=None)
def test_random_search_suggest_diff(api_args, n_suggest, seed):
    # Hard to know how many iters needed for arbitrary space that we need to
    # run so that we don't get dupes by chance. So, for now, let's just stick
    # with this simple space.
    dim = {"space": "linear", "type": "real", "range": [1.0, 5.0]}

    # Use at least 10 n_suggest to make sure don't get same answer by chance
    X_w, y = api_args

    D = X_w.shape[1]
    param_names = ["x%d" % ii for ii in range(5)]
    meta = dict(zip(param_names, [dim] * D))

    # Get the unwarped X
    S = sp.JointSpace(meta)
    lower, upper = S.get_bounds().T
    X_w = linear_rescale(X_w, lb0=0.0, ub0=1.0, lb1=lower, ub1=upper)
    X = S.unwarp(X_w)
    S.validate(X)

    seed = seed // 2  # Keep in bounds even after add 7

    x_guess = suggest_dict(X, y, meta, n_suggest, random=np.random.RandomState(seed))
    # Use diff seed to intentionally get diff result
    x_guess2 = suggest_dict(X, y, meta, n_suggest, random=np.random.RandomState(seed + 7))

    # Check types too
    assert len(x_guess) == n_suggest
    assert len(x_guess2) == n_suggest
    assert not np.all(x_guess == x_guess2)
    # Make sure validated
    S.validate(x_guess)
    S.validate(x_guess2)

    # Test sanity of output
    D, = lower.shape

    x_guess_w = S.warp(x_guess)
    assert type(x_guess_w) == np.ndarray
    assert x_guess_w.dtype.kind == "f"
    assert x_guess_w.shape == (n_suggest, D)
    assert x_guess_w.shape == (n_suggest, D)
    assert np.all(x_guess_w <= upper)

    x_guess_w = S.warp(x_guess2)
    assert type(x_guess_w) == np.ndarray
    assert x_guess_w.dtype.kind == "f"
    assert x_guess_w.shape == (n_suggest, D)
    assert x_guess_w.shape == (n_suggest, D)
    assert np.all(x_guess_w <= upper)
