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
from hypothesis import given, settings

import bayesmark.experiment_baseline as base
from bayesmark import experiment_analysis as anal
from bayesmark.constants import TRIAL
from bayesmark.np_util import argmin_2d
from hypothesis_util import gufunc_floats
from util import perf_dataarrays


@given(gufunc_floats("(n,p,t),(n,p,t)->(n,t)", allow_nan=False, unique=True, min_side=1))
def test_get_perf_array(args):
    """Behavior for tie-breaking in `evals_visible` is complex, so only testing all unique case here."""
    evals, evals_visible = args

    n_iter, _, n_trials = evals.shape

    perf_array = anal.get_perf_array(evals, evals_visible)
    assert perf_array.shape == (n_iter, n_trials)

    for ii in range(n_iter):
        for jj in range(n_trials):
            idx0, idx1 = argmin_2d(evals_visible[: ii + 1, :, jj])
            assert perf_array[ii, jj] == evals[idx0, idx1, jj]


@given(perf_dataarrays(min_trial=2))
@settings(deadline=None)
def test_compute_aggregates(perf_da):
    n_trial = perf_da.sizes[TRIAL]

    split = n_trial // 2
    assert isinstance(split, int)

    perf_da1 = perf_da.isel({TRIAL: slice(None, split)})
    assert perf_da1.sizes[TRIAL] >= 1

    perf_da2 = perf_da.isel({TRIAL: slice(split, None)})
    assert perf_da2.sizes[TRIAL] >= 1
    perf_da2.coords[TRIAL] = list(range(perf_da2.sizes[TRIAL]))

    baseline_ds = base.compute_baseline(perf_da1)
    anal.compute_aggregates(perf_da2, baseline_ds)


@given(perf_dataarrays(min_trial=4))
@settings(deadline=None)
def test_compute_aggregates_with_aux(perf_da):
    # Split to get baseline
    n_trial = perf_da.sizes[TRIAL]
    split = n_trial // 2
    assert isinstance(split, int)
    perf_da1 = perf_da.isel({TRIAL: slice(None, split)})
    assert perf_da1.sizes[TRIAL] >= 1
    perf_da2 = perf_da.isel({TRIAL: slice(split, None)})
    assert perf_da2.sizes[TRIAL] >= 1
    perf_da2.coords[TRIAL] = list(range(perf_da2.sizes[TRIAL]))
    baseline_ds = base.compute_baseline(perf_da1)
    perf_da = perf_da2

    # Split to get visible
    n_trial = perf_da.sizes[TRIAL]
    split = n_trial // 2
    assert isinstance(split, int)
    perf_da1 = perf_da.isel({TRIAL: slice(None, split)})
    assert perf_da1.sizes[TRIAL] >= 1
    perf_da2 = perf_da.isel({TRIAL: slice(split, 2 * split)})
    assert perf_da2.sizes[TRIAL] >= 1
    perf_da2.coords[TRIAL] = list(range(perf_da2.sizes[TRIAL]))

    anal.compute_aggregates(perf_da2, baseline_ds, visible_perf_da=perf_da1)
