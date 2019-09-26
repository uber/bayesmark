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
from util import perf_dataarrays


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
