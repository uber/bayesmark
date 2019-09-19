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
from itertools import product

import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats
from hypothesis_gufunc.extra.xr import (
    fixed_datasets,
    simple_coords,
    simple_dataarrays,
    simple_datasets,
    xr_coords,
    xr_dims,
)

import bo_benchmark.experiment_aggregate as agg
from bo_benchmark.constants import EVAL_PHASE, ITER, METHOD, OBS_PHASE, SUGGEST, SUGGEST_PHASE, TEST_CASE, TRIAL
from bo_benchmark.signatures import N_SUGGESTIONS

N_SIG = N_SUGGESTIONS
SIG_POINT = "sig_point"


def data_to_concat():
    def separate(ds):
        G = product(
            ds.coords[TEST_CASE].values.tolist(), ds.coords[METHOD].values.tolist(), ds.coords[TRIAL].values.tolist()
        )

        L = []
        for test_case, method, trial in G:
            # Could swap out trial for UUID here
            meta_data = (test_case, method, trial)

            ds_sub = ds.sel({TEST_CASE: test_case, METHOD: method, TRIAL: trial}, drop=True)

            perf_da = ds_sub["perf"]
            time_ds = ds_sub[[SUGGEST_PHASE, EVAL_PHASE, OBS_PHASE]]
            sig = ds_sub["sig"].values.tolist()
            data = (perf_da, time_ds, sig)
            L.append((meta_data, data))
            assert not np.any(np.isnan(perf_da.values))
            assert not any(np.any(np.isnan(time_ds[kk].values)) for kk in time_ds)
            assert not np.any(np.isnan(sig))
        return L

    vars_to_dims = {
        "perf": (ITER, SUGGEST, TEST_CASE, METHOD, TRIAL),
        "sig": (SIG_POINT, TEST_CASE, METHOD, TRIAL),
        SUGGEST_PHASE: (ITER, TEST_CASE, METHOD, TRIAL),
        EVAL_PHASE: (ITER, SUGGEST, TEST_CASE, METHOD, TRIAL),
        OBS_PHASE: (ITER, TEST_CASE, METHOD, TRIAL),
    }

    float_no_nan = floats(allow_nan=False, min_value=-10, max_value=10)
    dtype = {SUGGEST_PHASE: np.float_, EVAL_PHASE: np.float_, OBS_PHASE: np.float_, "perf": np.float_, "sig": np.float_}
    # Using on str following dim conventions for coords here
    coords_st = {
        ITER: simple_coords(min_side=1),
        SUGGEST: simple_coords(min_side=1),
        TEST_CASE: xr_coords(elements=xr_dims(), min_side=1),
        METHOD: xr_coords(elements=xr_dims(), min_side=1),
        TRIAL: simple_coords(min_side=1),
        SIG_POINT: simple_coords(min_side=N_SIG, max_side=N_SIG),
    }
    S = fixed_datasets(vars_to_dims, dtype=dtype, elements=float_no_nan, coords_st=coords_st, min_side=1).map(separate)
    return S


def time_datasets():
    vars_to_dims = {SUGGEST_PHASE: (ITER,), EVAL_PHASE: (ITER, SUGGEST), OBS_PHASE: (ITER,)}
    dtype = {SUGGEST_PHASE: np.float_, EVAL_PHASE: np.float_, OBS_PHASE: np.float_}
    elements = floats(min_value=0, allow_infinity=False, allow_nan=False)
    S = simple_datasets(vars_to_dims, dtype=dtype, elements=elements, min_side=1)
    return S


def perf_dataarrays():
    dims = (ITER, SUGGEST)
    elements = floats(allow_nan=False)
    S = simple_dataarrays(dims, dtype=np.float_, elements=elements)
    return S


@given(time_datasets())
def test_summarize_time(all_time):
    time_summary = agg.summarize_time(all_time)
    assert time_summary is not None


@given(perf_dataarrays())
def test_ravel_perf(perf_da):
    perf_da = agg._ravel_perf(perf_da)


@given(time_datasets())
def test_ravel_time(time_ds):
    time_ds = agg._ravel_time(time_ds)


@given(data_to_concat())
@settings(deadline=None)
def test_concat_experiments(all_experiments):
    all_experiments = list(all_experiments)
    all_perf, all_time, all_sigs = agg.concat_experiments(all_experiments, ravel=False)


@given(data_to_concat())
@settings(deadline=None)
def test_concat_experiments_ravel(all_experiments):
    all_perf, all_time, all_sigs = agg.concat_experiments(all_experiments, ravel=True)
