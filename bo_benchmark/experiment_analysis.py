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
"""Perform analysis to compare different optimizers across problems.
"""
import logging
import warnings

import numpy as np
import xarray as xr

import bo_benchmark.constants as cc
import bo_benchmark.quantiles as qt
import bo_benchmark.xr_util as xru
from bo_benchmark.cmd_parse import CmdArgs, general_parser, parse_args, serializable_dict
from bo_benchmark.constants import (
    ITER,
    LB_MEAN,
    LB_MED,
    LB_NORMED_MEAN,
    METHOD,
    NORMED_MEAN,
    NORMED_MED,
    PERF_BEST,
    PERF_CLIP,
    PERF_MEAN,
    PERF_MED,
    SUGGEST,
    TEST_CASE,
    TRIAL,
    UB_MEAN,
    UB_MED,
    UB_NORMED_MEAN,
)
from bo_benchmark.experiment_aggregate import validate_agg_perf
from bo_benchmark.experiment_baseline import do_baseline
from bo_benchmark.np_util import linear_rescale
from bo_benchmark.serialize import XRSerializer
from bo_benchmark.signatures import analyze_signature_pair
from bo_benchmark.stats import t_EB

# Mathematical settings
EVAL_Q = 0.5  # Evaluate based on median loss across n_trials
ALPHA = 0.05  # ==> 95% CIs

logger = logging.getLogger(__name__)


def compute_aggregates(perf_da, baseline_ds):
    """Aggregate function evaluations in the experiments to get performance summaries of each method.

    Parameters
    ----------
    perf_da : :class:`xarray:xarray.DataArray`
        Aggregate experimental results with each function evaluation in the experiments. `all_perf` has dimensions
        ``(ITER, SUGGEST, TEST_CASE, METHOD, TRIAL)`` as is assumed to have no nan values.
    baseline_ds : :class:`xarray:xarray.Dataset`
        Dataset with baseline performance. It was variables ``(PERF_MED, PERF_MEAN, PERF_CLIP, PERF_BEST)`` with
        dimensions ``(ITER, TEST_CASE)``, ``(ITER, TEST_CASE)``, ``(TEST_CASE,)``, and ``(TEST_CASE,)``, respectively.
        `PERF_MED` is a baseline of performance based on random search when using medians to summarize performance.
        Likewise, `PERF_MEAN` is for means. `PERF_CLIP` is an upperbound to clip poor performance when using the mean.
        `PERF_BEST` is an estimate on the global minimum.

    Returns
    -------
    agg_result : :class:`xarray:xarray.Dataset`
        Dataset with summary of performance for each method and test case combination. Contains variables:
        ``(PERF_MED, LB_MED, UB_MED, NORMED_MED, PERF_MEAN, LB_MEAN, UB_MEAN, NORMED_MEAN)``
        each with dimensions ``(ITER, METHOD, TEST_CASE)``. `PERF_MED` is a median summary of performance with `LB_MED`
        and `UB_MED` as error bars. `NORMED_MED` is a rescaled `PERF_MED` so we expect the optimal performance is 0,
        and random search gives 1 at all `ITER`. Likewise, `PERF_MEAN`, `LB_MEAN`, `UB_MEAN`, `NORMED_MEAN` are for
        mean performance.
    summary : :class:`xarray:xarray.Dataset`
        Dataset with overall summary of performance of each method. Contains variables
        ``(PERF_MED, LB_MED, UB_MED, PERF_MEAN, LB_MEAN, UB_MEAN)``
        each with dimensions ``(ITER, METHOD)``.
    """
    validate_agg_perf(perf_da, min_trial=1)

    assert isinstance(baseline_ds, xr.Dataset)
    assert tuple(baseline_ds[PERF_BEST].dims) == (TEST_CASE,)
    assert tuple(baseline_ds[PERF_CLIP].dims) == (TEST_CASE,)
    assert tuple(baseline_ds[PERF_MED].dims) == (ITER, TEST_CASE)
    assert tuple(baseline_ds[PERF_MEAN].dims) == (ITER, TEST_CASE)
    assert xru.coord_compat((perf_da, baseline_ds), (ITER, TEST_CASE))
    assert not any(np.any(np.isnan(baseline_ds[kk].values)) for kk in baseline_ds)

    # Now actually get the aggregate performance numbers per test case
    agg_result = xru.ds_like(
        perf_da,
        (PERF_MED, LB_MED, UB_MED, NORMED_MED, PERF_MEAN, LB_MEAN, UB_MEAN, NORMED_MEAN),
        (ITER, METHOD, TEST_CASE),
    )
    baseline_mean_da = xru.only_dataarray(xru.ds_like(perf_da, ["ref"], (ITER, TEST_CASE)))
    # Using values here since just clearer to get raw items than xr object for func_name
    for func_name in perf_da.coords[TEST_CASE].values:
        rand_perf_med = baseline_ds[PERF_MED].sel({TEST_CASE: func_name}, drop=True).values
        rand_perf_mean = baseline_ds[PERF_MEAN].sel({TEST_CASE: func_name}, drop=True).values
        best_opt = baseline_ds[PERF_BEST].sel({TEST_CASE: func_name}, drop=True).values
        base_clip_val = baseline_ds[PERF_CLIP].sel({TEST_CASE: func_name}, drop=True).values

        assert np.all(np.diff(rand_perf_med) <= 0), "Baseline should be decreasing with iteration"
        assert np.all(np.diff(rand_perf_mean) <= 0), "Baseline should be decreasing with iteration"
        assert np.all(rand_perf_med > best_opt)
        assert np.all(rand_perf_mean > best_opt)
        assert np.all(rand_perf_mean <= base_clip_val)

        baseline_mean_da.loc[{TEST_CASE: func_name}] = linear_rescale(
            rand_perf_mean, best_opt, base_clip_val, 0.0, 1.0, enforce_bounds=False
        )
        for method_name in perf_da.coords[METHOD].values:
            # Take the minimum over all suggestion at given iter + sanity check perf_da
            curr_da = perf_da.sel({METHOD: method_name, TEST_CASE: func_name}, drop=True).min(dim=SUGGEST)
            assert curr_da.dims == (ITER, TRIAL)

            # Want to evaluate minimum so far during optimization
            perf_array = np.minimum.accumulate(curr_da.values, axis=0)

            # Compute median perf and CI on it
            med_perf, LB, UB = qt.quantile_and_CI(perf_array, EVAL_Q, alpha=ALPHA)
            assert med_perf.shape == rand_perf_med.shape
            agg_result[PERF_MED].loc[{TEST_CASE: func_name, METHOD: method_name}] = med_perf
            agg_result[LB_MED].loc[{TEST_CASE: func_name, METHOD: method_name}] = LB
            agg_result[UB_MED].loc[{TEST_CASE: func_name, METHOD: method_name}] = UB

            # Now store normed version, which is better for aggregation
            normed = linear_rescale(med_perf, best_opt, rand_perf_med, 0.0, 1.0, enforce_bounds=False)
            agg_result[NORMED_MED].loc[{TEST_CASE: func_name, METHOD: method_name}] = normed

            # Compute mean perf and CI on it
            perf_array = np.minimum(base_clip_val, perf_array)
            mean_perf = np.mean(perf_array, axis=1)
            assert mean_perf.shape == rand_perf_mean.shape
            EB = t_EB(perf_array, alpha=ALPHA, axis=1)
            assert EB.shape == rand_perf_mean.shape
            agg_result[PERF_MEAN].loc[{TEST_CASE: func_name, METHOD: method_name}] = mean_perf
            agg_result[LB_MEAN].loc[{TEST_CASE: func_name, METHOD: method_name}] = mean_perf - EB
            agg_result[UB_MEAN].loc[{TEST_CASE: func_name, METHOD: method_name}] = mean_perf + EB

            # Now store normed version, which is better for aggregation
            normed = linear_rescale(mean_perf, best_opt, base_clip_val, 0.0, 1.0, enforce_bounds=False)
            agg_result[NORMED_MEAN].loc[{TEST_CASE: func_name, METHOD: method_name}] = normed
    assert not any(np.any(np.isnan(agg_result[kk].values)) for kk in agg_result)

    # Compute summary score over all test cases, summarize performance of each method
    summary = xru.ds_like(
        perf_da,
        (PERF_MED, LB_MED, UB_MED, PERF_MEAN, LB_MEAN, UB_MEAN, NORMED_MEAN, LB_NORMED_MEAN, UB_NORMED_MEAN),
        (ITER, METHOD),
    )
    summary[PERF_MED], summary[LB_MED], summary[UB_MED] = xr.apply_ufunc(
        qt.quantile_and_CI,
        agg_result[NORMED_MED],
        input_core_dims=[[TEST_CASE]],
        kwargs={"q": EVAL_Q, "alpha": ALPHA},
        output_core_dims=[[], [], []],
    )

    summary[PERF_MEAN] = agg_result[NORMED_MEAN].mean(dim=TEST_CASE)
    EB = xr.apply_ufunc(t_EB, agg_result[NORMED_MEAN], input_core_dims=[[TEST_CASE]])
    summary[LB_MEAN] = summary[PERF_MEAN] - EB
    summary[UB_MEAN] = summary[PERF_MEAN] + EB

    normalizer = baseline_mean_da.mean(dim=TEST_CASE)
    summary[NORMED_MEAN] = summary[PERF_MEAN] / normalizer
    summary[LB_NORMED_MEAN] = summary[LB_MEAN] / normalizer
    summary[UB_NORMED_MEAN] = summary[UB_MEAN] / normalizer

    assert all(tuple(summary[kk].dims) == (ITER, METHOD) for kk in summary)
    return agg_result, summary


def main():
    """See README.md for instructions on calling analysis.
    """
    description = "Analyze results from aggregated studies"
    args = parse_args(general_parser(description))

    logger.setLevel(logging.INFO)  # Note this is the module-wide logger
    if args[CmdArgs.verbose]:
        logger.addHandler(logging.StreamHandler())

    # Load in the eval data and sanity check
    perf_da, meta = XRSerializer.load_derived(args[CmdArgs.db_root], db=args[CmdArgs.db], key=cc.EVAL_RESULTS)
    perf_da = xru.only_dataarray(perf_da)
    logger.info("Meta data from source file: %s" % str(meta["args"]))

    # Check if there is baselines file, other make one
    if cc.BASELINE not in XRSerializer.get_derived_keys(args[CmdArgs.db_root], db=args[CmdArgs.db]):
        warnings.warn("Baselines not found. Need to construct baseline.")
        do_baseline(args)

    # Load in baseline scores data and sanity check (including compatibility with eval data)
    baseline_ds, meta_ref = XRSerializer.load_derived(args[CmdArgs.db_root], db=args[CmdArgs.db], key=cc.BASELINE)
    logger.info("baseline data from source ref file: %s" % str(meta_ref["args"]))

    # Check test case signatures match between eval data and baseline data
    sig_errs, signatures = analyze_signature_pair(meta["signature"], meta_ref["signature"])
    logger.info("Signature errors:\n%s" % sig_errs.to_string())

    # Do the actual computation
    agg_result, summary = compute_aggregates(perf_da, baseline_ds)

    final_score = summary[PERF_MED][{ITER: -1}]
    logger.info("median score @ %d:\n%s" % (summary.sizes[ITER], xru.da_to_string(final_score)))
    final_score = summary[PERF_MEAN][{ITER: -1}]
    logger.info("mean score @ %d:\n%s" % (summary.sizes[ITER], xru.da_to_string(final_score)))
    final_score = summary[NORMED_MEAN][{ITER: -1}]
    logger.info("normed mean score @ %d:\n%s" % (summary.sizes[ITER], xru.da_to_string(final_score)))

    # Now saving results
    meta = {"args": serializable_dict(args), "signature": signatures}
    XRSerializer.save_derived(agg_result, meta, args[CmdArgs.db_root], db=args[CmdArgs.db], key=cc.PERF_RESULTS)

    XRSerializer.save_derived(summary, meta, args[CmdArgs.db_root], db=args[CmdArgs.db], key=cc.MEAN_SCORE)

    logger.info("done")


if __name__ == "__main__":
    main()  # pragma: main
