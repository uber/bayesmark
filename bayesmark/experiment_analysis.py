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
import json
import logging
import warnings
from collections import OrderedDict

import numpy as np
import xarray as xr

import bayesmark.constants as cc
import bayesmark.quantiles as qt
import bayesmark.xr_util as xru
from bayesmark.cmd_parse import CmdArgs, general_parser, parse_args, serializable_dict
from bayesmark.constants import (
    ITER,
    LB_MEAN,
    LB_MED,
    LB_NORMED_MEAN,
    METHOD,
    NORMED_MEAN,
    NORMED_MED,
    OBJECTIVE,
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
from bayesmark.experiment_aggregate import validate_agg_perf
from bayesmark.experiment_baseline import do_baseline
from bayesmark.np_util import cummin, linear_rescale
from bayesmark.serialize import XRSerializer
from bayesmark.signatures import analyze_signature_pair
from bayesmark.stats import t_EB

# Mathematical settings
EVAL_Q = 0.5  # Evaluate based on median loss across n_trials
ALPHA = 0.05  # ==> 95% CIs

logger = logging.getLogger(__name__)


def get_perf_array(evals, evals_visible):
    """Get the actual (e.g., generalization loss) over iterations.

    Parameters
    ----------
    evals : :class:`numpy:numpy.ndarray` of shape (n_iter, n_batch, n_trials)
        The actual loss (e.g., generalization) for a given experiment.
    evals_visible : :class:`numpy:numpy.ndarray` of shape (n_iter, n_batch, n_trials)
        The observable loss (e.g., validation) for a given experiment.

    Returns
    -------
    perf_array : :class:`numpy:numpy.ndarray` of shape (n_iter, n_trials)
        The best performance so far at iteration i from `evals`. Where the best has been selected according to
        `evals_visible`.
    """
    n_iter, _, n_trials = evals.shape
    assert evals.size > 0, "perf array not supported for empty arrays"
    assert evals_visible.shape == evals.shape
    assert not np.any(np.isnan(evals))
    assert not np.any(np.isnan(evals_visible))

    idx = np.argmin(evals_visible, axis=1)
    perf_array = np.take_along_axis(evals, idx[:, None, :], axis=1).squeeze(axis=1)
    assert perf_array.shape == (n_iter, n_trials)

    visible_perf_array = np.min(evals_visible, axis=1)
    assert visible_perf_array.shape == (n_iter, n_trials)

    # Get the minimum from the visible loss
    perf_array = cummin(perf_array, visible_perf_array)
    return perf_array


def compute_aggregates(perf_da, baseline_ds, visible_perf_da=None):
    """Aggregate function evaluations in the experiments to get performance summaries of each method.

    Parameters
    ----------
    perf_da : :class:`xarray:xarray.DataArray`
        Aggregate experimental results with each function evaluation in the experiments according to true loss
        (e.g., generalization). `perf_da` has dimensions ``(ITER, SUGGEST, TEST_CASE, METHOD, TRIAL)`` as is assumed
        to have no nan values.
    baseline_ds : :class:`xarray:xarray.Dataset`
        Dataset with baseline performance. It was variables ``(PERF_MED, PERF_MEAN, PERF_CLIP, PERF_BEST)`` with
        dimensions ``(ITER, TEST_CASE)``, ``(ITER, TEST_CASE)``, ``(TEST_CASE,)``, and ``(TEST_CASE,)``, respectively.
        `PERF_MED` is a baseline of performance based on random search when using medians to summarize performance.
        Likewise, `PERF_MEAN` is for means. `PERF_CLIP` is an upperbound to clip poor performance when using the mean.
        `PERF_BEST` is an estimate on the global minimum.
    visible_perf_da : :class:`xarray:xarray.DataArray`
        Aggregate experimental results with each function evaluation in the experiments according to visible loss
        (e.g., validation). `visible_perf_da` has dimensions ``(ITER, SUGGEST, TEST_CASE, METHOD, TRIAL)`` as is
        assumed to have no nan values. If `None`, we set ``visible_perf_da = perf_da``.

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
            curr_da = perf_da.sel({METHOD: method_name, TEST_CASE: func_name}, drop=True)
            assert curr_da.dims == (ITER, SUGGEST, TRIAL)

            if visible_perf_da is None:
                perf_array = get_perf_array(curr_da.values, curr_da.values)

                curr_da_ = perf_da.sel({METHOD: method_name, TEST_CASE: func_name}, drop=True).min(dim=SUGGEST)
                assert curr_da_.dims == (ITER, TRIAL)
                perf_array_ = np.minimum.accumulate(curr_da_.values, axis=0)
                assert np.allclose(perf_array, perf_array_)
            else:
                curr_visible_da = visible_perf_da.sel({METHOD: method_name, TEST_CASE: func_name}, drop=True)
                assert curr_visible_da.dims == (ITER, SUGGEST, TRIAL)
                perf_array = get_perf_array(curr_da.values, curr_visible_da.values)

            # Compute median perf and CI on it
            med_perf, LB, UB = qt.quantile_and_CI(perf_array, EVAL_Q, alpha=ALPHA)
            assert med_perf.shape == rand_perf_med.shape
            agg_result[PERF_MED].loc[{TEST_CASE: func_name, METHOD: method_name}] = med_perf
            agg_result[LB_MED].loc[{TEST_CASE: func_name, METHOD: method_name}] = LB
            agg_result[UB_MED].loc[{TEST_CASE: func_name, METHOD: method_name}] = UB

            # Now store normed version, which is better for aggregation
            normed = linear_rescale(med_perf, best_opt, rand_perf_med, 0.0, 1.0, enforce_bounds=False)
            agg_result[NORMED_MED].loc[{TEST_CASE: func_name, METHOD: method_name}] = normed

            # Store normed mean version
            normed = linear_rescale(perf_array, best_opt, base_clip_val, 0.0, 1.0, enforce_bounds=False)
            # Also, clip the score from below at -1 to limit max influence of single run on final average
            normed = np.clip(normed, -1.0, 1.0)
            normed = np.mean(normed, axis=1)
            agg_result[NORMED_MEAN].loc[{TEST_CASE: func_name, METHOD: method_name}] = normed

            # Compute mean perf and CI on it
            perf_array = np.minimum(base_clip_val, perf_array)
            mean_perf = np.mean(perf_array, axis=1)
            assert mean_perf.shape == rand_perf_mean.shape
            EB = t_EB(perf_array, alpha=ALPHA, axis=1)
            assert EB.shape == rand_perf_mean.shape
            agg_result[PERF_MEAN].loc[{TEST_CASE: func_name, METHOD: method_name}] = mean_perf
            agg_result[LB_MEAN].loc[{TEST_CASE: func_name, METHOD: method_name}] = mean_perf - EB
            agg_result[UB_MEAN].loc[{TEST_CASE: func_name, METHOD: method_name}] = mean_perf + EB
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
    """See README for instructions on calling analysis.
    """
    description = "Analyze results from aggregated studies"
    args = parse_args(general_parser(description))

    # Metric used on leaderboard
    leaderboard_metric = cc.VISIBLE_TO_OPT

    logger.setLevel(logging.INFO)  # Note this is the module-wide logger
    if args[CmdArgs.verbose]:
        logger.addHandler(logging.StreamHandler())

    # Load in the eval data and sanity check
    perf_ds, meta = XRSerializer.load_derived(args[CmdArgs.db_root], db=args[CmdArgs.db], key=cc.EVAL_RESULTS)
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
    print(json.dumps({"exp-anal sig errors": sig_errs.T.to_dict()}))

    # Subset baseline to only the test cases run in the experiments
    test_cases_run = perf_ds.coords[TEST_CASE].values.tolist()
    assert set(test_cases_run) <= set(
        baseline_ds.coords[TEST_CASE].values.tolist()
    ), "Data set contains test cases not found in baseline."
    baseline_ds = baseline_ds.sel({TEST_CASE: test_cases_run})

    # Also subset to allow shorter runs
    iters_run = perf_ds.coords[ITER].values.tolist()
    assert set(iters_run) <= set(
        baseline_ds.coords[ITER].values.tolist()
    ), "Data set not same batch size or too many iters compared to baseline."
    baseline_ds = baseline_ds.sel({ITER: iters_run})

    # Do the actual computation
    perf_visible = perf_ds[cc.VISIBLE_TO_OPT]
    agg_result = OrderedDict()
    summary = OrderedDict()
    for metric_for_scoring in sorted(perf_ds):
        perf_da = perf_ds[metric_for_scoring]
        baseline_ds_ = baseline_ds.sel({OBJECTIVE: metric_for_scoring}, drop=True)
        agg_result[(metric_for_scoring,)], summary[(metric_for_scoring,)] = compute_aggregates(
            perf_da, baseline_ds_, perf_visible
        )
    agg_result = xru.ds_concat(agg_result, dims=(cc.OBJECTIVE,))
    summary = xru.ds_concat(summary, dims=(cc.OBJECTIVE,))

    for metric_for_scoring in sorted(perf_ds):
        # Print summary by problem
        # Recall that:
        # ... summary[PERF_MEAN] = agg_result[NORMED_MEAN].mean(dim=TEST_CASE)
        # ... summary[NORMED_MEAN] = summary[PERF_MEAN] / normalizer
        # Where normalizer is constant across all problems, optimizers
        print("Scores by problem (JSON):\n")
        agg_df = agg_result[NORMED_MEAN].sel({cc.OBJECTIVE: metric_for_scoring}, drop=True)[{ITER: -1}].to_pandas().T
        print(json.dumps({metric_for_scoring: agg_df.to_dict()}))
        print("\n")

        final_score = summary[PERF_MED].sel({cc.OBJECTIVE: metric_for_scoring}, drop=True)[{ITER: -1}]
        logger.info("median score @ %d:\n%s" % (summary.sizes[ITER], xru.da_to_string(final_score)))
        final_score = summary[PERF_MEAN].sel({cc.OBJECTIVE: metric_for_scoring}, drop=True)[{ITER: -1}]
        logger.info("mean score @ %d:\n%s" % (summary.sizes[ITER], xru.da_to_string(final_score)))

        print("Final scores (JSON):\n")
        print(json.dumps({metric_for_scoring: final_score.to_series().to_dict()}))
        print("\n")

        final_score = summary[NORMED_MEAN].sel({cc.OBJECTIVE: metric_for_scoring}, drop=True)[{ITER: -1}]
        logger.info("normed mean score @ %d:\n%s" % (summary.sizes[ITER], xru.da_to_string(final_score)))

    # Now saving results
    meta = {"args": serializable_dict(args), "signature": signatures}
    XRSerializer.save_derived(agg_result, meta, args[CmdArgs.db_root], db=args[CmdArgs.db], key=cc.PERF_RESULTS)

    XRSerializer.save_derived(summary, meta, args[CmdArgs.db_root], db=args[CmdArgs.db], key=cc.MEAN_SCORE)

    final_msg = xru.da_to_string(
        100 * (1.0 - summary[PERF_MEAN].sel({cc.OBJECTIVE: leaderboard_metric}, drop=True)[{ITER: -1}])
    )
    logger.info("-" * 20)
    logger.info("Final score `100 x (1-loss)` for leaderboard:\n%s" % final_msg)


if __name__ == "__main__":
    main()  # pragma: main
