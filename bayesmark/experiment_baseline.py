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
"""Build performance baselines from aggregate results to prepare analysis.
"""
import logging
import warnings
from collections import OrderedDict

import numpy as np

import bayesmark.constants as cc
import bayesmark.expected_max as em
import bayesmark.quantiles as qt
from bayesmark.cmd_parse import CmdArgs, general_parser, parse_args
from bayesmark.constants import ARG_DELIM, ITER, METHOD, PERF_BEST, PERF_CLIP, PERF_MEAN, PERF_MED, SUGGEST, TEST_CASE
from bayesmark.experiment_aggregate import validate_agg_perf
from bayesmark.serialize import XRSerializer
from bayesmark.util import str_join_safe
from bayesmark.xr_util import ds_concat, ds_like_mixed

# Mathematical settings
# We could move these to constants to eliminate repetition but we will probably phase out anyway
EVAL_Q = 0.5  # Evaluate based on median loss across n_trials
ALPHA = 0.05  # ==> 95% CIs
MIN_POS = np.nextafter(0, 1)
PAD_FACTOR = 10000

logger = logging.getLogger(__name__)


def validate(baseline_ds):
    """Perform same tracks as will happen in analysis."""
    for func_name in baseline_ds.coords[TEST_CASE].values:
        rand_perf_med = baseline_ds[PERF_MED].sel({TEST_CASE: func_name}, drop=True).values
        rand_perf_mean = baseline_ds[PERF_MEAN].sel({TEST_CASE: func_name}, drop=True).values
        best_opt = baseline_ds[PERF_BEST].sel({TEST_CASE: func_name}, drop=True).values
        base_clip_val = baseline_ds[PERF_CLIP].sel({TEST_CASE: func_name}, drop=True).values

        assert np.all(np.diff(rand_perf_med) <= 0), "Baseline should be decreasing with iteration"
        assert np.all(np.diff(rand_perf_mean) <= 0), "Baseline should be decreasing with iteration"
        assert np.all(rand_perf_med > best_opt)
        assert np.all(rand_perf_mean > best_opt)
        assert np.all(rand_perf_mean <= base_clip_val)


def compute_baseline(perf_da):
    """Compute a performance baseline of base and best performance from the aggregate experimental results.

    Parameters
    ----------
    perf_da : :class:`xarray:xarray.DataArray`
        Aggregate experimental results with each function evaluation in the experiments. `all_perf` has dimensions
        ``(ITER, SUGGEST, TEST_CASE, METHOD, TRIAL)`` as is assumed to have no nan values.

    Returns
    -------
    baseline_ds : :class:`xarray:xarray.Dataset`
        Dataset with baseline performance. It was variables ``(PERF_MED, PERF_MEAN, PERF_CLIP, PERF_BEST)`` with
        dimensions ``(ITER, TEST_CASE)``, ``(ITER, TEST_CASE)``, ``(TEST_CASE,)``, and ``(TEST_CASE,)``, respectively.
        `PERF_MED` is a baseline of performance based on random search when using medians to summarize performance.
        Likewise, `PERF_MEAN` is for means. `PERF_CLIP` is an upperbound to clip poor performance when using the mean.
        `PERF_BEST` is an estimate on the global minimum.
    """
    validate_agg_perf(perf_da)

    ref_prefix = str_join_safe(ARG_DELIM, (cc.RANDOM_SEARCH, ""))
    ref_random = [kk for kk in perf_da.coords[METHOD].values if kk.startswith(ref_prefix)]
    assert len(ref_random) > 0, "Did not find any random search in methods."

    # Now many points we will have after each batch
    trials_grid = perf_da.sizes[SUGGEST] * (1 + np.arange(perf_da.sizes[ITER]))

    # Now iterate over problems and get baseline performance
    baseline_ds = ds_like_mixed(
        perf_da,
        [
            (PERF_MED, [ITER, TEST_CASE]),
            (PERF_MEAN, [ITER, TEST_CASE]),
            (PERF_CLIP, [TEST_CASE]),
            (PERF_BEST, [TEST_CASE]),
        ],
        (ITER, TEST_CASE),
    )
    for func_name in perf_da.coords[TEST_CASE].values:
        random_evals = np.ravel(perf_da.sel({METHOD: ref_random, TEST_CASE: func_name}, drop=True).values)
        assert random_evals.size > 0

        # We will likely change this to a min mean (instead of median) using a different util in near future:
        assert np.all(trials_grid == perf_da.sizes[SUGGEST] * (1 + baseline_ds.coords[ITER].values))
        rand_perf, _, _ = qt.min_quantile_CI(random_evals, EVAL_Q, trials_grid, alpha=ALPHA)
        baseline_ds[PERF_MED].loc[{TEST_CASE: func_name}] = rand_perf

        # Decide on a level to clip when computing the mean
        base_clip_val = qt.quantile(random_evals, EVAL_Q)
        assert np.isfinite(base_clip_val), "Median random search performance is not even finite."
        assert (perf_da.sizes[SUGGEST] > 1) or np.isclose(base_clip_val, rand_perf[0])
        baseline_ds[PERF_CLIP].loc[{TEST_CASE: func_name}] = base_clip_val

        # Estimate the global min via best of any method
        best_opt = np.min(perf_da.sel({TEST_CASE: func_name}, drop=True).values)
        if np.any(rand_perf <= best_opt):
            warnings.warn(
                "Random search is also the best search on %s, the normalized score may be meaningless." % func_name,
                RuntimeWarning,
            )
        assert np.isfinite(best_opt), "Best performance found is not even finite."
        logger.info("best %s %f" % (func_name, best_opt))

        # Now make sure strictly less than to avoid assert error in linear_rescale. This will likely give normalized
        # scores of +inf or -inf, but with median summary that is ok. When everything goes to mean, we will need to
        # change this:
        pad = PAD_FACTOR * np.spacing(-np.maximum(MIN_POS, np.abs(best_opt)))
        assert pad < 0
        best_opt = best_opt + pad
        assert np.isfinite(best_opt), "Best performance too close to limit of float range."
        assert np.all(rand_perf > best_opt)
        baseline_ds[PERF_BEST].loc[{TEST_CASE: func_name}] = best_opt

        random_evals = np.minimum(base_clip_val, random_evals)
        assert np.all(np.isfinite(random_evals))
        assert np.all(best_opt <= random_evals)

        rand_perf = em.expected_min(random_evals, trials_grid)
        rand_perf_fixed = np.minimum(base_clip_val, rand_perf)
        assert np.allclose(rand_perf, rand_perf_fixed)
        rand_perf_fixed = np.minimum.accumulate(rand_perf_fixed)
        assert np.allclose(rand_perf, rand_perf_fixed)
        baseline_ds[PERF_MEAN].loc[{TEST_CASE: func_name}] = rand_perf_fixed
    assert not any(np.any(np.isnan(baseline_ds[kk].values)) for kk in baseline_ds)
    validate(baseline_ds)
    return baseline_ds


def do_baseline(args):  # pragma: io
    """Alternate entry into the program without calling the actual main.
    """
    # Load in the eval data and sanity check
    perf_ds, meta = XRSerializer.load_derived(args[CmdArgs.db_root], db=args[CmdArgs.db], key=cc.EVAL_RESULTS)
    logger.info("Meta data from source file: %s" % str(meta["args"]))

    D = OrderedDict()
    for kk in perf_ds:
        perf_da = perf_ds[kk]
        D[(kk,)] = compute_baseline(perf_da)
    baseline_ds = ds_concat(D, dims=(cc.OBJECTIVE,))

    # Keep in same order for cleanliness
    baseline_ds = baseline_ds.sel({cc.OBJECTIVE: list(perf_ds)})
    assert list(perf_ds) == baseline_ds.coords[cc.OBJECTIVE].values.tolist()

    # Could optionally remove this once we think things have enough tests
    for kk in D:
        assert baseline_ds.sel({cc.OBJECTIVE: kk[0]}, drop=True).identical(D[kk])

    # Now dump the results
    XRSerializer.save_derived(baseline_ds, meta, args[CmdArgs.db_root], db=args[CmdArgs.db], key=cc.BASELINE)


def main():
    """See README for instructions on calling baseline.
    """
    description = "Aggregate the baselines for later analysis in benchmark"
    args = parse_args(general_parser(description))

    logger.setLevel(logging.INFO)  # Note this is the module-wide logger
    if args[CmdArgs.verbose]:
        logger.addHandler(logging.StreamHandler())

    do_baseline(args)
    logger.info("done")


if __name__ == "__main__":
    main()  # pragma: main
