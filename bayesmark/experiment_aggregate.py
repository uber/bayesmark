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
"""Aggregate the results of many studies to prepare analysis.
"""
import logging
from collections import Counter

import numpy as np
import xarray as xr

import bayesmark.constants as cc
import bayesmark.xr_util as xru
from bayesmark.cmd_parse import CmdArgs, agg_parser, parse_args, serializable_dict, unserializable_dict
from bayesmark.constants import ARG_DELIM, EVAL_RESULTS, ITER, METHOD, SUGGEST, TEST_CASE, TIME_RESULTS, TRIAL
from bayesmark.experiment import build_eval_ds, build_timing_ds
from bayesmark.serialize import XRSerializer
from bayesmark.signatures import analyze_signatures
from bayesmark.util import str_join_safe

logger = logging.getLogger(__name__)


def validate_time(all_time):
    assert isinstance(all_time, xr.Dataset)
    assert all_time[cc.SUGGEST_PHASE].dims == (ITER,)
    assert all_time[cc.EVAL_PHASE].dims == (ITER, SUGGEST)
    assert all_time[cc.OBS_PHASE].dims == (ITER,)
    assert xru.is_simple_coords(all_time.coords, min_side=1)


def validate_perf(perf_da):
    assert isinstance(perf_da, xr.DataArray)
    assert perf_da.dims == (ITER, SUGGEST)
    assert xru.is_simple_coords(perf_da.coords)
    assert not np.any(np.isnan(perf_da.values))


def validate_agg_perf(perf_da, min_trial=1):
    assert isinstance(perf_da, xr.DataArray)
    assert perf_da.dims == (ITER, SUGGEST, TEST_CASE, METHOD, TRIAL)
    assert xru.is_simple_coords(perf_da.coords, dims=(ITER, SUGGEST, TRIAL))
    assert not np.any(np.isnan(perf_da.values))
    assert perf_da.sizes[TRIAL] >= min_trial


def summarize_time(all_time):
    """Transform a single timing dataset from an experiment into a form better for aggregation.

    Parameters
    ----------
    all_time : :class:`xarray:xarray.Dataset`
        Dataset with variables ``(SUGGEST_PHASE, EVAL_PHASE, OBS_PHASE)`` which have dimensions ``(ITER,)``,
        ``(ITER, SUGGEST)``, and ``(ITER,)``, respectively. The variable `EVAL_PHASE` has the function evaluation time
        for each parallel suggestion.

    Returns
    -------
    time_summary : :class:`xarray:xarray.Dataset`
        Dataset with variables ``(SUGGEST_PHASE, OBS_PHASE, EVAL_PHASE_MAX, EVAL_PHASE_SUM)`` which all have dimensions
        ``(ITER,)``. The maximum `EVAL_PHASE_MAX` is relevant for wall clock time, while `EVAL_PHASE_SUM` is relevant
        for CPU time.
    """
    validate_time(all_time)

    time_summary = xr.Dataset(coords=all_time.coords)

    time_summary[cc.SUGGEST_PHASE] = all_time[cc.SUGGEST_PHASE]
    time_summary[cc.OBS_PHASE] = all_time[cc.OBS_PHASE]
    time_summary[cc.EVAL_PHASE_MAX] = all_time[cc.EVAL_PHASE].max(dim=SUGGEST)
    time_summary[cc.EVAL_PHASE_SUM] = all_time[cc.EVAL_PHASE].sum(dim=SUGGEST)
    return time_summary


def _ravel_perf(perf_da):
    validate_perf(perf_da)

    function_evals = np.ravel(perf_da.values, order="C")
    perf_ds = build_eval_ds(function_evals[:, None])
    perf_da = xru.only_dataarray(perf_ds)
    return perf_da


def _ravel_time(time_ds):
    validate_time(time_ds)

    n_iter, n_suggest = time_ds.sizes[ITER], time_ds.sizes[SUGGEST]

    def padder(X):
        Y = np.pad(X[:, None], [(0, 0), (0, n_suggest - 1)], mode="constant", constant_values=0)
        assert Y.shape == (n_iter, n_suggest)
        return Y

    # Unclear if this is the most sensible way to ravel time variables when comparing across batch sizes, but we don't
    # really use time in analysis. So, this is good enough for now.
    suggest_time = np.ravel(padder(time_ds[cc.SUGGEST_PHASE].values), order="C")
    eval_time = np.ravel(time_ds[cc.EVAL_PHASE].values, order="C")[:, None]
    observe_time = np.ravel(padder(time_ds[cc.OBS_PHASE].values), order="C")

    time_ds = build_timing_ds(suggest_time, eval_time, observe_time)
    return time_ds


def concat_experiments(all_experiments, ravel=False):
    """Aggregate the Datasets from a series of experiments into combined Dataset.

    Parameters
    ----------
    all_experiments : typing.Iterable
        Iterable (possible from a generator) with the Datasets from each experiment. Each item in `all_experiments` is
        a pair containing ``(meta_data, data)``. The `meta_data` contains a `tuple` of `str` with
        ``test_case, optimizer, uuid``. The `data` contains a tuple of ``(perf_da, time_ds, sig)``. The `perf_da` is an
        :class:`xarray:xarray.DataArray` containing the evaluation results with dimensions ``(ITER, SUGGEST)``. The
        `time_ds` is an :class:`xarray:xarray.Dataset` containing the timing results of the form accepted by
        `summarize_time`. The coordinates must be compatible with `perf_da`. Finally, `sig` contains the `test_case`
        signature and must be `list(float)`.
    ravel : bool
        If true, ravel all studies to store batch suggestions as if they were serial.

    Returns
    -------
    all_perf : :class:`xarray:xarray.DataArray`
        DataArray containing all of the `perf_da` from the experiments. The meta-data from the experiments are included
        as extra dimensions. `all_perf` has dimensions ``(ITER, SUGGEST, TEST_CASE, METHOD, TRIAL)``. To convert the
        `uuid` to a trial, there must be an equal number of repetition in the experiments for each `TEST_CASE`,
        `METHOD` combination. Likewise, all of the experiments need an equal number of `ITER` and `SUGGEST`. If `ravel`
        is true, then the `SUGGEST` is singleton.
    all_time : :class:`xarray:xarray.Dataset`
        Dataset containing all of the `time_ds` from the experiments. The new dimensions are
        ``(ITER, TEST_CASE, METHOD, TRIAL)``. It has the same variables as `time_ds`.
    all_sigs : dict(str, list(list(float)))
        Aggregate of all experiment signatures.
    """
    all_perf = {}
    all_time = {}
    all_sigs = {}
    trial_counter = Counter()
    for (test_case, optimizer, uuid), (perf_da, time_ds, sig) in all_experiments:
        if ravel:
            n_suggest = perf_da.sizes[SUGGEST]
            perf_da = _ravel_perf(perf_da)
            time_ds = _ravel_time(time_ds)
            optimizer = str_join_safe(ARG_DELIM, (optimizer, "p%d" % n_suggest), append=True)

        case_key = (test_case, optimizer, trial_counter[(test_case, optimizer)])
        trial_counter[(test_case, optimizer)] += 1

        # Process perf data
        assert perf_da.dims == (ITER, SUGGEST)
        all_perf[case_key] = perf_da

        # Process time data
        all_time[case_key] = summarize_time(time_ds)

        # Handle the signatures
        all_sigs.setdefault(test_case, []).append(sig)
    assert min(trial_counter.values()) == max(trial_counter.values()), "Uneven number of trials per test case"

    # Now need to concat dict of datasets into single dataset
    all_perf = xru.da_concat(all_perf, dims=(TEST_CASE, METHOD, TRIAL))
    assert all_perf.dims == (ITER, SUGGEST, TEST_CASE, METHOD, TRIAL)
    assert not np.any(np.isnan(all_perf.values)), "Missing combinations of method and test case"

    all_time = xru.ds_concat(all_time, dims=(TEST_CASE, METHOD, TRIAL))
    assert all(all_time[kk].dims == (ITER, TEST_CASE, METHOD, TRIAL) for kk in all_time)
    assert not any(np.any(np.isnan(all_time[kk].values)) for kk in all_time)
    assert xru.coord_compat((all_perf, all_time), (ITER, TEST_CASE, METHOD, TRIAL))

    return all_perf, all_time, all_sigs


def load_experiments(uuid_list, db_root, dbid):  # pragma: io
    """Generator to load the results of the experiments.

    Parameters
    ----------
    uuid_list : list(uuid.UUID)
        List of UUIDs corresponding to experiments to load.
    db_root : str
        Root location for data store as requested by the serializer used.
    dbid : str
        Name of the data store as requested by the serializer used.

    Yields
    ------
    meta_data : (str, str, str)
        The `meta_data` contains a `tuple` of `str` with ``test_case, optimizer, uuid``.
    data : (:class:`xarray:xarray.DataArray`, :class:`xarray:xarray.Dataset`, list(float))
        The `data` contains a tuple of ``(perf_da, time_ds, sig)``. The `perf_da` is an
        :class:`xarray:xarray.DataArray` containing the evaluation results with dimensions ``(ITER, SUGGEST)``. The
        `time_ds` is an :class:`xarray:xarray.Dataset` containing the timing results of the form accepted by
        `summarize_time`. The coordinates must be compatible with `perf_da`. Finally, `sig` contains the `test_case`
        signature and must be `list(float)`.
    """
    uuids_seen = set()
    for uuid_ in uuid_list:
        logger.info(uuid_.hex)

        # Load perf and timing data
        perf_da, meta = XRSerializer.load(db_root, db=dbid, key=cc.EVAL, uuid_=uuid_)
        perf_da = xru.only_dataarray(perf_da)
        time_ds, meta_t = XRSerializer.load(db_root, db=dbid, key=cc.TIME, uuid_=uuid_)
        assert meta == meta_t, "meta data should between time and eval files"

        # Get signature to pass out as well
        _, sig = meta["signature"]
        logger.info(meta)
        logger.info(sig)

        # Build the new indices for combined data, this could be put in function for easier testing
        eval_args = unserializable_dict(meta["args"])  # Unpack meta-data
        test_case = str_join_safe(
            ARG_DELIM, (eval_args[CmdArgs.classifier], eval_args[CmdArgs.data], eval_args[CmdArgs.metric])
        )
        optimizer = str_join_safe(
            ARG_DELIM, (eval_args[CmdArgs.optimizer], eval_args[CmdArgs.opt_rev], eval_args[CmdArgs.rev])
        )
        args_uuid = eval_args[CmdArgs.uuid]

        # Check UUID sanity
        assert isinstance(args_uuid, str)
        assert args_uuid == uuid_.hex, "UUID meta-data does not match filename"
        assert args_uuid not in uuids_seen, "uuids being reused between studies"
        uuids_seen.add(args_uuid)

        # Return key -> data so this generator can be iterated over in dict like manner
        meta_data = (test_case, optimizer, args_uuid)
        data = (perf_da, time_ds, sig)
        yield meta_data, data


def main():
    """See README for instructions on calling aggregate.
    """
    description = "Aggregate study results across functions and optimizers"
    args = parse_args(agg_parser(description))

    logger.setLevel(logging.INFO)  # Note this is the module-wide logger
    if args[CmdArgs.verbose]:
        logger.addHandler(logging.StreamHandler())

    # Always sort after listdir
    uuid_list = XRSerializer.get_uuids(args[CmdArgs.db_root], db=args[CmdArgs.db], key=cc.EVAL)
    uuid_list_ = XRSerializer.get_uuids(args[CmdArgs.db_root], db=args[CmdArgs.db], key=cc.TIME)
    assert uuid_list == uuid_list_, "UUID list does not match between time and eval results"

    # Get iterator of all experiment data dumps, load in and process, and concat
    data_G = load_experiments(uuid_list, args[CmdArgs.db_root], args[CmdArgs.db])
    all_perf, all_time, all_sigs = concat_experiments(data_G, ravel=args[CmdArgs.ravel])

    # Check the concat signatures make are coherent
    sig_errs, signatures_median = analyze_signatures(all_sigs)
    logger.info("Signature errors:\n%s" % sig_errs.to_string())

    # Dump and save it all out
    logger.info("saving")
    meta = {"args": serializable_dict(args), "signature": signatures_median}
    all_perf = all_perf.to_dataset(name="results")
    XRSerializer.save_derived(all_perf, meta, args[CmdArgs.db_root], db=args[CmdArgs.db], key=EVAL_RESULTS)
    XRSerializer.save_derived(all_time, meta, args[CmdArgs.db_root], db=args[CmdArgs.db], key=TIME_RESULTS)

    logger.info("done")


if __name__ == "__main__":
    main()  # pragma: main
