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
"""Perform a study.
"""
import json
import logging
import random as pyrandom
import uuid
import warnings
from collections import OrderedDict
from time import time

import numpy as np
import xarray as xr

import bayesmark.cmd_parse as cmd
import bayesmark.constants as cc
import bayesmark.random_search as rs
from bayesmark.builtin_opt.config import CONFIG
from bayesmark.cmd_parse import CmdArgs
from bayesmark.constants import ARG_DELIM, ITER, OBJECTIVE, SUGGEST
from bayesmark.data import METRICS_LOOKUP, get_problem_type
from bayesmark.np_util import argmin_2d, linear_rescale, random_seed
from bayesmark.serialize import XRSerializer
from bayesmark.signatures import analyze_signature_pair, get_func_signature
from bayesmark.sklearn_funcs import SklearnModel, SklearnSurrogate
from bayesmark.space import JointSpace
from bayesmark.util import chomp, str_join_safe

logger = logging.getLogger(__name__)

# For now treat the objective names as global const. However, in the future these could vary by type of problem.
OBJECTIVE_NAMES = SklearnModel.objective_names


def _build_test_problem(model_name, dataset, scorer, path):
    """Build the class with the class to use an objective. Sort of a factory.

    Parameters
    ----------
    model_name : str
        Which sklearn model we are attempting to tune, must be an element of `constants.MODEL_NAMES`.
    dataset : str
        Which data set the model is being tuned to, which must be either a) an element of
        `constants.DATA_LOADER_NAMES`, or b) the name of a csv file in the `data_root` folder for a custom data set.
    scorer : str
        Which metric to use when evaluating the model. This must be an element of `sklearn_funcs.SCORERS_CLF` for
        classification models, or `sklearn_funcs.SCORERS_REG` for regression models.
    path : str or None
        Absolute path to folder containing custom data sets/pickle files with surrogate model.

    Returns
    -------
    prob : :class:`.sklearn_funcs.TestFunction`
        The test function to evaluate in experiments.
    """
    if model_name.endswith("-surr"):
        # Requires IO to test these, so will add the pargma here. Maybe that points towards a possible design change.
        model_name = chomp(model_name, "-surr")  # pragma: io
        prob = SklearnSurrogate(model_name, dataset, scorer, path=path)  # pragma: io
    else:
        prob = SklearnModel(model_name, dataset, scorer, data_root=path)
    return prob


def run_study(optimizer, test_problem, n_calls, n_suggestions, n_obj=1, callback=None):
    """Run a study for a single optimizer on a single test problem.

    This function can be used for benchmarking on general stateless objectives (not just `sklearn`).

    Parameters
    ----------
    optimizer : :class:`.abstract_optimizer.AbstractOptimizer`
        Instance of one of the wrapper optimizers.
    test_problem : :class:`.sklearn_funcs.TestFunction`
        Instance of test function to attempt to minimize.
    n_calls : int
        How many iterations of minimization to run.
    n_suggestions : int
        How many parallel evaluation we run each iteration. Must be ``>= 1``.
    n_obj : int
        Number of different objectives measured, only objective 0 is seen by optimizer. Must be ``>= 1``.
    callback : callable
        Optional callback taking the current best function evaluation, and the number of iterations finished. Takes
        array of shape `(n_obj,)`.

    Returns
    -------
    function_evals : :class:`numpy:numpy.ndarray` of shape (n_calls, n_suggestions, n_obj)
        Value of objective for each evaluation.
    timing_evals : (:class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`)
        Tuple of 3 timing results: ``(suggest_time, eval_time, observe_time)`` with shapes ``(n_calls,)``,
        ``(n_calls, n_suggestions)``, and ``(n_calls,)``. These are the time to make each suggestion, the time for each
        evaluation of the objective function, and the time to make an observe call.
    suggest_log : list(list(dict(str, object)))
        Log of the suggestions corresponding to the `function_evals`.
    """
    assert n_suggestions >= 1, "batch size must be at least 1"
    assert n_obj >= 1, "Must be at least one objective"

    space_for_validate = JointSpace(test_problem.get_api_config())

    if callback is not None:
        # First do initial log at inf score, in case we don't even get to first eval before crash/job timeout
        callback(np.full((n_obj,), np.inf, dtype=float), 0)

    suggest_time = np.zeros(n_calls)
    observe_time = np.zeros(n_calls)
    eval_time = np.zeros((n_calls, n_suggestions))
    function_evals = np.zeros((n_calls, n_suggestions, n_obj))
    suggest_log = [None] * n_calls
    for ii in range(n_calls):
        tt = time()
        try:
            next_points = optimizer.suggest(n_suggestions)
        except Exception as e:
            logger.warning("Failure in optimizer suggest. Falling back to random search.")
            logger.exception(e, exc_info=True)
            print(json.dumps({"optimizer_suggest_exception": {ITER: ii}}))
            api_config = test_problem.get_api_config()
            next_points = rs.suggest_dict([], [], api_config, n_suggestions=n_suggestions)
        suggest_time[ii] = time() - tt

        logger.info("suggestion time taken %f iter %d next_points %s" % (suggest_time[ii], ii, str(next_points)))
        assert len(next_points) == n_suggestions, "invalid number of suggestions provided by the optimizer"

        # We could put this inside the TestProblem class, but ok here for now.
        try:
            space_for_validate.validate(next_points)  # Fails if suggestions outside allowed range
        except Exception:
            raise ValueError("Optimizer suggestion is out of range.")

        for jj, next_point in enumerate(next_points):
            tt = time()
            try:
                f_current_eval = test_problem.evaluate(next_point)
            except Exception as e:
                logger.warning("Failure in function eval. Setting to inf.")
                logger.exception(e, exc_info=True)
                f_current_eval = np.full((n_obj,), np.inf, dtype=float)
            eval_time[ii, jj] = time() - tt
            assert np.shape(f_current_eval) == (n_obj,)

            suggest_log[ii] = next_points
            function_evals[ii, jj, :] = f_current_eval
            logger.info(
                "function_evaluation time %f value %f suggestion %s"
                % (eval_time[ii, jj], f_current_eval[0], str(next_point))
            )

        # Note: this could be inf in the event of a crash in f evaluation, the optimizer must be able to handle that.
        # Only objective 0 is seen by optimizer.
        eval_list = function_evals[ii, :, 0].tolist()

        if callback is not None:
            idx_ii, idx_jj = argmin_2d(function_evals[: ii + 1, :, 0])
            callback(function_evals[idx_ii, idx_jj, :], ii + 1)

        tt = time()
        try:
            optimizer.observe(next_points, eval_list)
        except Exception as e:
            logger.warning("Failure in optimizer observe. Ignoring these observations.")
            logger.exception(e, exc_info=True)
            print(json.dumps({"optimizer_observe_exception": {ITER: ii}}))
        observe_time[ii] = time() - tt

        logger.info(
            "observation time %f, current best %f at iter %d"
            % (observe_time[ii], np.min(function_evals[: ii + 1, :, 0]), ii)
        )

    return function_evals, (suggest_time, eval_time, observe_time), suggest_log


def run_sklearn_study(
    opt_class, opt_kwargs, model_name, dataset, scorer, n_calls, n_suggestions, data_root=None, callback=None
):
    """Run a study for a single optimizer on a single `sklearn` model/data set combination.

    This routine is meant for benchmarking when tuning `sklearn` models, as opposed to the more general
    :func:`.run_study`.

    Parameters
    ----------
    opt_class : :class:`.abstract_optimizer.AbstractOptimizer`
        Type of wrapper optimizer must be subclass of :class:`.abstract_optimizer.AbstractOptimizer`.
    opt_kwargs : kwargs
        `kwargs` to use when instantiating the wrapper class.
    model_name : str
        Which sklearn model we are attempting to tune, must be an element of `constants.MODEL_NAMES`.
    dataset : str
        Which data set the model is being tuned to, which must be either a) an element of
        `constants.DATA_LOADER_NAMES`, or b) the name of a csv file in the `data_root` folder for a custom data set.
    scorer : str
        Which metric to use when evaluating the model. This must be an element of `sklearn_funcs.SCORERS_CLF` for
        classification models, or `sklearn_funcs.SCORERS_REG` for regression models.
    n_calls : int
        How many iterations of minimization to run.
    n_suggestions : int
        How many parallel evaluation we run each iteration. Must be ``>= 1``.
    data_root : str
        Absolute path to folder containing custom data sets. This may be ``None`` if no custom data sets are used.``
    callback : callable
        Optional callback taking the current best function evaluation, and the number of iterations finished. Takes
        array of shape `(n_obj,)`.

    Returns
    -------
    function_evals : :class:`numpy:numpy.ndarray` of shape (n_calls, n_suggestions, n_obj)
        Value of objective for each evaluation.
    timing_evals : (:class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`)
        Tuple of 3 timing results: ``(suggest_time, eval_time, observe_time)`` with shapes ``(n_calls,)``,
        ``(n_calls, n_suggestions)``, and ``(n_calls,)``. These are the time to make each suggestion, the time for each
        evaluation of the objective function, and the time to make an observe call.
    suggest_log : list(list(dict(str, object)))
        Log of the suggestions corresponding to the `function_evals`.
    """
    # Setup test function
    function_instance = _build_test_problem(model_name, dataset, scorer, data_root)

    # Setup optimizer
    api_config = function_instance.get_api_config()
    optimizer_instance = opt_class(api_config, **opt_kwargs)

    assert function_instance.objective_names == OBJECTIVE_NAMES
    assert OBJECTIVE_NAMES[0] == cc.VISIBLE_TO_OPT
    n_obj = len(OBJECTIVE_NAMES)

    # Now actually do the experiment
    function_evals, timing, suggest_log = run_study(
        optimizer_instance, function_instance, n_calls, n_suggestions, n_obj=n_obj, callback=callback
    )
    return function_evals, timing, suggest_log


def get_objective_signature(model_name, dataset, scorer, data_root=None):
    """Get signature of an objective function specified by an sklearn model and dataset.

    This routine specializes :func:`.signatures.get_func_signature` for the `sklearn` study case.

    Parameters
    ----------
    model_name : str
        Which sklearn model we are attempting to tune, must be an element of `constants.MODEL_NAMES`.
    dataset : str
        Which data set the model is being tuned to, which must be either a) an element of
        `constants.DATA_LOADER_NAMES`, or b) the name of a csv file in the `data_root` folder for a custom data set.
    scorer : str
        Which metric to use when evaluating the model. This must be an element of `sklearn_funcs.SCORERS_CLF` for
        classification models, or `sklearn_funcs.SCORERS_REG` for regression models.
    data_root : str
        Absolute path to folder containing custom data sets. This may be ``None`` if no custom data sets are used.``

    Returns
    -------
    signature : list(str)
        The signature of this test function.
    """
    function_instance = _build_test_problem(model_name, dataset, scorer, data_root)
    api_config = function_instance.get_api_config()
    signature = get_func_signature(function_instance.evaluate, api_config)
    return signature


def build_eval_ds(function_evals, objective_names):
    """Convert :class:`numpy:numpy.ndarray` with function evaluations to :class:`xarray:xarray.Dataset`.

    This function is a data cleanup routine after running an experiment, before serializing the data to end the study.

    Parameters
    ----------
    function_evals : :class:`numpy:numpy.ndarray` of shape (n_calls, n_suggestions, n_obj)
        Value of objective for each evaluation.
    objective_names : list(str) of shape (n_obj,)
        The names of each objective.

    Returns
    -------
    eval_ds : :class:`xarray:xarray.Dataset`
        :class:`xarray:xarray.Dataset` containing one variable for each objective with the objective function
        evaluations. It has dimensions ``(ITER, SUGGEST)``.
    """
    n_call, n_suggest, n_obj = np.shape(function_evals)
    assert len(objective_names) == n_obj
    assert len(set(objective_names)) == n_obj, "Objective names must be unique"

    coords = {ITER: range(n_call), SUGGEST: range(n_suggest), OBJECTIVE: list(objective_names)}
    dims = (ITER, SUGGEST, OBJECTIVE)
    da = xr.DataArray(data=function_evals, coords=coords, dims=dims)
    eval_ds = da.to_dataset(dim=OBJECTIVE)
    return eval_ds


def build_timing_ds(suggest_time, eval_time, observe_time):
    """Convert :class:`numpy:numpy.ndarray` with timing evaluations to :class:`xarray:xarray.Dataset`.

    This function is a data cleanup routine after running an experiment, before serializing the data to end the study.

    Parameters
    ----------
    suggest_time : :class:`numpy:numpy.ndarray` of shape (n_calls,)
        The time to make each (batch) suggestion.
    eval_time : :class:`numpy:numpy.ndarray` of shape (n_calls, n_suggestions)
        The time for each evaluation of the objective function.
    observe_time : :class:`numpy:numpy.ndarray` of shape (n_calls,)
        The time for each (batch) evaluation of the objective function, and the time to make an observe call.

    Returns
    -------
    time_ds : :class:`xarray:xarray.Dataset`
        Dataset with variables ``(SUGGEST_PHASE, EVAL_PHASE, OBS_PHASE)`` which have dimensions ``(ITER,)``,
        ``(ITER, SUGGEST)``, and ``(ITER,)``, respectively. The variable `EVAL_PHASE` has the function evaluation time
        for each parallel suggestion.
    """
    n_call, n_suggest = np.shape(eval_time)
    assert np.shape(suggest_time) == (n_call,)
    assert np.shape(observe_time) == (n_call,)

    coords = OrderedDict([(ITER, range(n_call)), (SUGGEST, range(n_suggest))])

    data = OrderedDict()
    data[cc.SUGGEST_PHASE] = ((ITER,), suggest_time)
    data[cc.EVAL_PHASE] = ((ITER, SUGGEST), eval_time)
    data[cc.OBS_PHASE] = ((ITER,), observe_time)

    time_ds = xr.Dataset(data, coords=coords)
    return time_ds


def build_suggest_ds(suggest_log):
    """Convert :class:`numpy:numpy.ndarray` with function evaluation inputs to :class:`xarray:xarray.Dataset`.

    This function is a data cleanup routine after running an experiment, before serializing the data to end the study.

    Parameters
    ----------
    suggest_log : list(list(dict(str, object)))
        Log of the suggestions. It has shape `(n_call, n_suggest)`.

    Returns
    -------
    suggest_ds : :class:`xarray:xarray.Dataset`
        :class:`xarray:xarray.Dataset` containing one variable for each input with the objective function evaluations.
        It has dimensions ``(ITER, SUGGEST)``.
    """
    n_call, n_suggest = np.shape(suggest_log)
    assert n_call * n_suggest > 0

    # Setup the dims
    ds_vars = sorted(suggest_log[0][0].keys())
    coords = OrderedDict([(ITER, range(n_call)), (SUGGEST, range(n_suggest))])

    # There is prob a way to vectorize this more but good enough for now. Using np.full to infer dtype from 1st element
    data = OrderedDict([(kk, ((ITER, SUGGEST), np.full((n_call, n_suggest), suggest_log[0][0][kk]))) for kk in ds_vars])
    for ii in range(n_call):
        for jj in range(n_suggest):
            for kk in ds_vars:
                data[kk][1][ii, jj] = suggest_log[ii][jj][kk]

    suggest_ds = xr.Dataset(data, coords=coords)
    return suggest_ds


def load_optimizer_kwargs(optimizer_name, opt_root):  # pragma: io
    """Load the kwarg options for this optimizer being tested.

    This is part of the general experiment setup before a study.

    Parameters
    ----------
    optimizer_name : str
        Name of the optimizer being tested. This optimizer name must be present in optimizer config file.
    opt_root : str
        Absolute path to folder containing the config file.

    Returns
    -------
    kwargs : dict(str, object)
        The kwargs setting to pass into the optimizer wrapper constructor.
    """
    if optimizer_name in CONFIG:
        _, kwargs = CONFIG[optimizer_name]
    else:
        settings = cmd.load_optimizer_settings(opt_root)
        assert optimizer_name in settings, "optimizer %s not found in settings file %s" % optimizer_name
        _, kwargs = settings[optimizer_name]
    return kwargs


def _setup_seeds(hex_str):  # pragma: main
    """This function should only be called from main. Be careful with this function as it manipulates the global random
    streams.

    This is part of the general experiment setup before a study.

    If torch becomes used in any of our optimizers then this will need to come back, could also do TF seed init.
    ```
    torch.manual_seed(random_seed(master_stream))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed(master_stream))
    ```
    """
    # Set all random seeds: avoid correlated streams ==> must use diff seeds.
    # Could use UUID class, but more direct to just convert the hex to py int.
    # pyrandom is better for master because it is not limited to 32-bit seeds.
    master_stream = pyrandom.Random(int(hex_str, 16))
    pyrandom.seed(random_seed(master_stream))
    np.random.seed(random_seed(master_stream))


def experiment_main(opt_class, args=None):  # pragma: main
    """This is in effect the `main` routine for this experiment. However, it is called from the optimizer wrapper file
    so the class can be passed in. The optimizers are assumed to be outside the package, so the optimizer class can't
    be named from inside the main function without using hacky stuff like `eval`.
    """
    if args is None:
        description = "Run a study with one benchmark function and an optimizer"
        args = cmd.parse_args(cmd.experiment_parser(description))
    args[CmdArgs.opt_rev] = opt_class.get_version()

    run_uuid = uuid.UUID(args[CmdArgs.uuid])

    logging.captureWarnings(True)

    # Setup logging to both a file and stdout (if verbose is set to True)
    logger.setLevel(logging.INFO)  # Note this is the module-wide logger
    logfile = XRSerializer.logging_path(args[CmdArgs.db_root], args[CmdArgs.db], run_uuid)
    logger_file_handler = logging.FileHandler(logfile, mode="w")
    logger.addHandler(logger_file_handler)
    if args[CmdArgs.verbose]:
        logger.addHandler(logging.StreamHandler())

    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.addHandler(logger_file_handler)
    if args[CmdArgs.verbose]:
        warnings_logger.addHandler(logging.StreamHandler())

    logger.info("running: %s" % str(cmd.serializable_dict(args)))
    logger.info("cmd: %s" % cmd.cmd_str())

    assert (
        args[CmdArgs.metric] in METRICS_LOOKUP[get_problem_type(args[CmdArgs.data])]
    ), "reg/clf metrics can only be used on compatible dataset"

    # Setup random streams for computing the signature, must use same seed
    # across all runs to ensure signature is consistent. This seed is random:
    _setup_seeds("7e9f2cabb0dd4f44bc10cf18e440b427")  # pragma: allowlist secret
    signature = get_objective_signature(
        args[CmdArgs.classifier], args[CmdArgs.data], args[CmdArgs.metric], data_root=args[CmdArgs.data_root]
    )
    logger.info("computed signature: %s" % str(signature))

    opt_kwargs = load_optimizer_kwargs(args[CmdArgs.optimizer], args[CmdArgs.optimizer_root])

    # Setup the call back for intermediate logging
    if cc.BASELINE not in XRSerializer.get_derived_keys(args[CmdArgs.db_root], db=args[CmdArgs.db]):
        warnings.warn("Baselines not found. Will not log intermediate scores.")
        callback = None
    else:
        test_case_str = SklearnModel.test_case_str(args[CmdArgs.classifier], args[CmdArgs.data], args[CmdArgs.metric])
        optimizer_str = str_join_safe(ARG_DELIM, (args[CmdArgs.optimizer], args[CmdArgs.opt_rev], args[CmdArgs.rev]))

        baseline_ds, baselines_meta = XRSerializer.load_derived(
            args[CmdArgs.db_root], db=args[CmdArgs.db], key=cc.BASELINE
        )

        # Check the objective function signatures match in the baseline file
        sig_errs, _ = analyze_signature_pair({test_case_str: signature[1]}, baselines_meta["signature"])
        logger.info("Signature errors:\n%s" % sig_errs.to_string())
        print(json.dumps({"exp sig errors": sig_errs.T.to_dict()}))

        def log_mean_score_json(evals, iters):
            assert evals.shape == (len(OBJECTIVE_NAMES),)
            assert not np.any(np.isnan(evals))

            log_msg = {
                cc.TEST_CASE: test_case_str,
                cc.METHOD: optimizer_str,
                cc.TRIAL: args[CmdArgs.uuid],
                cc.ITER: iters,
            }

            for idx, obj in enumerate(OBJECTIVE_NAMES):
                assert OBJECTIVE_NAMES[idx] == obj

                # Extract relevant rescaling info
                slice_ = {cc.TEST_CASE: test_case_str, OBJECTIVE: obj}
                best_opt = baseline_ds[cc.PERF_BEST].sel(slice_, drop=True).values.item()
                base_clip_val = baseline_ds[cc.PERF_CLIP].sel(slice_, drop=True).values.item()

                # Perform the same rescaling as found in experiment_analysis.compute_aggregates()
                score = linear_rescale(evals[idx], best_opt, base_clip_val, 0.0, 1.0, enforce_bounds=False)
                # Also, clip the score from below at -1 to limit max influence of single run on final average
                score = np.clip(score, -1.0, 1.0)
                score = score.item()  # Make easiest for logging in JSON
                assert isinstance(score, float)

                # Note: This is not the raw score but the rescaled one!
                log_msg[obj] = score
            log_msg = json.dumps(log_msg)
            print(log_msg)

        callback = log_mean_score_json

    # Now set the seeds for the actual experiment
    _setup_seeds(args[CmdArgs.uuid])

    # Now do the experiment
    logger.info(
        "starting sklearn study %s %s %s %s %d %d"
        % (
            args[CmdArgs.optimizer],
            args[CmdArgs.classifier],
            args[CmdArgs.data],
            args[CmdArgs.metric],
            args[CmdArgs.n_calls],
            args[CmdArgs.n_suggest],
        )
    )
    logger.info("with data root: %s" % args[CmdArgs.data_root])
    function_evals, timing, suggest_log = run_sklearn_study(
        opt_class,
        opt_kwargs,
        args[CmdArgs.classifier],
        args[CmdArgs.data],
        args[CmdArgs.metric],
        args[CmdArgs.n_calls],
        args[CmdArgs.n_suggest],
        data_root=args[CmdArgs.data_root],
        callback=callback,
    )

    # Curate results into clean dataframes
    eval_ds = build_eval_ds(function_evals, OBJECTIVE_NAMES)
    time_ds = build_timing_ds(*timing)
    suggest_ds = build_suggest_ds(suggest_log)

    # setup meta:
    meta = {"args": cmd.serializable_dict(args), "signature": signature}
    logger.info("saving meta data: %s" % str(meta))

    # Now the final IO to export the results
    logger.info("saving results")
    XRSerializer.save(eval_ds, meta, args[CmdArgs.db_root], db=args[CmdArgs.db], key=cc.EVAL, uuid_=run_uuid)

    logger.info("saving timing")
    XRSerializer.save(time_ds, meta, args[CmdArgs.db_root], db=args[CmdArgs.db], key=cc.TIME, uuid_=run_uuid)

    logger.info("saving suggest log")
    XRSerializer.save(suggest_ds, meta, args[CmdArgs.db_root], db=args[CmdArgs.db], key=cc.SUGGEST_LOG, uuid_=run_uuid)

    logger.info("done")


def _get_opt_class(opt_name):
    """Load the relevant wrapper class based on this optimizer name.

    There is inherently a bit ugly, but is only called at the main() level before the inner workings get going. There
    are a few ways to do this with some pro and con:
    1) The way done here: based on the filename, load that module via conditional imports and if-else. cons:
        - uses conditional imports
        - must manually repeat yourself in the if-else, but these are checked in unit testing
    2) Import everything and then pick the right optimizer based on a dict of name_str -> class. cons:
        - loads every dependency no matter which is used so could be slow
        - also a stupid dependency might change global state in a way that corrupts experiments
    3) Use the wrapper file as the entry point and add that to setup.py. cons:
        - Will clutter the CLI namespace with one command for each wrapper
    4) Use importlib to import the specified file. cons:
        - Makes assumptions about relative path structure. For pip-installed packages, probably safer to let python
        find the file via import.
    This option (1) seems least objectionable. However, this function could easily be switched to use importlib without
    any changes elsewhere.
    """
    wrapper_file, _ = CONFIG[opt_name]

    if wrapper_file == "hyperopt_optimizer.py":
        import bayesmark.builtin_opt.hyperopt_optimizer as opt
    elif wrapper_file == "nevergrad_optimizer.py":
        import bayesmark.builtin_opt.nevergrad_optimizer as opt
    elif wrapper_file == "opentuner_optimizer.py":
        import bayesmark.builtin_opt.opentuner_optimizer as opt
    elif wrapper_file == "pysot_optimizer.py":
        import bayesmark.builtin_opt.pysot_optimizer as opt
    elif wrapper_file == "random_optimizer.py":
        import bayesmark.builtin_opt.random_optimizer as opt
    elif wrapper_file == "scikit_optimizer.py":
        import bayesmark.builtin_opt.scikit_optimizer as opt
    else:
        assert False, "CONFIG for built in optimizers has added a new optimizer, but not updated this function."

    opt_class = opt.opt_wrapper
    return opt_class


def main():  # pragma: main
    """This is where experiments happen. Usually called by the experiment launcher."""
    description = "Run a study with one benchmark function and an optimizer"
    args = cmd.parse_args(cmd.experiment_parser(description))

    opt_class = _get_opt_class(args[CmdArgs.optimizer])
    experiment_main(opt_class, args=args)


if __name__ == "__main__":
    main()  # pragma: main
