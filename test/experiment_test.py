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
import inspect
import os.path

import numpy as np
from hypothesis import assume, given, settings
from hypothesis.strategies import floats, integers, sampled_from, text
from hypothesis_gufunc.extra.xr import simple_datasets
from hypothesis_gufunc.gufunc import gufunc_args

import bayesmark.experiment as exp
import bayesmark.random_search as rs
from bayesmark import data, np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.builtin_opt.config import CONFIG
from bayesmark.constants import DATA_LOADER_NAMES, ITER, METRICS, MODEL_NAMES, SUGGEST
from bayesmark.sklearn_funcs import SklearnModel, TestFunction
from hypothesis_util import seeds
from util import space_configs


class RandomOptimizer(AbstractOptimizer):
    # Unclear what is best package to list for primary_import here.
    primary_import = "bayesmark"

    def __init__(self, api_config, random=np_util.random, flaky=False):
        AbstractOptimizer.__init__(self, api_config)
        self.random = random
        self.flaky = flaky

    def suggest(self, n_suggestions=1):
        if self.flaky:
            assert self.random.rand() <= 0.5
        x_guess = rs.suggest_dict([], [], self.api_config, n_suggestions=n_suggestions, random=self.random)
        return x_guess

    def observe(self, X, y):
        # Random search so don't do anything for observe
        if self.flaky:
            assert self.random.rand() <= 0.5


class OutOfBoundsOptimizer(AbstractOptimizer):
    def __init__(self, api_config, random=np_util.random):
        AbstractOptimizer.__init__(self, api_config)
        self.random = random
        self.param_list = sorted([kk for kk in api_config.keys() if api_config[kk]["type"] in ("real", "int")])

    def suggest(self, n_suggestions=1):
        x_guess = rs.suggest_dict([], [], self.api_config, n_suggestions=n_suggestions, random=self.random)

        ii = self.random.randint(0, n_suggestions)
        pp = self.random.choice(self.param_list)

        if self.api_config[pp]["type"] == "real":
            eps = self.random.rand()
        else:
            eps = self.random.randint(1, 10)

        if self.random.rand() <= 0.5:
            x_guess[ii][pp] = self.api_config[pp]["range"][0] - eps
        else:
            x_guess[ii][pp] = self.api_config[pp]["range"][1] + eps
        return x_guess

    def observe(self, X, y):
        pass


class FlakyProblem(TestFunction):
    def __init__(self, api_config, random):
        TestFunction.__init__(self)
        self.api_config = api_config
        self.random = random

    def evaluate(self, params):
        assert self.random.rand() <= 0.5
        return [0.0]


@given(
    sampled_from(MODEL_NAMES),
    sampled_from(DATA_LOADER_NAMES),
    sampled_from(METRICS),
    integers(0, 5),
    integers(1, 3),
    seeds(),
)
@settings(max_examples=10, deadline=None)
def test_run_study(model_name, dataset, scorer, n_calls, n_suggestions, seed):
    prob_type = data.get_problem_type(dataset)
    assume(scorer in data.METRICS_LOOKUP[prob_type])

    function_instance = SklearnModel(model_name, dataset, scorer)
    optimizer = RandomOptimizer(function_instance.get_api_config(), random=np.random.RandomState(seed))
    optimizer.get_version()
    exp.run_study(optimizer, function_instance, n_calls, n_suggestions, n_obj=len(function_instance.objective_names))


@given(
    sampled_from(MODEL_NAMES),
    sampled_from(DATA_LOADER_NAMES),
    sampled_from(METRICS),
    integers(1, 5),
    integers(1, 3),
    seeds(),
)
def test_run_study_bounds_fail(model_name, dataset, scorer, n_calls, n_suggestions, seed):
    prob_type = data.get_problem_type(dataset)
    assume(scorer in data.METRICS_LOOKUP[prob_type])

    function_instance = SklearnModel(model_name, dataset, scorer)
    optimizer = OutOfBoundsOptimizer(function_instance.get_api_config(), random=np.random.RandomState(seed))
    optimizer.get_version()

    # pytest have some assert failed tools we could use instead, but this is ok for now
    bounds_fails = False
    try:
        exp.run_study(
            optimizer, function_instance, n_calls, n_suggestions, n_obj=len(function_instance.objective_names)
        )
    except Exception as e:
        bounds_fails = str(e) == "Optimizer suggestion is out of range."
    assert bounds_fails


@given(
    sampled_from(MODEL_NAMES),
    sampled_from(DATA_LOADER_NAMES),
    sampled_from(METRICS),
    integers(0, 5),
    integers(1, 3),
    seeds(),
)
@settings(max_examples=10, deadline=None)
def test_run_study_callback(model_name, dataset, scorer, n_calls, n_suggestions, seed):
    prob_type = data.get_problem_type(dataset)
    assume(scorer in data.METRICS_LOOKUP[prob_type])

    function_instance = SklearnModel(model_name, dataset, scorer)
    optimizer = RandomOptimizer(function_instance.get_api_config(), random=np.random.RandomState(seed))
    optimizer.get_version()
    n_obj = len(function_instance.objective_names)

    function_evals_cmin = np.zeros((n_calls, n_obj), dtype=float)
    iters_list = []

    def callback(f_min, iters):
        assert f_min.shape == (n_obj,)

        iters_list.append(iters)
        if iters == 0:
            assert np.all(f_min == np.inf)
            return

        function_evals_cmin[iters - 1, :] = f_min

    function_evals, _, _ = exp.run_study(
        optimizer, function_instance, n_calls, n_suggestions, n_obj=n_obj, callback=callback
    )

    assert iters_list == list(range(n_calls + 1))

    for ii in range(n_obj):
        for jj in range(n_calls):
            idx0, idx1 = np_util.argmin_2d(function_evals[: jj + 1, :, 0])
            assert function_evals_cmin[jj, ii] == function_evals[idx0, idx1, ii]


@given(space_configs(allow_missing=True), integers(0, 5), integers(1, 3), seeds(), seeds())
@settings(deadline=None)
def test_run_study_flaky(api_config, n_calls, n_suggestions, seed1, seed2):
    api_config, _, _, _ = api_config

    function_instance = FlakyProblem(api_config=api_config, random=np.random.RandomState(seed1))
    optimizer = RandomOptimizer(api_config, random=np.random.RandomState(seed2), flaky=True)
    optimizer.get_version()
    exp.run_study(optimizer, function_instance, n_calls, n_suggestions)


@given(
    space_configs(allow_missing=True),
    sampled_from(MODEL_NAMES),
    sampled_from(DATA_LOADER_NAMES),
    sampled_from(METRICS),
    integers(0, 5),
    integers(1, 3),
    seeds(),
)
@settings(max_examples=10, deadline=None)
def test_run_sklearn_study(api_config, model_name, dataset, scorer, n_calls, n_suggestions, seed):
    prob_type = data.get_problem_type(dataset)
    assume(scorer in data.METRICS_LOOKUP[prob_type])

    random = np.random.RandomState(seed)
    exp.run_sklearn_study(RandomOptimizer, {"random": random}, model_name, dataset, scorer, n_calls, n_suggestions)


@given(
    space_configs(allow_missing=True),
    sampled_from(MODEL_NAMES),
    sampled_from(DATA_LOADER_NAMES),
    sampled_from(METRICS),
    integers(0, 5),
    integers(1, 3),
)
@settings(max_examples=10, deadline=None)
def test_run_sklearn_study_real(api_config, model_name, dataset, scorer, n_calls, n_suggestions):
    prob_type = data.get_problem_type(dataset)
    assume(scorer in data.METRICS_LOOKUP[prob_type])

    # Should really do parametric test but for loop good enough
    for opt_name in sorted(CONFIG.keys()):
        opt_class = exp._get_opt_class(opt_name)
        # opt_root=None should work with built-in opt
        opt_kwargs = exp.load_optimizer_kwargs(opt_name, opt_root=None)

        exp.run_sklearn_study(opt_class, opt_kwargs, model_name, dataset, scorer, n_calls, n_suggestions)


@given(sampled_from(MODEL_NAMES), sampled_from(DATA_LOADER_NAMES), sampled_from(METRICS))
@settings(deadline=None)
def test_get_objective_signature(model_name, dataset, scorer):
    prob_type = data.get_problem_type(dataset)
    assume(scorer in data.METRICS_LOOKUP[prob_type])

    exp.get_objective_signature(model_name, dataset, scorer)


@given(gufunc_args("(n,m,k),(k)->()", dtype=[np.float_, str], elements=[floats(), text()], unique=[False, True]))
def test_build_eval_ds(args):
    function_evals, objective_names = args
    exp.build_eval_ds(function_evals, objective_names)


@given(gufunc_args("(n),(n,m),(n)->()", dtype=np.float_, elements=floats(min_value=0, max_value=1e6)))
def test_build_timing_ds(args):
    suggest_time, eval_time, observe_time = args
    exp.build_timing_ds(suggest_time, eval_time, observe_time)


@given(
    simple_datasets(
        {"int": (ITER, SUGGEST), "real": (ITER, SUGGEST), "binary": (ITER, SUGGEST), "cat": (ITER, SUGGEST)},
        dtype={"int": int, "real": float, "binary": bool, "cat": str},
        min_side=1,
    )
)
def test_build_suggest_ds(suggest_ds):
    ds_vars = list(suggest_ds)

    n_call, n_suggest = suggest_ds[ds_vars[0]].values.shape
    suggest_log = np.zeros((n_call, n_suggest), dtype=object)
    for ii in range(n_call):
        for jj in range(n_suggest):
            suggest_log[ii, jj] = {}
            for kk in ds_vars:
                suggest_log[ii, jj][kk] = suggest_ds[kk].sel({ITER: ii, SUGGEST: jj}, drop=True).values.item()
    suggest_log = suggest_log.tolist()

    suggest_ds_2 = exp.build_suggest_ds(suggest_log)

    assert suggest_ds.equals(suggest_ds_2)


def test_get_opt_class_module():
    # Should really do parametric test but for loop good enough
    for opt_name in sorted(CONFIG.keys()):
        opt_class = exp._get_opt_class(opt_name)

        fname = inspect.getfile(opt_class)
        fname = os.path.basename(fname)

        wrapper_file, _ = CONFIG[opt_name]

        assert fname == wrapper_file
