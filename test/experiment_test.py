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
from hypothesis.strategies import floats, integers, sampled_from
from hypothesis_gufunc.gufunc import gufunc_args

import bayesmark.experiment as exp
import bayesmark.random_search as rs
from bayesmark import data, np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.builtin_opt.config import CONFIG
from bayesmark.constants import DATA_LOADER_NAMES, METRICS, MODEL_NAMES
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


class FlakyProblem(TestFunction):
    def __init__(self, api_config, random):
        TestFunction.__init__(self)
        self.api_config = api_config
        self.random = random

    def evaluate(self, params):
        assert self.random.rand() <= 0.5
        return 0.0


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
def test_run_study(api_config, model_name, dataset, scorer, n_calls, n_suggestions, seed):
    api_config, _, _, _ = api_config

    prob_type = data.get_problem_type(dataset)
    assume(scorer in data.METRICS_LOOKUP[prob_type])

    function_instance = SklearnModel(model_name, dataset, scorer)
    optimizer = RandomOptimizer(api_config, random=np.random.RandomState(seed))
    optimizer.get_version()
    exp.run_study(optimizer, function_instance, n_calls, n_suggestions)


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


@given(gufunc_args("(n,m)->()", dtype=np.float_, elements=floats()))
def test_build_eval_ds(args):
    function_evals, = args
    exp.build_eval_ds(function_evals)


@given(gufunc_args("(n),(n,m),(n)->()", dtype=np.float_, elements=floats(min_value=0, max_value=1e6)))
def test_build_timing_ds(args):
    suggest_time, eval_time, observe_time = args
    exp.build_timing_ds(suggest_time, eval_time, observe_time)


def test_get_opt_class_module():
    # Should really do parametric test but for loop good enough
    for opt_name in sorted(CONFIG.keys()):
        opt_class = exp._get_opt_class(opt_name)

        fname = inspect.getfile(opt_class)
        fname = os.path.basename(fname)

        wrapper_file, _ = CONFIG[opt_name]

        assert fname == wrapper_file
