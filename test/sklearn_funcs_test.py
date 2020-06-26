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
import pickle as pkl

import numpy as np
from hypothesis import assume, given, settings
from hypothesis.strategies import sampled_from, text
from sklearn.linear_model import LinearRegression

from bayesmark import data
from bayesmark import sklearn_funcs as skf
from bayesmark.constants import ARG_DELIM, DATA_LOADER_NAMES, METRICS, MODEL_NAMES
from bayesmark.random_search import suggest_dict
from bayesmark.space import JointSpace
from hypothesis_util import seeds


@given(sampled_from(MODEL_NAMES), sampled_from(DATA_LOADER_NAMES), sampled_from(METRICS), seeds(), seeds())
@settings(deadline=None)
def test_sklearn_model(model, dataset, metric, shuffle_seed, rs_seed):
    prob_type = data.get_problem_type(dataset)
    assume(metric in data.METRICS_LOOKUP[prob_type])

    test_prob = skf.SklearnModel(model, dataset, metric, shuffle_seed=shuffle_seed)

    api_config = test_prob.get_api_config()
    x_guess, = suggest_dict([], [], api_config, n_suggestions=1, random=np.random.RandomState(rs_seed))

    loss = test_prob.evaluate(x_guess)

    assert isinstance(loss, tuple)
    assert all(isinstance(xx, float) for xx in loss)
    assert np.shape(loss) == np.shape(test_prob.objective_names)


@given(text(), text(), text())
def test_inverse_test_case_str(model, dataset, scorer):
    assume(ARG_DELIM not in (model + dataset + scorer))

    test_case = skf.SklearnModel.test_case_str(model, dataset, scorer)
    R = skf.SklearnModel.inverse_test_case_str(test_case)

    assert R == (model, dataset, scorer)


@given(sampled_from(MODEL_NAMES), sampled_from(DATA_LOADER_NAMES), sampled_from(METRICS), seeds(), seeds())
@settings(deadline=None)
def test_sklearn_model_surr(model, dataset, metric, model_seed, rs_seed):
    prob_type = data.get_problem_type(dataset)
    assume(metric in data.METRICS_LOOKUP[prob_type])

    test_prob = skf.SklearnModel(model, dataset, metric, shuffle_seed=0)
    api_config = test_prob.get_api_config()
    space = JointSpace(api_config)

    n_obj = len(test_prob.objective_names)

    n_suggestions = 20

    x_guess = suggest_dict([], [], api_config, n_suggestions=n_suggestions, random=np.random.RandomState(rs_seed))
    x_guess_w = space.warp(x_guess)

    random = np.random.RandomState(model_seed)
    y = random.randn(n_suggestions, n_obj)

    reg = LinearRegression()
    reg.fit(x_guess_w, y)
    loss0 = reg.predict(x_guess_w)

    path = pkl.dumps(reg)
    del reg
    assert isinstance(path, bytes)

    test_prob_surr = skf.SklearnSurrogate(model, dataset, metric, path)
    loss = test_prob_surr.evaluate(x_guess[0])

    assert isinstance(loss, tuple)
    assert all(isinstance(xx, float) for xx in loss)
    assert np.shape(loss) == np.shape(test_prob.objective_names)

    assert np.allclose(loss0[0], np.array(loss))
