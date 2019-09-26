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
import numpy as np
from hypothesis import assume, given, settings
from hypothesis.strategies import sampled_from

from bayesmark import data
from bayesmark import sklearn_funcs as skf
from bayesmark.constants import DATA_LOADER_NAMES, METRICS, MODEL_NAMES
from bayesmark.random_search import suggest_dict
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
    assert np.isscalar(loss)
