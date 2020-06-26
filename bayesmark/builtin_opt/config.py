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
from bayesmark.constants import RANDOM_SEARCH

CONFIG = {
    "HyperOpt": ["hyperopt_optimizer.py", {}],
    "Nevergrad-OnePlusOne": ["nevergrad_optimizer.py", {"budget": 300, "tool": "OnePlusOne"}],
    "OpenTuner-BanditA": ["opentuner_optimizer.py", {"techniques": ["AUCBanditMetaTechniqueA"]}],
    "OpenTuner-GA": ["opentuner_optimizer.py", {"techniques": ["PSO_GA_Bandit"]}],
    "OpenTuner-GA-DE": ["opentuner_optimizer.py", {"techniques": ["PSO_GA_DE"]}],
    "PySOT": ["pysot_optimizer.py", {}],
    "RandomSearch": ["random_optimizer.py", {}],
    "Scikit-GBRT-Hedge": [
        "scikit_optimizer.py",
        {"acq_func": "gp_hedge", "base_estimator": "GBRT", "n_initial_points": 5},
    ],
    "Scikit-GP-Hedge": ["scikit_optimizer.py", {"acq_func": "gp_hedge", "base_estimator": "GP", "n_initial_points": 5}],
    "Scikit-GP-LCB": ["scikit_optimizer.py", {"acq_func": "LCB", "base_estimator": "GP", "n_initial_points": 5}],
}

assert RANDOM_SEARCH in CONFIG, "%s required in settings file." % RANDOM_SEARCH
