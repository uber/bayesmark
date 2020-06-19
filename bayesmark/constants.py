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
"""General constants that are used in multiple files in the code base.
"""

# Special constant for random search since it gets used as our reference point in the baselines
RANDOM_SEARCH = "RandomSearch"
OPTIMIZERS_FILE = "config.json"
ARG_DELIM = "_"  # Delimeter used when concat cmd argument for any reason
PY_INTERPRETER = "python"  # What command to call for sub process, we could specify version number here also.

# Variables to save in SAL
EVAL = "eval"
TIME = "time"
SUGGEST_LOG = "suggest_log"
EXP_VARS = (EVAL, TIME, SUGGEST_LOG)

# Derived variables to save in SAL
TIME_RESULTS = "time"
EVAL_RESULTS = "eval"
BASELINE = "baseline"
PERF_RESULTS = "perf"
MEAN_SCORE = "summary"

# Coordinate dim names needed in saved xr Datasets
ITER = "iter"
TEST_CASE = "function"
METHOD = "optimizer"
TRIAL = "study_id"
SUGGEST = "suggestion"
OBJECTIVE = "objective"

# Dataset variables for eval results
VISIBLE_TO_OPT = "_visible_to_opt"

# Dataset variables for time results
SUGGEST_PHASE = "suggest"
OBS_PHASE = "observe"
EVAL_PHASE = "eval"
EVAL_PHASE_SUM = "eval_sum"
EVAL_PHASE_MAX = "eval_max"

# Dataset variables for aggregate results
PERF_MED = "median"
LB_MED = "median LB"
UB_MED = "median UB"
NORMED_MED = "median normed"
PERF_MEAN = "mean"
LB_MEAN = "mean LB"
UB_MEAN = "mean UB"
NORMED_MEAN = "mean normed"
LB_NORMED_MEAN = "mean normed LB"
UB_NORMED_MEAN = "mean normed UB"
PERF_BEST = "best"
PERF_CLIP = "clip"

# Choices used for test problems, there is some redundant specification with sklearn funcs file here
MODEL_NAMES = ("DT", "MLP-adam", "MLP-sgd", "RF", "SVM", "ada", "kNN", "lasso", "linear")
DATA_LOADER_NAMES = ("breast", "digits", "iris", "wine", "boston", "diabetes")

SCORERS_CLF = ("nll", "acc")
SCORERS_REG = ("mae", "mse")
METRICS = tuple(sorted(SCORERS_CLF + SCORERS_REG))
