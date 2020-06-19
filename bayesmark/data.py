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
"""Module to deal with all matters relating to loading example data sets, which we tune ML models to.
"""
from enum import IntEnum, auto

import numpy as np
import pandas as pd  # only needed for csv reader, maybe try something else
from sklearn import datasets

from bayesmark.constants import DATA_LOADER_NAMES, SCORERS_CLF, SCORERS_REG
from bayesmark.path_util import join_safe_r
from bayesmark.stats import robust_standardize


class ProblemType(IntEnum):
    """The different problem types we consider. Currently, just regression (`reg`) and classification (`clf`).
    """

    clf = auto()
    reg = auto()


DATA_LOADERS = {
    "digits": (datasets.load_digits, ProblemType.clf),
    "iris": (datasets.load_iris, ProblemType.clf),
    "wine": (datasets.load_wine, ProblemType.clf),
    "breast": (datasets.load_breast_cancer, ProblemType.clf),
    "boston": (datasets.load_boston, ProblemType.reg),
    "diabetes": (datasets.load_diabetes, ProblemType.reg),
}

assert sorted(DATA_LOADERS.keys()) == sorted(DATA_LOADER_NAMES)

# Arguably, this could go in constants, but doesn't cause extra imports being here.
METRICS_LOOKUP = {ProblemType.clf: SCORERS_CLF, ProblemType.reg: SCORERS_REG}


def get_problem_type(dataset_name):
    """Determine if this dataset is a regression of classification problem.

    Parameters
    ----------
    dataset : str
        Which data set to use, must be key in `DATA_LOADERS` dict, or name of custom csv file.

    Returns
    -------
    problem_type : ProblemType
        `Enum` to indicate if regression of classification data set.
    """
    if dataset_name in DATA_LOADERS:
        _, problem_type = DATA_LOADERS[dataset_name]
        return problem_type

    # Maybe we can come up with a better system, but for now let's use a convention based on the naming of the csv file.
    if dataset_name.startswith("reg-"):
        return ProblemType.reg
    if dataset_name.startswith("clf-"):
        return ProblemType.clf
    assert False, "Can't determine problem type from dataset name."


def _csv_loader(dataset_name, return_X_y, data_root, clip_x=100):  # pragma: io
    """Load custom csv files for use in the benchmark.

    This function assumes ``dataset_name + ".csv"`` is a csv file found in the `data_root` path.  It also assumes the
    last column of the csv file is the target and the other columns are features.

    The target column should be `int` for classification and `float` for regression. Column names ending in ``"_cat"``
    are assumed to be categorical and will be one-hot encoded.

    The features (and target for regression) are robust standardized. The features are also clipped to be in
    ``[-clip_x, clip_x]`` *after* standardization.
    """
    assert return_X_y, "Only returning (X,y) tuple supported right now."
    assert clip_x >= 0

    # Quantile range for robust standardization. The 86% range is the most efficient for Gaussians. See:
    # https://github.com/scikit-learn/scikit-learn/issues/10139#issuecomment-344705040
    q_level = 0.86

    path = join_safe_r(data_root, dataset_name + ".csv")

    # For now, use convention that can get problem type based on data set name
    problem_type = get_problem_type(dataset_name)

    # Assuming no missing data in source csv files at the moment, these will
    # result in error.
    df = pd.read_csv(
        path, header=0, index_col=False, engine="c", na_filter=False, true_values=["true"], false_values=["false"]
    )

    label = df.columns[-1]  # Assume last col is target

    target = df.pop(label).values
    if problem_type == ProblemType.clf:
        assert target.dtype in (np.bool_, np.int_)
        target = target.astype(np.int_)  # convert to int for skl
    if problem_type == ProblemType.reg:
        assert target.dtype == np.float_
        # 86% range is the most efficient (at least for Gaussians)
        target = robust_standardize(target, q_level=q_level)

    # Fill in an categorical variables (object dtype of cols names ..._cat)
    cat_cols = sorted(cc for cc in df.columns if cc.endswith("_cat") or df[cc].dtype.kind == "O")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=np.float_)
    # Could also sort all columns to be sure it will be reprod

    # Everything should now be in float
    assert (df.dtypes == np.float_).all()

    data = df.values
    data = robust_standardize(data, q_level=q_level)
    # Debatable if we should include this, but there are a lot of outliers
    data = np.clip(data, -clip_x, clip_x)

    # We should probably do some logging or something to wrap up
    return data, target, problem_type


def load_data(dataset_name, data_root=None):  # pragma: io
    """Load a data set and return it in, pre-processed into numpy arrays.

    Parameters
    ----------
    dataset : str
        Which data set to use, must be key in `DATA_LOADERS` dict, or name of custom csv file.
    data_root : str
        Root directory to look for all custom csv files. May be ``None`` for sklearn data sets.

    Returns
    -------
    data : :class:`numpy:numpy.ndarray` of shape (n, d)
        The feature matrix of the data set. It will be `float` array.
    target : :class:`numpy:numpy.ndarray` of shape (n,)
        The target vector for the problem, which is `int` for classification and `float` for regression.
    problem_type : :class:`bayesmark.data.ProblemType`
        `Enum` to indicate if regression of classification data set.
    """
    if dataset_name in DATA_LOADERS:
        loader_f, problem_type = DATA_LOADERS[dataset_name]
        data, target = loader_f(return_X_y=True)
    else:  # try to load as custom csv
        assert data_root is not None, "data root cannot be None when custom csv requested."
        data, target, problem_type = _csv_loader(dataset_name, return_X_y=True, data_root=data_root)
    return data, target, problem_type
