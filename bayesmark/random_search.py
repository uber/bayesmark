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
"""A baseline random search in our standardized optimizer interface. Useful for baselines.
"""
import numpy as np

from bayesmark import np_util
from bayesmark.space import JointSpace


def suggest_dict(X, y, meta, n_suggestions=1, random=np_util.random):
    """Stateless function to create suggestions for next query point in random search optimization.

    This implements the API for general structures of different data types.

    Parameters
    ----------
    X : list(dict)
        Places where the objective function has already been evaluated. Not actually used in random search.
    y : :class:`numpy:numpy.ndarray`, shape (n,)
        Corresponding values where objective has been evaluated. Not actually used in random search.
    meta : dict(str, dict)
        Configuration of the optimization variables. See API description.
    n_suggestions : int
        Desired number of parallel suggestions in the output
    random : :class:`numpy:numpy.random.RandomState`
        Optionally pass in random stream for reproducibility.

    Returns
    -------
    next_guess : list(dict)
        List of `n_suggestions` suggestions to evaluate the objective function.
        Each suggestion is a dictionary where each key corresponds to a parameter being optimized.
    """
    # Warp and get bounds
    space_x = JointSpace(meta)
    X_warped = space_x.warp(X)
    bounds = space_x.get_bounds()
    _, n_params = _check_x_y(X_warped, y, allow_impute=True)
    lb, ub = _check_bounds(bounds, n_params)

    # Get the suggestion
    suggest_x = random.uniform(lb, ub, size=(n_suggestions, n_params))

    # Unwarp
    next_guess = space_x.unwarp(suggest_x)
    return next_guess


def _check_x_y(X, y, allow_impute=False):  # pragma: validator
    """Input validation for `suggest` routine."""
    if not (np.ndim(X) == 2):
        raise ValueError("X must be 2-dimensional got %s." % str(np.shape(X)))
    n_obs, n_params = np.shape(X)

    assert n_params >= 1, "We do not support suggest on empty space."

    if not (np.shape(y) == (n_obs,)):
        raise ValueError("y must be %s not %s." % (str((n_obs,)), str(np.shape(y))))

    if not np.all(np.isfinite(X)):
        raise ValueError("X must be finite.")

    n_real_obs = n_obs
    if allow_impute:
        if not np.all(np.isfinite(y) | np.isnan(y)):
            raise ValueError("y can't contain infs even with data imputation.")
        n_real_obs = np.sum(np.isfinite(y))
    else:
        if not np.all(np.isfinite(y)):
            raise ValueError("y must be finite when data imputation not used.")

    return n_real_obs, n_params


def _check_bounds(bounds, n_params):  # pragma: validator
    """Input validation for `suggest` routine."""
    if not (np.shape(bounds) == (n_params, 2)):
        raise ValueError("bounds must have shape %s not %s." % (str((n_params, 2)), str(np.shape(bounds))))

    lb, ub = np.asarray(bounds).T
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
        raise ValueError("bounds must be finite.")
    if not (np.all(lb <= ub)):
        raise ValueError("lower bound must be less than upper bound.")
    return lb, ub
