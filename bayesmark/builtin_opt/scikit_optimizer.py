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
from scipy.interpolate import interp1d
from skopt import Optimizer as SkOpt
from skopt.space import Categorical, Integer, Real

from bayesmark.abstract_optimizer import AbstractOptimizer


class ScikitOptimizer(AbstractOptimizer):
    primary_import = "scikit-optimize"

    def __init__(self, api_config, base_estimator="GP", acq_func="gp_hedge", n_initial_points=5, **kwargs):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        base_estimator : {'GP', 'RF', 'ET', 'GBRT'}
            How to estimate the objective function.
        acq_func : {'LCB', 'EI', 'PI', 'gp_hedge', 'EIps', 'PIps'}
            Acquisition objective to decide next suggestion.
        n_initial_points : int
            Number of points to sample randomly before actual Bayes opt.
        """
        AbstractOptimizer.__init__(self, api_config)

        dimensions, self.round_to_values = ScikitOptimizer.get_sk_dimensions(api_config)

        # Older versions of skopt don't copy over the dimensions names during
        # normalization and hence the names are missing in
        # self.skopt.space.dimensions. Therefore, we save our own copy of
        # dimensions list to be safe. If we can commit to using the newer
        # versions of skopt we can delete self.dimensions.
        self.dimensions_list = tuple(dd.name for dd in dimensions)

        # Undecided where we want to pass the kwargs, so for now just make sure
        # they are blank
        assert len(kwargs) == 0

        self.skopt = SkOpt(
            dimensions,
            n_initial_points=n_initial_points,
            base_estimator=base_estimator,
            acq_func=acq_func,
            acq_optimizer="auto",
            acq_func_kwargs={},
            acq_optimizer_kwargs={},
        )

    @staticmethod
    def get_sk_dimensions(api_config, transform="normalize"):
        """Help routine to setup skopt search space in constructor.

        Take api_config as argument so this can be static.
        """
        # The ordering of iteration prob makes no difference, but just to be
        # safe and consistnent with space.py, I will make sorted.
        param_list = sorted(api_config.keys())

        sk_dims = []
        round_to_values = {}
        for param_name in param_list:
            param_config = api_config[param_name]

            param_type = param_config["type"]

            param_space = param_config.get("space", None)
            param_range = param_config.get("range", None)
            param_values = param_config.get("values", None)

            # Some setup for case that whitelist of values is provided:
            values_only_type = param_type in ("cat", "ordinal")
            if (param_values is not None) and (not values_only_type):
                assert param_range is None
                param_values = np.unique(param_values)
                param_range = (param_values[0], param_values[-1])
                round_to_values[param_name] = interp1d(
                    param_values, param_values, kind="nearest", fill_value="extrapolate"
                )

            if param_type == "int":
                # Integer space in sklearn does not support any warping => Need
                # to leave the warping as linear in skopt.
                sk_dims.append(Integer(param_range[0], param_range[-1], transform=transform, name=param_name))
            elif param_type == "bool":
                assert param_range is None
                assert param_values is None
                sk_dims.append(Integer(0, 1, transform=transform, name=param_name))
            elif param_type in ("cat", "ordinal"):
                assert param_range is None
                # Leave x-form to one-hot as per skopt default
                sk_dims.append(Categorical(param_values, name=param_name))
            elif param_type == "real":
                # Skopt doesn't support all our warpings, so need to pick
                # closest substitute it does support.
                prior = "log-uniform" if param_space in ("log", "logit") else "uniform"
                sk_dims.append(Real(param_range[0], param_range[-1], prior=prior, transform=transform, name=param_name))
            else:
                assert False, "type %s not handled in API" % param_type
        return sk_dims, round_to_values

    def suggest(self, n_suggestions=1):
        """Get a suggestion from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        # First get list of lists from skopt.ask()
        next_guess = self.skopt.ask(n_points=n_suggestions)
        # Then convert to list of dicts
        next_guess = [dict(zip(self.dimensions_list, x)) for x in next_guess]

        # Now do the rounding, custom rounding is not supported in skopt. Note
        # that there is not nec a round function for each dimension here.
        for param_name, round_f in self.round_to_values.items():
            for xx in next_guess:
                xx[param_name] = round_f(xx[param_name])
        return next_guess

    def observe(self, X, y):
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        # Supposedly skopt can handle blocks, but not sure about interface for
        # that. Just do loop to be safe for now.
        for xx, yy in zip(X, y):
            # skopt needs lists instead of dicts
            xx = [xx[dim_name] for dim_name in self.dimensions_list]
            # Just ignore, any inf observations we got, unclear if right thing
            if np.isfinite(yy):
                self.skopt.tell(xx, yy)


opt_wrapper = ScikitOptimizer
