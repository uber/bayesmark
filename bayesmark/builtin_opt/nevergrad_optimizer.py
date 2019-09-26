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
import nevergrad.optimization as optimization
import numpy as np
from nevergrad import instrumentation as inst
from scipy.stats import norm

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.np_util import linear_rescale
from bayesmark.space import Real


class NevergradOptimizer(AbstractOptimizer):
    primary_import = "nevergrad"

    def __init__(self, api_config, tool, budget):
        """Build wrapper class to use nevergrad optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        budget : int
            Expected number of max function evals
        """
        AbstractOptimizer.__init__(self, api_config)

        self.instrum, self.space = NevergradOptimizer.get_nvg_dimensions(api_config)

        dimension = self.instrum.dimension
        opt_class = optimization.registry[tool]
        self.optim = opt_class(dimension=dimension, budget=budget)

    @staticmethod
    def get_nvg_dimensions(api_config):
        """Help routine to setup nevergrad search space in constructor.

        Take api_config as argument so this can be static.
        """
        # The ordering of iteration prob makes no difference, but just to be
        # safe and consistnent with space.py, I will make sorted.
        param_list = sorted(api_config.keys())

        all_args = {}
        all_prewarp = {}
        for param_name in param_list:
            param_config = api_config[param_name]

            param_type = param_config["type"]

            param_space = param_config.get("space", None)
            param_range = param_config.get("range", None)
            param_values = param_config.get("values", None)

            prewarp = None
            if param_type == "cat":
                assert param_space is None
                assert param_range is None
                arg = inst.var.SoftmaxCategorical(param_values)
            elif param_type == "bool":
                assert param_space is None
                assert param_range is None
                assert param_values is None
                arg = inst.var.OrderedDiscrete([False, True])
            elif param_values is not None:
                assert param_type in ("int", "ordinal", "real")
                arg = inst.var.OrderedDiscrete(param_values)
                # We are throwing away information here, but OrderedDiscrete
                # appears to be invariant to monotonic transformation anyway.
            elif param_type == "int":
                assert param_values is None
                # Need +1 since API in inclusive
                choices = range(int(param_range[0]), int(param_range[-1]) + 1)
                arg = inst.var.OrderedDiscrete(choices)
                # We are throwing away information here, but OrderedDiscrete
                # appears to be invariant to monotonic transformation anyway.
            elif param_type == "real":
                assert param_values is None
                assert param_range is not None
                # Will need to warp to this space sep.
                arg = inst.var.Gaussian(mean=0, std=1)
                prewarp = Real(warp=param_space, range_=param_range)
            else:
                assert False, "type %s not handled in API" % param_type

            all_args[param_name] = arg
            all_prewarp[param_name] = prewarp
        instrum = inst.Instrumentation(**all_args)
        return instrum, all_prewarp

    def prewarp(self, xx):
        """Extra work needed to get variables into the Gaussian space
        representation."""
        xxw = {}
        for arg_name, vv in xx.items():
            assert np.isscalar(vv)
            space = self.space[arg_name]

            if space is not None:
                # Warp so we think it is apriori uniform in [a, b]
                vv = space.warp(vv)
                assert vv.size == 1

                # Now make uniform on [0, 1], also unpack warped to scalar
                (lb, ub), = space.get_bounds()
                vv = linear_rescale(vv.item(), lb, ub, 0, 1)

                # Now make std Gaussian apriori
                vv = norm.ppf(vv)
            assert np.isscalar(vv)
            xxw[arg_name] = vv
        return xxw

    def postwarp(self, xxw):
        """Extra work needed to undo the Gaussian space representation."""
        xx = {}
        for arg_name, vv in xxw.items():
            assert np.isscalar(vv)
            space = self.space[arg_name]

            if space is not None:
                # Now make std Gaussian apriori
                vv = norm.cdf(vv)

                # Now make uniform on [0, 1]
                (lb, ub), = space.get_bounds()
                vv = linear_rescale(vv, 0, 1, lb, ub)

                # Warp so we think it is apriori uniform in [a, b]
                vv = space.unwarp([vv])
            assert np.isscalar(vv)
            xx[arg_name] = vv
        return xx

    def suggest(self, n_suggestions=1):
        """Get suggestion from nevergrad.

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
        x_guess_data = [self.optim.ask() for _ in range(n_suggestions)]

        x_guess = [None] * n_suggestions
        for ii, xx in enumerate(x_guess_data):
            x_pos, x_kwarg = self.instrum.data_to_arguments(xx)
            assert x_pos == ()
            x_guess[ii] = self.postwarp(x_kwarg)

        return x_guess

    def observe(self, X, y):
        """Feed an observation back to nevergrad.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        for xx, yy in zip(X, y):
            xx = self.prewarp(xx)
            xx = self.instrum.arguments_to_data(**xx)
            self.optim.tell(xx, yy)


opt_wrapper = NevergradOptimizer
