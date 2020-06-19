from time import sleep

import bayesmark.random_search as rs
from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main


class FlakyOptimizer(AbstractOptimizer):
    def __init__(self, api_config, random=np_util.random):
        """Build wrapper class to use random search function in benchmark.

        Settings for `suggest_dict` can be passed using kwargs.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        self.random = random
        self.mode = self.random.choice(["normal", "crash", "delay"])

    def suggest(self, n_suggestions=1):
        """Get suggestion.

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
        if self.random.rand() <= 0.5 or self.mode == "normal":
            x_guess = rs.suggest_dict([], [], self.api_config, n_suggestions=n_suggestions, random=self.random)
        elif self.mode == "delay":
            sleep(15 * 60)  # 15 minutes
            x_guess = rs.suggest_dict([], [], self.api_config, n_suggestions=n_suggestions, random=self.random)
        elif self.mode == "crash":
            assert False, "Crashing for testing purposes"
        else:
            assert False, "Crashing, not for testing purposes"

        return x_guess

    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        # Random search so don't do anything
        pass


if __name__ == "__main__":
    experiment_main(FlakyOptimizer)
