[//]: # (pandoc -t beamer slides.md -o slides.pdf --highlight-style=espresso -H make-code-footnotesize.tex)

# Bayesmark

* General tool for benchmarking Bayesian optimization (BO) packages on realistic problems

`pip install bayesmark`

# Strategy

* Aims to become a standard in BO research
* Wrapper class with standardized API for all tools
* Move the standard API for BO to the "open loop" type API that we prefer
    * It would be good if all tools followed our open loop API

# Framework

* Optimize any black box objective function
* Open loop:
    * Take past observations of the objective using an `observe` function
    * Guess the next place to optimize the objective from a `suggest` function

# Packages

There are many open source packages:

* Botorch
* HyperOpt
* Nevergrad
* OpenTuner
* PySOT
* Scikit-optimize
* ...

. . .

* Need good to tool to easily compare them

# Stages

* Organized system for running many experiments
    * Produce jobs files in reproducible way
* Analysis system to construct aggregate scores in a fair way
* Generate plots with confidence intervals

![Experiment Flow](https://github.com/uber/bayesmark/files/3714708/flowchart.pdf){ width=90% }

# Example

```console
> # setup
> DB_ROOT=./notebooks  # path/to/where/you/put/results
> DBID=bo_example_folder
> mkdir $DB_ROOT
> # experiments
> bayesmark-launch -n 15 -r 3 -dir $DB_ROOT -b $DBID -o RandomSearch PySOT -c SVM DT -v
Supply --uuid 3adc3182635e44ea96969d267591f034 to reproduce this run.
Supply --dbid bo_example_folder to append to this experiment or reproduce jobs file.
User must ensure equal reps of each optimizer for unbiased results
-c DT -d boston -o PySOT -u a1b287b450385ad09b2abd7582f404a2 -m mae -n 15 -p 1 -dir /notebooks -b bo_example_folder
-c DT -d boston -o PySOT -u 63746599ae3f5111a96942d930ba1898 -m mse -n 15 -p 1 -dir /notebooks -b bo_example_folder
-c DT -d boston -o RandomSearch -u 8ba16c880ef45b27ba0909199ab7aa8a -m mae -n 15 -p 1 -dir /notebooks -b bo_example_folder
...
0 failures of benchmark script after 144 studies.
done
```

# Example II

```console
> # aggregate
> bayesmark-agg -dir $DB_ROOT -b $DBID
> # analyze
> bayesmark-anal -dir $DB_ROOT -b $DBID -v
...
median score @ 15:
optimizer
PySOT_0.2.3_9b766b6           0.330404
RandomSearch_0.0.1_9b766b6    0.961829
mean score @ 15:
optimizer
PySOT_0.2.3_9b766b6           0.124262
RandomSearch_0.0.1_9b766b6    0.256422
normed mean score @ 15:
optimizer
PySOT_0.2.3_9b766b6           0.475775
RandomSearch_0.0.1_9b766b6    0.981787
done
```

# Notes

* Reproducibility
    * All randomness seeded using a system of UUIDs
    * Can reproduce runs with the UUID
    * Reproducibility plays a key role in this benchmark
* Experiment aggregate scripts takes independent data results and makes single file
* Experiment analysis computes aggregated scores across all test problems
* Provide notebooks for plotting
    * Would like to try out `streamlit` package instead

# Adding a new optimizer

* Creating a wrapper class around the new optimizer
    * Follow the interface specified of the `AbstractOptimizer`
* Configuration `config.json` allows for multiple setups for single wrapper

Simple interface:

* `suggest` and `observe` functions
* We can observe/try inputs that were never suggested
* We can ignore suggestions
* The objective function may not be something as simple as a Python function
* `scipy.optimization` is closed loop

# Adding a new optimizer I

```python
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main

class NewOptimizerName(AbstractOptimizer):
    # Used for determining the version number of package used
    primary_import = "name of import used e.g, opentuner"

    def __init__(self, api_config, optional_arg_foo=None):
        """Build wrapper class to use optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables.
        """
        AbstractOptimizer.__init__(self, api_config)
        # Do whatever other setup is needed
        # ...
```

# Adding a new optimizer II

```python
    def suggest(self, n_suggestions=1):
        """Get suggestion from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the
            objective function. Each suggestion is a dictionary
            where each key corresponds to a parameter being
            optimized.
        """
        # Do whatever is needed to get the parallel guesses
        # ...
        return x_guess
```

# Adding a new optimizer III

```python
    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been
            evaluated. Each suggestion is a dictionary where each
            key corresponds to a parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values
        """
        # Update model with new objective function observations
        # ...
        # No return statement needed
```

# Adding a new optimizer IV

In `config.json`:

```json
{
    "OpenTuner-BanditA-New": [
        "opentuner_optimizer.py",
        {"techniques": ["AUCBanditMetaTechniqueA"]}
    ],
    "OpenTuner-GA-DE-New": [
        "opentuner_optimizer.py",
        {"techniques": ["PSO_GA_DE"]}
    ],
    "OpenTuner-GA-New": [
        "opentuner_optimizer.py",
        {"techniques": ["PSO_GA_Bandit"]}
    ]
}
```

# Get started

* Use `pip install bayesmark[optimizers,notebooks]`
* Docs: `https://bayesmark.readthedocs.io`
* GitHub: `https://github.com/uber/bayesmark`
* PyPI: `https://pypi.org/project/bayesmark`

# Dry run for cluster jobs

* "dry run" of the launcher by specifying a value for `--num-jobs` greater than zero
* Produce file with one command (job) per line (for `quicksilver`)

For example,
```
# running: {'--uuid': None, '-db-root': '/foo', '--opt-root': '/example_opt_root', '--data-root': None, '--db': 'bo_example_folder', '--opt': ['RandomSearch', 'PySOT'], '--data': None, '--classifier': ['SVM', 'DT'], '--metric': None, '--calls': 15, '--suggestions': 1, '--repeat': 3, '--num-jobs': 50, '--jobs-file': '/jobs.txt', '--verbose': False, 'dry_run': True, 'rev': '9a14ef2', 'opt_rev': None}
# cmd: python bayesmark-launch -n 15 -r 3 -dir foo -o RandomSearch PySOT -c SVM DT -nj 50 -b bo_example_folder
job_e2b63a9_00 bayesmark-exp -c SVM -d diabetes -o PySOT -u 079a155f03095d2ba414a5d2cedde08c -m mse -n 15 -p 1 -dir foo -b bo_example_folder && bayesmark-exp -c SVM -d boston -o RandomSearch -u 400e4c0be8295ad59db22d9b5f31d153 -m mse -n 15 -p 1 -dir foo -b bo_example_folder && bayesmark-exp -c SVM -d digits -o RandomSearch -u fe73a2aa960a5e3f8d78bfc4bcf51428 -m acc -n 15 -p 1 -dir foo -b bo_example_folder
job_e2b63a9_01 bayesmark-exp -c DT -d diabetes -o PySOT -u db1d9297948554e096006c172a0486fb -m mse -n 15 -p 1 -dir foo -b bo_example_folder && bayesmark-exp -c SVM -d boston -o RandomSearch -u 7148f690ed6a543890639cc59db8320b -m mse -n 15 -p 1 -dir foo -b bo_example_folder && bayesmark-exp -c SVM -d breast -o PySOT -u 72c104ba1b6d5bb8a546b0064a7c52b1 -m nll -n 15 -p 1 -dir foo -b bo_example_folder
job_e2b63a9_02 bayesmark-exp -c SVM -d iris -o PySOT -u cc63b2c1e4315a9aac0f5f7b496bfb0f -m nll -n 15 -p 1 -dir foo -b bo_example_folder && bayesmark-exp -c DT -d breast -o RandomSearch -u aec62e1c8b5552e6b12836f0c59c1681 -m nll -n 15 -p 1 -dir foo -b bo_example_folder && bayesmark-exp -c DT -d digits -o RandomSearch -u 4d0a175d56105b6bb3055c3b62937b2d -m acc -n 15 -p 1 -dir foo -b bo_example_folder
...
```

# Launch the experiments

```
usage: bayesmark-launch [-h] [-dir DB_ROOT] [-odir OPTIMIZER_ROOT] [-v] [-u UUID]
                  [-dr DATA_ROOT] [-b DB] [-o OPTIMIZER [OPTIMIZER ...]]
                  [-d DATA [DATA ...]]
                  [-c [{DT,MLP-adam,MLP-sgd,RF,SVM,ada,kNN,lasso,linear} ...]]
                  [-m [{acc,mae,mse,nll} ...]] [-n N_CALLS]
                  [-p N_SUGGEST] [-r N_REPEAT] [-nj N_JOBS] [-ofile JOBS_FILE]
```

# Options

```
-dir DB_ROOT, -db-root DB_ROOT
                      root directory for all benchmark experiments output
-odir OPTIMIZER_ROOT, --opt-root OPTIMIZER_ROOT
                      Directory with optimization wrappers
-v, --verbose         print the study logs to console
-u UUID, --uuid UUID  length 32 hex UUID for this experiment
-dr DATA_ROOT, --data-root DATA_ROOT
                      root directory for all custom csv files
-b DB, --db DB        database ID of this benchmark experiment
-o OPTIMIZER [OPTIMIZER ...], --opt OPTIMIZER [OPTIMIZER ...]
                      optimizers to use
-d DATA [DATA ...], --data DATA [DATA ...]
                      data sets to use
-c, --classifier [{DT,MLP-adam,MLP-sgd,RF,SVM,ada,kNN,lasso,linear} ...]
                      classifiers to use
-m, --metric [{acc,mae,mse,nll} ...]
                      scoring metric to use
-n N_CALLS, --calls N_CALLS
                      number of function evaluations
-p N_SUGGEST, --suggestions N_SUGGEST
                      number of suggestions to provide in parallel
-r N_REPEAT, --repeat N_REPEAT
                      number of repetitions of each study
-nj N_JOBS, --num-jobs N_JOBS
                      number of jobs to put in the dry run file, the default
                      0 value disables dry run (real run)
-ofile JOBS_FILE, --jobs-file JOBS_FILE
                      a jobs file with all commands to be run
```

# Built-ins

"built-in" optimizers:
```
"HyperOpt", "Nevergrad-OnePlusOne", "OpenTuner-BanditA",
"OpenTuner-GA", "OpenTuner-GA-DE", "PySOT", "RandomSearch",
"Scikit-GBRT-Hedge", "Scikit-GP-Hedge", "Scikit-GP-LCB"
```

"built-in" data sets:
```
"breast", "digits", "iris", "wine", "boston", "diabetes"
```

By default uses cartesian produce of data x model x metric x optimizer x repeated trial.

# Scoring

* $S_{pmtn} = \textrm{cumm-min}_t F_{pmtn}$
    * Always score based on the cumulative minimum at batch $t$
* $S'_{pmtn} = \min(S_{pmtn}, \textrm{clip}_p)$
    * For mean, need bounded loss since could be rare infinite losses

. . .

* $\textrm{mean-perf}_{pmt} = \textrm{mean}_n \, S'_{pmtn}$
* $\textrm{norm-mean-perf}_{pmt} = \frac{\textrm{mean-perf}_{pmt}  - \textrm{opt}_p}{\textrm{clip}_p  - \textrm{opt}_p}$
    * Put all problems on same scale (units), invariant to linear transformation

. . .

* $\textrm{mean-perf}_{mt} = \textrm{mean}_p \, \textrm{norm-mean-perf}_{pmt}$
* $\textrm{norm-mean-perf}_{mt} = \frac{\textrm{mean-perf}_{mt}}{\textrm{rand-mean-perf}_{t}}$
    * The best final summary metric
* $\textrm{rand-mean-perf}_{t} = \textrm{mean}_p \, \frac{\textrm{rand-mean-perf}_{pt} - \textrm{opt}_p}{\textrm{clip}_p  - \textrm{opt}_p}$

# Output

![Scores](https://user-images.githubusercontent.com/28273671/66338456-02516b80-e8f6-11e9-8156-2e84e04cf6fe.png){ width=95% }

# Randoms search baseline

* Conceptually, $\textrm{rand-mean-perf}_{pt} = \textrm{mean}_n \, S'_{pmtn}$
* But, all function evaluations for random search are iid across $t$
* So, use a more efficient pooled estimator

# Expected max estimation

* Get unbiased estimate on $\mu_m := \mathbb{E}[\max x_{1:m}]$ given $x_{1:n} \sim p$
* Derive L-estimator
* Basically, random subset of size $m$ is unbiased estimator with high variance
* Average over all possible subsets, can derive weights for order statistics $o_{1:n}$

$$\hat{\mu}_m = \sum_{i=1}^n w_i o_i$$
$$w_i \propto \frac{(i - 1)!}{(i - m)!} \propto \binom{i-1}{m-1}$$

* Implemented in more numerically safe way

$$w_i \propto \frac{(i - 1)!}{(i - m)!} \propto \binom{i-1}{m-1}$$

# Quantile estimation

* Likewise, for normalizing median scores we can get estimate on median $\max x_{1:m}$
* Just estimate quantile $q = 0.5^m$ from data (this property holds for any distribution)
    * Again from order statistics
* Can also get distribution free confidence intervals (CI) on quantiles using binomial argument
* Use CI: $[o_i, o_j]$
    * Get indices such that $\textrm{binom-cdf}(j - 1, n, q) - \textrm{binom-cdf}(i - 1, n, q) \geq 1 - \alpha$
