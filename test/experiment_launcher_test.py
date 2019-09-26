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
import os
import warnings
from io import StringIO
from string import ascii_letters, digits

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis.strategies import (
    booleans,
    fixed_dictionaries,
    from_regex,
    integers,
    lists,
    sampled_from,
    text,
    tuples,
    uuids,
)
from pathvalidate.argparse import validate_filename, validate_filepath

from bayesmark import data
from bayesmark import experiment_launcher as launcher
from bayesmark.builtin_opt.config import CONFIG
from bayesmark.cmd_parse import CmdArgs
from bayesmark.constants import ARG_DELIM, METRICS, MODEL_NAMES
from hypothesis_util import seeds

DATA_NAMES = sorted(data.DATA_LOADERS.keys())


def filepaths():
    def valid(ss):
        try:
            validate_filepath(ss)
        except Exception:
            return False
        return True

    alphabet = ascii_letters + digits + "_.-~" + os.sep
    S = text(alphabet=alphabet, min_size=1).map(lambda ss: os.sep + ss).filter(valid)
    return S


def filenames(suffix=""):
    def valid(ss):
        try:
            validate_filename(ss)
        except Exception:
            return False
        return True

    alphabet = ascii_letters + digits + "_.-~"
    S = text(alphabet=alphabet, min_size=1).map(lambda ss: ss + suffix).filter(valid)
    return S


def joinables():
    S = filenames().filter(lambda ss: ARG_DELIM not in ss)
    return S


def datasets():
    return sampled_from(DATA_NAMES) | from_regex("^reg-[A-Z]*$") | from_regex("^clf-[A-Z]*$")


def launcher_args(opts, min_jobs=0):
    args_dict = {
        CmdArgs.db_root: filepaths(),
        CmdArgs.optimizer_root: filepaths(),
        CmdArgs.uuid: uuids(),
        CmdArgs.data_root: filepaths(),
        CmdArgs.db: filenames(),
        CmdArgs.optimizer: lists(sampled_from(opts), min_size=1, max_size=len(opts)),
        CmdArgs.data: lists(datasets(), min_size=1),
        CmdArgs.classifier: lists(sampled_from(MODEL_NAMES), min_size=1, max_size=len(MODEL_NAMES)),
        CmdArgs.metric: lists(sampled_from(METRICS), min_size=1, max_size=len(METRICS)),
        CmdArgs.n_calls: integers(1, 100),
        CmdArgs.n_suggest: integers(1, 100),
        CmdArgs.n_repeat: integers(1, 100),
        CmdArgs.n_jobs: integers(min_jobs, 1000),
        CmdArgs.jobs_file: filepaths(),
        CmdArgs.verbose: booleans(),
    }
    S = fixed_dictionaries(args_dict)
    return S


def launcher_args_and_config(min_jobs=0):
    def args_and_config(opts):
        args = launcher_args(opts, min_jobs=min_jobs)
        configs = fixed_dictionaries({ss: filenames(suffix=".py") for ss in opts})
        args_and_configs = tuples(args, configs)
        return args_and_configs

    # Make opt names a mix of built in opts and arbitrary names
    optimizers = lists(joinables() | sampled_from(sorted(CONFIG.keys())), min_size=1)
    S = optimizers.flatmap(args_and_config)
    return S


@given(launcher_args_and_config(), uuids())
@settings(deadline=None, suppress_health_check=(HealthCheck.too_slow,))
def test_gen_commands(args, run_uuid):
    args, opt_file_lookup = args

    assume(all(launcher._is_arg_safe(ss) for ss in args.values() if isinstance(ss, str)))

    uniqify = [CmdArgs.optimizer, CmdArgs.data, CmdArgs.classifier, CmdArgs.metric]
    for uu in uniqify:
        assume(all(launcher._is_arg_safe(ss) for ss in args[uu]))
        args[uu] = list(set(args[uu]))

    m_set = set(args[CmdArgs.metric])
    m_lookup = {problem_type: sorted(m_set.intersection(mm)) for problem_type, mm in data.METRICS_LOOKUP.items()}
    ok = all(len(m_lookup[data.get_problem_type(dd)]) > 0 for dd in args[CmdArgs.data])
    assume(ok)

    G = launcher.gen_commands(args, opt_file_lookup, run_uuid)
    L = list(G)
    assert L is not None


@given(launcher_args_and_config(min_jobs=1), uuids(), seeds())
@settings(deadline=None, suppress_health_check=(HealthCheck.too_slow,))
def test_dry_run(args, run_uuid, seed):
    args, opt_file_lookup = args

    assume(all(launcher._is_arg_safe(ss) for ss in args.values() if isinstance(ss, str)))

    uniqify = [CmdArgs.optimizer, CmdArgs.data, CmdArgs.classifier, CmdArgs.metric]
    for uu in uniqify:
        assume(all(launcher._is_arg_safe(ss) for ss in args[uu]))
        args[uu] = list(set(args[uu]))

    m_set = set(args[CmdArgs.metric])
    m_lookup = {problem_type: sorted(m_set.intersection(mm)) for problem_type, mm in data.METRICS_LOOKUP.items()}
    ok = all(len(m_lookup[data.get_problem_type(dd)]) > 0 for dd in args[CmdArgs.data])
    assume(ok)

    fp_buf = StringIO()
    random = np.random.RandomState(seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        launcher.dry_run(args, opt_file_lookup, run_uuid, fp_buf, random=random)

    jobs = fp_buf.getvalue()
    assert jobs is not None
