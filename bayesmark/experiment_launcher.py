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
"""Launch studies in separate studies or do dry run to build jobs file with lists of commands to run.
"""
import json
import logging
import random as pyrandom
import uuid as pyuuid
import warnings
from itertools import product
from subprocess import TimeoutExpired, call

import numpy as np

import bayesmark.cmd_parse as cmd
from bayesmark.builtin_opt.config import CONFIG
from bayesmark.cmd_parse import CMD_STR, CmdArgs, serializable_dict
from bayesmark.constants import ARG_DELIM, DATA_LOADER_NAMES, EXP_VARS, METRICS, MODEL_NAMES, PY_INTERPRETER
from bayesmark.data import METRICS_LOOKUP, get_problem_type
from bayesmark.np_util import random as np_random
from bayesmark.np_util import random_seed, strat_split
from bayesmark.path_util import absopen
from bayesmark.serialize import XRSerializer
from bayesmark.util import range_str, shell_join, str_join_safe, strict_sorted

# How much of uuid to put in job name to avoid name clashes
UUID_JOB_CHARS = 7
# Warning: this name is also specified in setup.py, and violates the DRY principle. So if it gets changed in setup.py,
# it must also be changed here!
EXPERIMENT_ENTRY = "bayesmark-exp"

logger = logging.getLogger(__name__)


def _is_arg_safe(ss):
    """Check if `str` is safe as argument to `argparse`."""
    if len(ss) == 0:
        return False
    safe = ss[0] != "-"
    return safe


def arg_safe_str(val):
    """Cast value as `str`, raise error if not safe as argument to `argparse`."""
    ss = str(val)
    if not _is_arg_safe(ss):
        raise ValueError("%s is not safe for argparse" % ss)
    return ss


def gen_commands(args, opt_file_lookup, run_uuid):
    """Generator providing commands to launch processes for experiments.

    Parameters
    ----------
    args : dict(CmdArgs, [int, str])
        Arguments of options to pass to the experiments being launched. The keys corresponds to the same arguments
        passed to this program.
    opt_file_lookup : dict(str, str)
        Mapping from method name to filename containing wrapper class for the method.
    run_uuid : uuid.UUID
        UUID for this launcher run. Needed to generate different experiments UUIDs on each call. This function is
        deterministic provided the same `run_uuid`.

    Yields
    ------
    iteration_key : (str, str, str, str)
        Tuple containing ``(trial, classifier, data, optimizer)`` to index the experiment.
    full_cmd : tuple(str)
        Strings containing command and arguments to run a process with experiment. Join with whitespace or use
        :func:`.util.shell_join` to get string with executable command. The command omits ``--opt-root`` which means it
        will default to ``.`` if the command is executed. As such, the command assumes it is executed with
        ``--opt-root`` as the working directory.
    """
    args_to_pass_thru = [CmdArgs.n_calls, CmdArgs.n_suggest, CmdArgs.db_root, CmdArgs.db]
    # This could be made simpler and avoid if statement if we just always pass dataroot, even if no custom data used.
    if args[CmdArgs.data_root] is not None:
        args_to_pass_thru.append(CmdArgs.data_root)

    # Possibilities to iterate over. Put them in sorted order just for good measure.
    c_list = strict_sorted(MODEL_NAMES if args[CmdArgs.classifier] is None else args[CmdArgs.classifier])
    d_list = strict_sorted(DATA_LOADER_NAMES if args[CmdArgs.data] is None else args[CmdArgs.data])
    o_list = strict_sorted(
        list(opt_file_lookup.keys()) + list(CONFIG.keys())
        if args[CmdArgs.optimizer] is None
        else args[CmdArgs.optimizer]
    )
    assert all(
        ((optimizer in opt_file_lookup) or (optimizer in CONFIG)) for optimizer in o_list
    ), "unknown optimizer in optimizer list"

    m_set = set(METRICS if args[CmdArgs.metric] is None else args[CmdArgs.metric])
    m_lookup = {problem_type: sorted(m_set.intersection(mm)) for problem_type, mm in METRICS_LOOKUP.items()}
    assert all(
        (len(m_lookup[get_problem_type(data)]) > 0) for data in d_list
    ), "At one metric needed for each problem type of data sets"

    G = product(range_str(args[CmdArgs.n_repeat]), c_list, d_list, o_list)  # iterate all combos
    for rep, classifier, data, optimizer in G:
        _, rep_str = rep
        problem_type = get_problem_type(data)
        for metric in m_lookup[problem_type]:
            # Get a reproducible string based (conditioned on having same (run uuid), but should also never give
            # a duplicate (unless we force the same run uuid twice).
            iteration_key = (rep_str, classifier, data, optimizer, metric)
            iteration_id = str_join_safe(ARG_DELIM, iteration_key)
            sub_uuid = pyuuid.uuid5(run_uuid, iteration_id).hex

            # Build the argument list for subproc, passing some args thru
            cmd_args_pass_thru = [[CMD_STR[vv][0], arg_safe_str(args[vv])] for vv in args_to_pass_thru]
            # Technically, the optimizer is is not actually needed here for non-built in optimizers because it already
            # specified via the entry point: optimizer_wrapper_file
            cmd_args = [
                [CMD_STR[CmdArgs.classifier][0], arg_safe_str(classifier)],
                [CMD_STR[CmdArgs.data][0], arg_safe_str(data)],
                [CMD_STR[CmdArgs.optimizer][0], arg_safe_str(optimizer)],
                [CMD_STR[CmdArgs.uuid][0], arg_safe_str(sub_uuid)],
                [CMD_STR[CmdArgs.metric][0], arg_safe_str(metric)],
            ]
            cmd_args = tuple(sum(cmd_args + cmd_args_pass_thru, []))
            logger.info(" ".join(cmd_args))

            # The experiment command without the arguments
            if optimizer in CONFIG:  # => built in optimizer wrapper
                experiment_cmd = (EXPERIMENT_ENTRY,)
            else:
                optimizer_wrapper_file = opt_file_lookup[optimizer]
                assert optimizer_wrapper_file.endswith(".py"), "optimizer wrapper should a be .py file"
                experiment_cmd = (PY_INTERPRETER, optimizer_wrapper_file)

            # Check arg safe again, off elements in list need to be argsafe
            assert all((_is_arg_safe(ss) == (ii % 2 == 1)) for ii, ss in enumerate(cmd_args))

            full_cmd = experiment_cmd + cmd_args
            yield iteration_key, full_cmd


def dry_run(args, opt_file_lookup, run_uuid, fp, random=np_random):
    """Write to buffer description of commands for running all experiments.

    This function is almost pure by writing to a buffer, but it could be switched to a generator.

    Parameters
    ----------
    args : dict(CmdArgs, [int, str])
        Arguments of options to pass to the experiments being launched. The keys corresponds to the same arguments
        passed to this program.
    opt_file_lookup : dict(str, str)
        Mapping from method name to filename containing wrapper class for the method.
    run_uuid : uuid.UUID
        UUID for this launcher run. Needed to generate different experiments UUIDs on each call. This function is
        deterministic provided the same `run_uuid`.
    fp : writable buffer
        File handle to write out sequence of commands to execute (broken into jobs on each line) to execute all the
        experiments (possibly each job in parallel).
    random : RandomState
        Random stream to use for reproducibility.
    """
    assert args[CmdArgs.n_jobs] > 0, "Must have non-zero jobs for dry run"

    # Taking in file pointer since then we can test without actual file. Could also build generator that returns lines
    # to write.
    manual_setup_info = XRSerializer.init_db_manual(args[CmdArgs.db_root], db=args[CmdArgs.db], keys=EXP_VARS)
    warnings.warn(manual_setup_info, UserWarning)

    # Get the commands
    dry_run_commands = {}
    G = gen_commands(args, opt_file_lookup, run_uuid)
    for (_, _, _, optimizer, _), full_cmd in G:
        cmd_str = shell_join(full_cmd)
        dry_run_commands.setdefault(optimizer, []).append(cmd_str)

    # Make sure we never have any empty jobs, which is a waste
    n_commands = sum(len(v) for v in dry_run_commands.values())
    n_jobs = min(args[CmdArgs.n_jobs], n_commands)

    # Would prob also work with pyrandom, but only tested np random so far
    subcommands = strat_split(list(dry_run_commands.values()), n_jobs, random=random)
    # Make sure have same commands overall, delete once we trust strat_split
    assert sorted(np.concatenate(subcommands)) == sorted(sum(list(dry_run_commands.values()), []))

    job_suffix = run_uuid.hex[:UUID_JOB_CHARS]

    # Include comments as reproducibility lines
    args_str = serializable_dict(args)
    fp.write("# running: %s\n" % str(args_str))
    fp.write("# cmd: %s\n" % cmd.cmd_str())
    for ii, ii_str in range_str(n_jobs):
        assert len(subcommands[ii]) > 0
        fp.write("job_%s_%s %s\n" % (job_suffix, ii_str, " && ".join(subcommands[ii])))


def real_run(args, opt_file_lookup, run_uuid, timeout=None):  # pragma: io
    """Run sequence of independent experiments to fully run the benchmark.

    This uses `subprocess` to launch a separate process (in serial) for each experiment.

    Parameters
    ----------
    args : dict(CmdArgs, [int, str])
        Arguments of options to pass to the experiments being launched. The keys corresponds to the same arguments
        passed to this program.
    opt_file_lookup : dict(str, str)
        Mapping from method name to filename containing wrapper class for the method.
    run_uuid : uuid.UUID
        UUID for this launcher run. Needed to generate different experiments UUIDs on each call. This function is
        deterministic provided the same `run_uuid`.
    timeout : int
        Max seconds per experiment
    """
    args[CmdArgs.db] = XRSerializer.init_db(args[CmdArgs.db_root], db=args[CmdArgs.db], keys=EXP_VARS, exist_ok=True)
    logger.info("Supply --db %s to append to this experiment or reproduce jobs file." % args[CmdArgs.db])

    # Get and run the commands in a sub-process
    counter = 0
    G = gen_commands(args, opt_file_lookup, run_uuid)
    for _, full_cmd in G:
        try:
            status = call(full_cmd, shell=False, cwd=args[CmdArgs.optimizer_root], timeout=timeout)
            if status != 0:
                raise ChildProcessError("status code %d returned from:\n%s" % (status, " ".join(full_cmd)))
        except TimeoutExpired:
            logger.info(f"Experiment timeout after {timeout} seconds.")
            print(json.dumps({"experiment_timeout_exception": " ".join(full_cmd)}))

        counter += 1
    logger.info(f"Benchmark script ran {counter} studies successfully.")


def main():
    """See README for instructions on calling launcher.
    """
    description = "Launch series of studies across functions and optimizers"
    args = cmd.parse_args(cmd.launcher_parser(description))

    logger.setLevel(logging.INFO)  # Note this is the module-wide logger
    if args[CmdArgs.verbose]:
        logger.addHandler(logging.StreamHandler())

    # Get optimizer settings, says which file to call for each optimizer
    settings = cmd.load_optimizer_settings(args[CmdArgs.optimizer_root])
    opt_file_lookup = {optimizer: wrapper_file for optimizer, (wrapper_file, _) in settings.items()}

    # Setup uuid
    if args[CmdArgs.uuid] is None:
        args[CmdArgs.uuid] = pyuuid.uuid4().hex  # debatable if uuid1 or uuid4 is better here
    else:
        warnings.warn(
            "User UUID supplied. This is only desired for debugging. Careless use could lead to study id conflicts.",
            UserWarning,
        )
    run_uuid = pyuuid.UUID(hex=args[CmdArgs.uuid])
    assert run_uuid.hex == args[CmdArgs.uuid]
    logger.info("Supply --uuid %s to reproduce this run." % run_uuid.hex)

    # Log all the options
    print("Launcher options (JSON):\n")
    print(json.dumps({"bayesmark-launch-args": cmd.serializable_dict(args)}))
    print("\n")

    # Set the master seed (derive from the uuid we just setup)
    pyrandom.seed(run_uuid.int)
    np.random.seed(random_seed(pyrandom))

    # Now run it, either to dry run file or executes sub-processes
    if args[CmdArgs.dry_run]:
        with absopen(args[CmdArgs.jobs_file], "w") as fp:
            dry_run(args, opt_file_lookup, run_uuid, fp)
    else:
        timeout = args[CmdArgs.timeout] if args[CmdArgs.timeout] > 0 else None
        real_run(args, opt_file_lookup, run_uuid, timeout)

    logger.info("done")


if __name__ == "__main__":
    main()  # pragma: main
