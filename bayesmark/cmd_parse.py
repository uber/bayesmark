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
"""Anything related to parsing command line arguments goes in here. There are some custom data structures to
represent all the options available in the experiments here.

Not currently any doc strings in this file because it may become obsolete with the use of fire package.
"""
import argparse
import json
import os.path
import sys
import uuid as pyuuid
from enum import IntEnum, auto
from pathlib import PosixPath

import git
from git.exc import InvalidGitRepositoryError
from pathvalidate.argparse import sanitize_filename, validate_filename, validate_filepath

from bayesmark.builtin_opt.config import CONFIG
from bayesmark.constants import ARG_DELIM, DATA_LOADER_NAMES, METRICS, MODEL_NAMES, OPTIMIZERS_FILE, PY_INTERPRETER
from bayesmark.path_util import absopen, abspath
from bayesmark.util import shell_join

assert not any(ARG_DELIM in opt for opt in MODEL_NAMES)
assert not any(ARG_DELIM in opt for opt in DATA_LOADER_NAMES)


class CmdArgs(IntEnum):
    uuid = auto()
    db_root = auto()
    optimizer_root = auto()
    data_root = auto()
    db = auto()
    optimizer = auto()
    data = auto()
    classifier = auto()
    metric = auto()
    n_calls = auto()
    n_suggest = auto()
    n_repeat = auto()
    n_jobs = auto()
    jobs_file = auto()
    ravel = auto()
    verbose = auto()
    dry_run = auto()
    rev = auto()
    opt_rev = auto()
    timeout = auto()


CMD_STR = {
    CmdArgs.uuid: ("-u", "--uuid"),
    CmdArgs.db_root: ("-dir", "-db-root"),
    CmdArgs.optimizer_root: ("-odir", "--opt-root"),
    CmdArgs.data_root: ("-dr", "--data-root"),
    CmdArgs.db: ("-b", "--db"),
    CmdArgs.optimizer: ("-o", "--opt"),
    CmdArgs.data: ("-d", "--data"),
    CmdArgs.classifier: ("-c", "--classifier"),
    CmdArgs.metric: ("-m", "--metric"),
    CmdArgs.n_calls: ("-n", "--calls"),
    CmdArgs.n_suggest: ("-p", "--suggestions"),
    CmdArgs.n_repeat: ("-r", "--repeat"),
    CmdArgs.n_jobs: ("-nj", "--num-jobs"),
    CmdArgs.jobs_file: ("-ofile", "--jobs-file"),
    CmdArgs.ravel: ("-rv", "--ravel"),
    CmdArgs.verbose: ("-v", "--verbose"),
    CmdArgs.timeout: ("-t", "--timeout"),
    CmdArgs.dry_run: (None, "dry_run"),  # Will not be specified from CLI
    CmdArgs.rev: (None, "rev"),  # Will not be specified from CLI
    CmdArgs.opt_rev: (None, "opt_rev"),  # Will not be specified from CLI. Which version of optimizer.
}


def arg_to_str(arg):
    # We can change this so it is arg.value, or someway to be usable by field interface
    _, dest = str(arg).split(".")
    return dest


def namespace_to_dict(args_ns):
    args = vars(args_ns)
    args = {kk: args[arg_to_str(kk)] for kk in CMD_STR if (arg_to_str(kk) in args)}
    return args


def serializable_dict(args):
    args_str = {CMD_STR[kk][1]: args[kk] for kk in CMD_STR if (kk in args)}
    assert len(args_str) == len(args)
    return args_str


def unserializable_dict(args_str):
    args = {kk: args_str[CMD_STR[kk][1]] for kk in CMD_STR if (CMD_STR[kk][1] in args_str)}
    assert len(args_str) == len(args)
    return args


def add_argument(parser, arg, **kwargs):
    short_name, long_name = CMD_STR[arg]
    dest = arg_to_str(arg)
    parser.add_argument(short_name, long_name, dest=dest, **kwargs)


def filepath(value):
    """Work around for `pathvalidate` bug."""
    if value == ".":
        return value
    validate_filepath(value, platform="auto")
    return value


def filename(value):
    validate_filename(value, platform="universal")
    return value


def uuid(val_str):
    val = str.lower(val_str)
    uuid_ = pyuuid.UUID(hex=val)
    assert val == uuid_.hex, "error in parsing uuid"
    return val


def positive_int(val_str):
    val = int(val_str)
    if val <= 0:
        msg = "expected positive, got %s" % val_str
        raise argparse.ArgumentTypeError(msg)
    return val


def joinable(val_str):
    val = str(val_str)  # just for good measure
    validate_filename(val, platform="universal")  # we choose to be at least as strict as filenames
    if ARG_DELIM in val:
        msg = "delimiter %s not allowed in choice %s" % (ARG_DELIM, val)
        raise argparse.ArgumentTypeError(msg)
    return val


def load_rev_number():
    # This function uses a lot of language "power features" that could be considered bad form:
    # 1) does a conditional import to get version
    # 2) uses __file__ to try and extract and git repo version during execution
    # We will let this fly anyway because:
    # 1) The results of this are only used for logging anyway
    # 2) This is a command parsing module of the code and inherently very non-pure and doing IO etc
    # 3) Unclear if there is a cleaner way to do this

    # Get rev from version file (if running inside the pip-installable wheel without the git repo)
    try:
        from bayesmark import version

        rev_file = version.VERSION
    except ImportError:
        rev_file = None
    else:
        rev_file = rev_file.strip()
        rev = rev_file

    # Get rev from git API if inside git repo (and not built wheel from pip install ...)
    wdir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    try:
        repo = git.Repo(path=wdir, search_parent_directories=False)
    except InvalidGitRepositoryError:
        rev_repo = None
    else:
        rev_repo = repo.head.commit.hexsha
        rev_repo = rev_repo.strip()
        rev = rev_repo

    # Check coherence of what we found
    if (rev_repo is None) and (rev_file is None):
        raise RuntimeError("Must specify version.py if not inside a git repo.")
    if (rev_repo is not None) and (rev_file is not None):
        assert rev_repo == rev_file, "Rev file %s does not match rev git %s" % (rev_file, rev_repo)

    assert rev == rev.strip()
    # We could first enforce is_lower_hex if we want to enforce that
    rev = rev[:7]
    return rev


def base_parser():
    parser = argparse.ArgumentParser(add_help=False)

    add_argument(
        parser, CmdArgs.db_root, default=".", type=filepath, help="root directory for all benchmark experiments output"
    )
    add_argument(
        parser, CmdArgs.optimizer_root, default=".", type=filepath, help="Directory with optimization wrappers"
    )
    # Always a verbose flag option
    add_argument(parser, CmdArgs.verbose, action="store_true", help="print the study logs to console")
    return parser


def launcher_parser(description):
    parser = argparse.ArgumentParser(description=description, parents=[base_parser()])

    add_argument(parser, CmdArgs.uuid, type=uuid, help="length 32 hex UUID for this experiment")
    add_argument(parser, CmdArgs.data_root, type=filepath, help="root directory for all custom csv files")
    add_argument(parser, CmdArgs.db, type=filename, help="database ID of this benchmark experiment")

    add_argument(parser, CmdArgs.optimizer, type=joinable, nargs="+", help="optimizers to use")
    add_argument(parser, CmdArgs.data, type=joinable, nargs="+", help="data sets to use")
    add_argument(parser, CmdArgs.classifier, type=joinable, nargs="+", help="classifiers to use")
    add_argument(parser, CmdArgs.metric, type=str, choices=METRICS, nargs="+", help="scoring metric to use")

    # Iterations counts used in experiments
    add_argument(parser, CmdArgs.n_calls, default=100, type=positive_int, help="number of function evaluations")
    add_argument(
        parser, CmdArgs.n_suggest, default=1, type=positive_int, help="number of suggestions to provide in parallel"
    )
    add_argument(parser, CmdArgs.n_repeat, default=20, type=positive_int, help="number of repetitions of each study")
    add_argument(parser, CmdArgs.timeout, default=0, type=int, help="Timeout per experiment (0 = no timeout)")

    # Arguments for creating dry run jobs file
    add_argument(
        parser,
        CmdArgs.n_jobs,
        type=int,
        default=0,
        help="number of jobs to put in the dry run file, the default 0 value disables dry run (real run)",
    )
    # Using default of current dir for jobs file output since that is generally the default for everything
    add_argument(
        parser, CmdArgs.jobs_file, type=filepath, default="./jobs.txt", help="a jobs file with all commands to be run"
    )
    return parser


def experiment_parser(description):
    parser = argparse.ArgumentParser(description=description, parents=[base_parser()])

    add_argument(parser, CmdArgs.uuid, type=uuid, required=True, help="length 32 hex UUID for this experiment")

    # This could be made simpler and use '.' default for dataroot, even if no custom data used.
    add_argument(parser, CmdArgs.data_root, type=filepath, help="root directory for all custom csv files")
    add_argument(parser, CmdArgs.db, type=filename, required=True, help="database ID of this benchmark experiment")
    add_argument(parser, CmdArgs.optimizer, required=True, type=joinable, help="optimizer to use")
    add_argument(parser, CmdArgs.data, required=True, type=joinable, help="data set to use")
    add_argument(parser, CmdArgs.classifier, required=True, type=joinable, help="classifier to use")
    add_argument(parser, CmdArgs.metric, required=True, type=str, choices=METRICS, help="scoring metric to use")

    add_argument(parser, CmdArgs.n_calls, default=100, type=positive_int, help="number of function evaluations")
    add_argument(
        parser, CmdArgs.n_suggest, default=1, type=positive_int, help="number of suggestions to provide in parallel"
    )
    return parser


def agg_parser(description):
    parser = argparse.ArgumentParser(description=description, parents=[base_parser()])
    add_argument(parser, CmdArgs.db, type=filename, required=True, help="database ID of this benchmark experiment")
    add_argument(
        parser,
        CmdArgs.ravel,
        action="store_true",
        help="ravel all studies to store batch suggestions as if they were serial (deprecated)",
    )
    return parser


def general_parser(description):
    parser = argparse.ArgumentParser(description=description, parents=[base_parser()])
    add_argument(parser, CmdArgs.db, type=filename, required=True, help="database ID of this benchmark experiment")
    return parser


def parse_args(parser, argv=None):
    """Note that this argument parser does not check compatibility between clf/reg metric and data set.
    """
    args = parser.parse_args(argv)
    args = namespace_to_dict(args)

    args[CmdArgs.dry_run] = (CmdArgs.n_jobs in args) and (args[CmdArgs.n_jobs] > 0)
    # Does not check dir actually exists here, but whatever
    args[CmdArgs.jobs_file] = abspath(args[CmdArgs.jobs_file], verify=False) if args[CmdArgs.dry_run] else None

    # Then make sure all path vars are abspath:
    # Dry run might be executing on diff system => cannot verify yet
    args[CmdArgs.db_root] = abspath(args[CmdArgs.db_root], verify=not args[CmdArgs.dry_run])
    args[CmdArgs.optimizer_root] = abspath(args[CmdArgs.optimizer_root], verify=True)
    if (CmdArgs.data_root in args) and (args[CmdArgs.data_root] is not None):
        args[CmdArgs.data_root] = abspath(args[CmdArgs.data_root], verify=not args[CmdArgs.dry_run])

    # Get git version of the benchmark itself for meta-data, just in case we need it.
    args[CmdArgs.rev] = load_rev_number()

    # We may support ability to specify version at args in the future, from now it is implied
    args[CmdArgs.opt_rev] = None
    return args


def _cleanup(filename_str):
    filename_str = sanitize_filename(filename_str, replacement_text="-", platform="universal")
    filename_str = filename_str.replace(ARG_DELIM, "-")
    return filename_str


def infer_settings(opt_root, opt_pattern="**/optimizer.py"):
    opt_root = PosixPath(opt_root)
    assert opt_root.is_dir(), "Opt root directory doesn't exist: %s" % opt_root
    assert opt_root.is_absolute(), "Only absolute path should have even gotten this far."

    # Always sort for reproducibility
    source_files = sorted(opt_root.glob(opt_pattern))
    source_files = [ss.relative_to(opt_root) for ss in source_files]

    settings = {_cleanup(str(ss.parent)): [str(ss), {}] for ss in source_files}

    assert all(joinable(kk) for kk in settings), "Something went wrong in name sanitization."
    assert len(settings) == len(source_files), "Name collision after sanitization of %s" % repr(source_files)
    assert len(set(CONFIG.keys()) & set(settings.keys())) == 0, "Name collision with builtin optimizers."

    return settings


def load_optimizer_settings(opt_root):
    try:
        with absopen(os.path.join(opt_root, OPTIMIZERS_FILE), "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        # Search for optimizers instead
        settings = infer_settings(opt_root)

    assert isinstance(settings, dict)
    assert not any((ARG_DELIM in opt) for opt in settings), "optimizer names violates name convention"
    return settings


def cmd_str():
    cmd = "%s %s" % (PY_INTERPRETER, shell_join(sys.argv))
    return cmd
