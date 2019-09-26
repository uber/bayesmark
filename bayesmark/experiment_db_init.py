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
"""Tool to create new datebase for results. This is just wrapper on serializer init call.
"""
import logging

import bayesmark.cmd_parse as cmd
from bayesmark.cmd_parse import CmdArgs
from bayesmark.constants import EXP_VARS
from bayesmark.serialize import XRSerializer

EXIST_OK = True

logger = logging.getLogger(__name__)


def main():
    """See README for instructions on calling db_init.
    """
    description = "Initialize the directories for running the experiments"
    args = cmd.parse_args(cmd.general_parser(description))

    assert not args[CmdArgs.dry_run], "Dry run doesn't make any sense when building dirs"

    logger.setLevel(logging.INFO)  # Note this is the module-wide logger
    if args[CmdArgs.verbose]:
        logger.addHandler(logging.StreamHandler())

    XRSerializer.init_db(args[CmdArgs.db_root], db=args[CmdArgs.db], keys=EXP_VARS, exist_ok=EXIST_OK)

    logger.info("done")


if __name__ == "__main__":
    main()  # pragma: main
