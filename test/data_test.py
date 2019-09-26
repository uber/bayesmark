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
from hypothesis import given
from hypothesis.strategies import from_regex, sampled_from

from bayesmark import data

DATA_NAMES = sorted(data.DATA_LOADERS.keys())


@given(sampled_from(DATA_NAMES) | from_regex("^reg-[A-Z]*") | from_regex("^clf-[A-Z]*"))
def test_get_problem_type(dataset_name):
    problem_type = data.get_problem_type(dataset_name)
    assert problem_type is not None
