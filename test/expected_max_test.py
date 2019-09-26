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
from hypothesis.strategies import floats, integers, lists

import bayesmark.expected_max as em


@given(integers(1, 10), integers(1, 10))
def test_get_expected_max_weights(n, m):
    pdf = em.get_expected_max_weights(n, m)
    assert pdf is not None


@given(lists(floats()), integers(1, 10))
def test_expected_max(x, m):
    E_max_x = em.expected_max(x, m)
    assert E_max_x is not None


@given(lists(floats()), integers(1, 10))
def test_expected_min(x, m):
    E_min_x = em.expected_min(x, m)
    assert E_min_x is not None
