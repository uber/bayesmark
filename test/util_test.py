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
import random as pyrandom

from hypothesis import assume, given
from hypothesis.strategies import floats, integers, iterables, lists, text

from bo_benchmark import util as bobm_util
from util import _hashable


def some_mock_f(x):
    """Some arbitrary deterministic test function.
    """
    random_stream = pyrandom.Random(hash(x))
    y = random_stream.gauss(0, 1)
    return y


@given(lists(_hashable()))
def test_all_unique(L):
    bobm_util.all_unique(L)


@given(lists(integers(), unique=True) | lists(text(), unique=True) | lists(floats(), unique=True))
def test_strict_sorted(L):
    bobm_util.strict_sorted(L)


@given(integers(-5, 1000))
def test_range_str(stop):
    list(bobm_util.range_str(stop))


@given(text(), lists(text()))
def test_str_join_safe(delim, str_vec):
    assume(not any(delim in ss for ss in str_vec))
    bobm_util.str_join_safe(delim, str_vec, append=False)


@given(text(), lists(text()), lists(text()))
def test_str_join_safe_append(delim, str_vec0, str_vec):
    assume(not any(delim in ss for ss in str_vec0))
    assume(not any(delim in ss for ss in str_vec))

    start = bobm_util.str_join_safe(delim, str_vec0, append=False)
    bobm_util.str_join_safe(delim, [start] + str_vec, append=True)


@given(text(), text(min_size=1))
def test_chomp(str_val, ext):
    bobm_util.chomp(str_val + ext, ext)


@given(iterables(_hashable()))
def test_preimage_func(x):
    bobm_util.preimage_func(some_mock_f, x)
