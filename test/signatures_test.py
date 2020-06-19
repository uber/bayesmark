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
import json
import random as pyrandom
import warnings

import numpy as np
from hypothesis import given
from hypothesis.strategies import dictionaries, floats, lists, text, tuples

import bayesmark.signatures as ss
from bayesmark.experiment import OBJECTIVE_NAMES
from util import space_configs

N_SIG = ss.N_SUGGESTIONS


def bsigs():
    S = lists(floats(allow_infinity=False, allow_nan=False), min_size=N_SIG, max_size=N_SIG)
    return S


def sigs():
    S = lists(bsigs(), min_size=1)
    return S


def sig_pair():
    def separate(D):
        signatures, signatures_ref = {}, {}
        for kk in D:
            if len(D[kk]) == 1:
                v_ref, = D[kk]
                signatures_ref[kk] = np.asarray(v_ref)
            elif len(D[kk]) == 2:
                v, v_ref = D[kk]
                signatures[kk] = np.asarray(v)
                signatures_ref[kk] = np.asarray(v_ref)
            else:
                assert False
        return signatures, signatures_ref

    sig_dict = dictionaries(text(), tuples(bsigs()) | tuples(bsigs(), bsigs()))
    S = sig_dict.map(separate)
    return S


def some_mock_f(x):
    """Some arbitrary deterministic test function.
    """
    random_stream = pyrandom.Random(json.dumps(x, sort_keys=True))
    y = [random_stream.gauss(0, 1) for _ in OBJECTIVE_NAMES]
    return y


@given(space_configs())
def test_get_func_signature(api_config):
    api_config, _, _, _ = api_config

    signature_x, signature_y = ss.get_func_signature(some_mock_f, api_config)


@given(dictionaries(text(), sigs()))
def test_analyze_signatures(signatures):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        sig_errs, signatures_median = ss.analyze_signatures(signatures)


@given(sig_pair())
def test_analyze_signature_pair(args):
    signatures, signatures_ref = args
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        sig_errs, signatures_pair = ss.analyze_signature_pair(signatures, signatures_ref)
