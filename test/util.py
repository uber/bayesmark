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
import numpy as np
from hypothesis import assume
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import (
    binary,
    booleans,
    composite,
    dictionaries,
    floats,
    from_regex,
    frozensets,
    integers,
    lists,
    sampled_from,
    text,
)
from hypothesis_gufunc.extra.xr import fixed_dataarrays, simple_coords, xr_coords

import bo_benchmark.space as sp
from bo_benchmark.constants import ARG_DELIM, ITER, METHOD, RANDOM_SEARCH, SUGGEST, TEST_CASE, TRIAL
from bo_benchmark.np_util import linear_rescale

NULL_PLUG = "\x00"


def _easy_text():
    # The NULL_PLUG confuses numpy arrays, so assume that is not in
    S = text().filter(lambda ss: NULL_PLUG not in ss)
    return S


def _hashable():
    S = floats() | integers() | _easy_text()
    return S


CAT_STGY = _easy_text if sp.CAT_KIND == "U" else binary

F_MIN = np.nextafter(0, 1)

RANGES = {"linear": (-1000, 1000), "log": (F_MIN, 1000), "logit": (F_MIN, np.nextafter(1, 0)), "bilog": (-100, 100)}

SPACES = tuple(sorted(sp.SPACE_DICT.keys()))


@composite
def space_vars(draw, max_values=5):
    """Build composite strategy for random API calls."""
    type_ = draw(sampled_from(SPACES))
    use_values = draw(booleans())

    if type_ == "real":
        warp = draw(sampled_from(("linear", "log", "logit", "bilog")))
        min_val, max_val = RANGES[warp]
        if use_values:
            # Generating unique values to ensure that always have more than 2
            # unique values, but code is designed to accept non-unique values
            # arrays as long as more than 2 non-unique. Could generalize this.
            values = draw(lists(floats(min_val, max_val), min_size=2, max_size=max_values, unique=True))
            D = {"type": type_, "space": warp, "values": values}
        else:
            range_ = tuple(sorted(draw(lists(floats(min_val, max_val), min_size=2, max_size=2, unique=True))))
            D = {"type": type_, "space": warp, "range": range_}
    elif type_ == "int":
        warp = draw(sampled_from(("linear", "log", "bilog")))
        min_val, max_val = RANGES[warp]
        # Must shrink these to next integers in range to keep hypothesis happy
        min_val = int(np.ceil(min_val))
        max_val = int(np.floor(max_val))
        if use_values:
            values = draw(lists(integers(min_val, max_val), min_size=2, max_size=max_values, unique=True))
            D = {"type": type_, "space": warp, "values": values}
        else:
            range_ = tuple(sorted(draw(lists(integers(min_val, max_val), min_size=2, max_size=2, unique=True))))
            D = {"type": type_, "space": warp, "range": range_}
    elif type_ == "bool":
        D = {"type": type_}
    elif type_ == "cat" or type_ == "ordinal":
        values = draw(lists(CAT_STGY(), min_size=2, max_size=max_values, unique=True))
        # This assume is needed because np.unique has bug for null plug
        # .. >>> np.unique([u'', u'\x00'])
        # .. array([u''], dtype='<U1')
        assume(len(np.unique(values)) == len(values))
        D = {"type": type_, "values": values}
    else:
        assert False

    return D


@composite
def space_configs(draw, max_vars=5, max_len=5, allow_missing=False, unique_y=False):
    meta = draw(dictionaries(text(), space_vars(), min_size=1, max_size=max_vars))

    S = sp.JointSpace(meta)
    lower, upper = S.get_bounds().T

    D = sum(len(var["values"]) if var["type"] in ("cat", "ordinal") else 1 for var in meta.values())

    # Let's draw warped variable because that will be a lot easier
    N = draw(integers(min_value=0, max_value=max_len))
    X_w = draw(arrays(dtype=float, shape=(N, D), elements=floats(min_value=0.0, max_value=1.0)))
    X_w = linear_rescale(X_w, lb0=0.0, ub0=1.0, lb1=lower, ub1=upper)
    X = S.unwarp(X_w)

    # Draw output too in case we want it
    y_elements = floats(allow_infinity=False, allow_nan=allow_missing)
    y = draw(arrays(dtype=float, shape=(N,), elements=y_elements, unique=unique_y))

    # Draw the fixed vars
    X_fixed_w = draw(arrays(dtype=float, shape=(1, D), elements=floats(min_value=0.0, max_value=1.0)))
    X_fixed_w = linear_rescale(X_fixed_w, lb0=0.0, ub0=1.0, lb1=lower, ub1=upper)
    X_fixed, = S.unwarp(X_fixed_w)

    # Make fixed_vars a subset of all vars.
    keep_in_fixed = draw(frozensets(sampled_from(tuple(X_fixed.keys()))))
    X_fixed = {k: X_fixed[k] for k in keep_in_fixed}

    return meta, X, y, X_fixed


_test_cases = _easy_text


def perf_dataarrays(min_trial=1):
    dims = (ITER, SUGGEST, TEST_CASE, METHOD, TRIAL)
    # Don't get too close to infinity because that can also create issues and isn't supported
    elements = floats(allow_nan=False, min_value=-1e300, max_value=1e300)

    ref = RANDOM_SEARCH + ARG_DELIM
    method_names = from_regex("^%s[A-Z]*" % ref) | text()
    method_st = xr_coords(elements=method_names).filter(lambda L: any(ss.startswith(ref) for ss in L))

    coords_st = {
        ITER: simple_coords(min_side=1),
        SUGGEST: simple_coords(min_side=1),
        TRIAL: simple_coords(min_side=min_trial),
        METHOD: method_st,
    }
    S = fixed_dataarrays(dims, dtype=np.float_, elements=elements, coords_elements=_test_cases(), coords_st=coords_st)
    return S
