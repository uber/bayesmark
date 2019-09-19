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
from collections import OrderedDict
from itertools import product

import xarray as xr
from hypothesis import assume, given, settings
from hypothesis.strategies import dictionaries, floats, integers, just, sampled_from, tuples
from hypothesis_gufunc.extra.xr import (
    _hashable,
    dataarrays,
    datasets,
    fixed_datasets,
    simple_dataarrays,
    subset_lists,
    vars_to_dims_dicts,
    xr_vars,
)

import bo_benchmark.xr_util as xru

xr_fill = _hashable


def intersect_seq(L):
    if len(L) == 0:
        return set([])

    S = set(L[0])
    for xx in L[1:]:
        S = S & set(xx)
    return S


def ds_vars_dims():
    def build_it(vars_to_dims_):
        all_dims = list(set(sum((list(dd) for dd in vars_to_dims_.values()), [])))

        ds = fixed_datasets(vars_to_dims_)
        vars_ = subset_lists(list(vars_to_dims_.keys()))
        dims = subset_lists(all_dims)
        return tuples(ds, vars_, dims)

    vars_to_dims_st = vars_to_dims_dicts()

    S = vars_to_dims_st.flatmap(build_it)
    return S


def ds_vars_dims_mixed():
    def build_it(vars_to_dims_):
        all_dims = list(set(sum((list(dd) for dd in vars_to_dims_.values()), [])))

        ds = fixed_datasets(vars_to_dims_)

        dims = subset_lists(all_dims)

        vars_ = sampled_from(list(vars_to_dims_.keys()))
        vars_dict = dictionaries(vars_, dims, dict_class=OrderedDict)
        vars_dict = vars_dict.map(OrderedDict.items).map(list)

        return tuples(ds, vars_dict, just(all_dims))

    vars_to_dims_st = vars_to_dims_dicts(min_vars=0, min_dims=0)

    S = vars_to_dims_st.flatmap(build_it)
    return S


@given(simple_dataarrays(("foo", "bar", "baz")) | dataarrays() | dataarrays(coords_elements=floats()), integers(0, 3))
def test_is_simple_coords(da, min_side):
    xru.is_simple_coords(da.coords, min_side=min_side)


@given(simple_dataarrays(("foo", "bar", "baz")))
def test_is_simple_coords_pass(da):
    simple = xru.is_simple_coords(da.coords)
    assert simple


@given(ds_vars_dims(), xr_fill())
def test_ds_like(args, fill):
    ref, vars_, dims = args

    xru.ds_like(ref, vars_, dims, fill=fill)


@given(ds_vars_dims_mixed(), xr_fill())
def test_ds_like_mixed(args, fill):
    ref, vars_, dims = args

    xru.ds_like_mixed(ref, vars_, dims, fill=fill)


@given(xr_vars(), dataarrays())
def test_only_dataarray(var_, da):
    assume(var_ not in da.dims)

    ds = xr.Dataset({var_: da})

    xru.only_dataarray(ds)


@given(datasets())
def test_coord_compat(ds):
    all_dims = [ds[kk].dims for kk in ds]
    common_dims = sorted(intersect_seq(all_dims))
    da_seq = [ds[kk] for kk in ds]

    compat = xru.coord_compat(da_seq, common_dims)
    assert compat


@given(datasets())
def test_coord_compat_false(ds):
    all_dims = [ds[kk].dims for kk in ds]
    common_dims = sorted(intersect_seq(all_dims))
    da_seq = [ds[kk] for kk in ds]

    assume(len(da_seq) > 0)
    assume(len(da_seq[0].dims) > 0)

    da = da_seq[0]
    kk = da.dims[0]
    da_seq[0] = da.assign_coords(**{kk: range(da.sizes[kk])})

    xru.coord_compat(da_seq, common_dims)


@given(dataarrays(min_dims=1, max_dims=1))
def test_da_to_string(da):
    xru.da_to_string(da)


@given(dataarrays(min_side=0, min_dims=0), integers(1, 3))
@settings(deadline=None)
def test_da_concat(da, n):
    assume(n < len(da.dims))

    da_dict, keys_to_slice = da_split(da, n)
    assume(len(da_dict) > 0)
    assert len(keys_to_slice) == n

    xru.da_concat(da_dict, dims=keys_to_slice)


def da_split(da, n):
    assert 0 < n
    assert n <= len(da.dims)

    keys_to_slice = da.dims[-n:]
    da_dict = {}
    vals = [da.coords[kk].values.tolist() for kk in keys_to_slice]
    for vv in product(*vals):
        lookup = dict(zip(keys_to_slice, vv))
        da_dict[tuple(vv)] = da.sel(lookup, drop=True)
    return da_dict, keys_to_slice


@given(datasets(min_side=1, min_dims=1), integers(1, 3))
@settings(deadline=None)
def test_ds_concat(ds, n):
    all_dims = [ds[kk].dims for kk in ds]
    common_dims = sorted(intersect_seq(all_dims))

    n = min([n, len(common_dims) - 1])
    assume(0 < n)

    keys_to_slice = common_dims[:n]
    ds_dict = {}
    vals = [ds.coords[kk].values.tolist() for kk in keys_to_slice]
    for vv in product(*vals):
        lookup = dict(zip(keys_to_slice, vv))
        ds_dict[vv] = ds.sel(lookup, drop=True)

    xru.ds_concat(ds_dict, dims=keys_to_slice)
