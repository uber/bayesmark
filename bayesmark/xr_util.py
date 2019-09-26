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
"""General utilities for `xarray` that should be included in `xarray`.
"""
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr

from bayesmark.util import all_unique


def is_simple_coords(coords, min_side=0, dims=None):
    """Check if all xr coordinates are "simple". That is, equals to ``np.arange(n)``.

    Parameters
    ----------
    coords : dict-like of coordinates
        The coordinates we would like to check, e.g. from ``DataArray.coords``.
    min_side : int
        The minimum side requirement. We can set this ``min_side=1`` and have empty coordinates result in a return
        value of ``False``.
    dims : None or list of dimension names
        Dimensions we want to check for simplicity. If ``None``, check all dimensions.

    Returns
    -------
    simple : bool
        True when all coordinates are simple.
    """
    for kk in coords:
        if (dims is None) or (kk in dims):
            C = coords[kk].values
            # Not checking dtype on empty coords, could check that too if we want to be strict
            if len(C) > 0 and C.dtype != np.int_:
                return False

            C = C.tolist()
            if len(C) < min_side:
                return False
            if C != list(range(len(C))):
                return False
    return True


def ds_like(ref, vars_, dims, fill=np.nan):
    """Produce a blank :class:`xarray:xarray.Dataset` copying some coordinates from another
    :class:`xarray:xarray.Dataset`.

    Parameters
    ----------
    ref : :class:`xarray:xarray.Dataset`
        The reference dataset we want to copy coordinates from.
    vars_ : typing.Iterable
        List of variable names we want in the new dataset.
    dims : list
        List of dimensions we want to copy over from `ref`. These are the dimensions of the output.
    fill : scalar
        Scalar value to fill the blank dataset. The `dtype` will be determined from the `fill` value.

    Returns
    -------
    ds : :class:`xarray:xarray.Dataset`
        A new dataset with variables `vars_` and dimensions `dims` where the coordinates have been copied from `ref`.
        All values are filled with `fill`.
    """
    size = [ref.sizes[dd] for dd in dims]

    # Use OrderedDict for good measure, probably not needed
    data = OrderedDict([(vv, (dims, np.full(size, fill))) for vv in vars_])
    coords = OrderedDict([(dd, ref.coords[dd].values) for dd in dims])
    ds = xr.Dataset(data, coords=coords)
    return ds


def ds_like_mixed(ref, vars_, dims, fill=np.nan):
    """The same as `ds_like` but allow different dimensions for each variable.

    Parameters
    ----------
    ref : :class:`xarray:xarray.Dataset`
        The reference dataset we want to copy coordinates from.
    vars_ : typing.Iterable
        List of (variable names, dimension) pairs we want in the new dataset. The dimensions for each variable must be
        a subset of `dims`.
    dims : list
        List of all dimensions we want to copy over from `ref`.
    fill : scalar
        Scalar value to fill the blank dataset. The `dtype` will be determined from the `fill` value.

    Returns
    -------
    ds : :class:`xarray:xarray.Dataset`
        A new dataset with variables `vars_` and dimensions `dims` where the coordinates have been copied from `ref`.
        All values are filled with `fill`.
    """
    coords = OrderedDict([(dd, ref.coords[dd].values) for dd in dims])

    data = OrderedDict()
    for var_name, var_dims in vars_:
        assert set(var_dims).issubset(dims)
        size = [ref.sizes[dd] for dd in var_dims]
        data[var_name] = (var_dims, np.full(size, np.nan))
    ds = xr.Dataset(data, coords=coords)
    return ds


def only_dataarray(ds):
    """Convert a :class:`xarray:xarray.Dataset` to a :class:`xarray:xarray.DataArray`. If the
    :class:`xarray:xarray.Dataset` has more than one variable, an error is raised.

    Parameters
    ----------
    ds : :class:`xarray:xarray.Dataset`
        :class:`xarray:xarray.Dataset` we would like to convert to a :class:`xarray:xarray.DataArray`. This must
        contain only one variable.

    Returns
    -------
    da : :class:`xarray:xarray.DataArray`
        The :class:`xarray:xarray.DataArray` extracted from `ds`.
    """
    name, = ds
    da = ds[name]
    return da


def coord_compat(da_seq, dims):
    """Check if a sequence of :class:`xarray:xarray.DataArray` have compatible coordinates.

    Parameters
    ----------
    da_seq : list(:class:`xarray:xarray.DataArray`)
        Sequence of :class:`xarray:xarray.DataArray` we would like to check for compatibility.
        :class:`xarray:xarray.Dataset` work too.
    dims : list
        Subset of all dimensions in the :class:`xarray:xarray.DataArray` we are concerned with for compatibility.

    Returns
    -------
    compat : bool
        True if all the :class:`xarray:xarray.DataArray` have compatible coordinates.
    """
    if len(da_seq) <= 1:
        return True

    ref = da_seq[0]
    for da in da_seq:
        # There is probably a better way to do this by attempting concat in try-except, but good enough for now:
        for dd in dims:
            assert dd in da.coords, "dim %s missing in dataarray" % dd
            if not np.all(ref.coords[dd].values == da.coords[dd].values):
                return False
    return True


def da_to_string(da):
    """Generate a human readable version of a 1D :class:`xarray:xarray.DataArray`.

    Parameters
    ----------
    da : :class:`xarray:xarray.DataArray`
        The :class:`xarray:xarray.DataArray` to display. Must only have one dimension.

    Returns
    -------
    str_val : str
        String with human readable version of `da`.
    """
    assert len(da.dims) == 1
    str_val = da.to_series().to_string()
    return str_val


def da_concat(da_dict, dims):
    """Concatenate a dictionary of :class:`xarray:xarray.DataArray` similar to :func:`pandas:pandas.concat`.

    Parameters
    ----------
    da_dict : dict(tuple(str), :class:`xarray:xarray.DataArray`)
        Dictionary of :class:`xarray:xarray.DataArray` to combine. The keys are tuples of index values. The
        :class:`xarray:xarray.DataArray` must have compatible coordinates.
    dims : list(str)
        The names of the new dimensions we create for the dictionary keys. This must be of the same length as the
        key tuples in `da_dict`.

    Returns
    -------
    da : :class:`xarray:xarray.DataArray`
        Combined data array. The new dimensions will be ``input_da.dims + dims``.
    """
    assert len(da_dict) > 0
    assert all(len(da.dims) > 0 for da in da_dict.values()), "0-dimensional DataArray not supported"
    assert all_unique(dims)

    cur_dims = list(da_dict.values())[0].dims
    assert all(da.dims == cur_dims for da in da_dict.values())
    assert len(set(cur_dims) & set(dims)) == 0

    def squeeze(tt):
        if len(tt) == 1:
            return tt[0]
        return tt

    D = OrderedDict([(squeeze(kk), da.to_series()) for kk, da in da_dict.items()])
    df = pd.concat(D, axis=1)

    assert df.columns.nlevels == len(dims)
    df.columns.names = dims

    df = df.stack(level=list(range(df.columns.nlevels)))
    assert isinstance(df, pd.Series)
    da = df.to_xarray()
    assert isinstance(da, xr.DataArray)
    return da


def ds_concat(ds_dict, dims):
    """Concatenate a dictionary of :class:`xarray:xarray.Dataset` similar to :func:`pandas:pandas.concat`, and a
    generalization of :func:`.da_concat`.

    Parameters
    ----------
    ds_dict : dict(tuple(str), :class:`xarray:xarray.DataArray`)
        Dictionary of :class:`xarray:xarray.Dataset` to combine. The keys are tuples of index values. The
        :class:`xarray:xarray.Dataset` must have compatible coordinates, and all have the same variables.
    dims : list(str)
        The names of the new dimensions we create for the dictionary keys. This must be of the same length as the
        key tuples in `ds_dict`.

    Returns
    -------
    ds : :class:`xarray:xarray.Dataset`
        Combined dataset. For each variable `var`, the new dimensions will be ``input_ds[var].dims + dims``.
    """
    assert len(ds_dict) > 0
    assert len(dims) > 0
    assert all(len(kk) == len(dims) for kk in ds_dict)

    # Get an arbitrary element as the reference
    k0 = list(ds_dict.keys())[0]

    # Check all vars the same
    vars_, = set([tuple(ds) for ds in ds_dict.values()])

    # Now combine da for each variable, one at a time
    ds = xr.Dataset(coords=ds_dict[k0].coords)
    for vv in vars_:
        da_dict = OrderedDict([(kk, da[vv]) for kk, da in ds_dict.items()])
        ds[vv] = da_concat(da_dict, dims)

    return ds
