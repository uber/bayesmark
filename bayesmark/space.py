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
"""Do the conversion of search spaces into a normalized cartesian space.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import expit as logistic  # because nobody calls it expit
from scipy.special import logit

from bayesmark.np_util import clip_chk, snap_to

WARPED_DTYPE = np.float_
N_GRID_DEFAULT = 8

# I can't make up mind of unicode or str is better wrt to Py 2/3 compatibility
# ==> Just make a global constant and make sure it works either way.
# Note: if we switch to np.str_, we will also need to update doc-strings!
CAT_DTYPE = np.unicode_
CAT_KIND = "U"
CAT_NATIVE_DTYPE = str
# Check to make sure consistent
assert CAT_KIND == np.dtype(CAT_DTYPE).kind
_infered = type(CAT_DTYPE("").item())
assert CAT_NATIVE_DTYPE == _infered

# ============================================================================
# These could go into util
# ============================================================================


def unravel_index(dims):
    """Builds tuple of coordinate arrays to traverse an `numpy` array.

    Wrapper around :func:`numpy:numpy.unravel_index` that avoids bug at corner case for ``dims=()``. The fix for this
    has been merged into the numpy master branch Oct 18, 2017 so future numpy releases will make this wrapper not
    needed. Otherwise, ``unravel_index(X.shape)`` is equivalent to: ``np.unravel_index(range(X.size), X.shape)``.

    Parameters
    ----------
    dims : tuple(int)
        The shape of the array to use for unraveling ``indices``.

    Returns
    -------
    unraveled_coords : tuple(:class:`numpy:numpy.ndarray`)
        Each array in the tuple has shape (n,) where ``n=np.prod(dims)``.

    References
    ----------
    unravel_index(0, ()) should return () (Trac #2120) #580
    https://github.com/numpy/numpy/issues/580
    Allow `unravel_index(0, ())` to return () #9884
    https://github.com/numpy/numpy/pull/9884
    """
    size = np.prod(dims)
    if dims == () or size == 0:  # The corner case
        return ()

    idx = np.unravel_index(range(np.prod(dims)), dims)
    return idx


def encode(X, labels, assume_sorted=False, dtype=bool, assume_valid=False):
    """Perform one hot encoding of categorical data in :class:`numpy:numpy.ndarray` variable `X` of any dimension.

    Parameters
    ----------
    X : :class:`numpy:numpy.ndarray` of shape (...)
        Categorical values of any standard type. Vectorized to work for any dimensional `X`.
    labels : :class:`numpy:numpy.ndarray` of shape (n,)
        Complete list of all possible labels. List is flattened if it is not already 1 dimensional.
    assume_sorted : bool
        If true, assume labels is already sorted and unique. This saves the computational cost of calling
        :func:`numpy:numpy.unique`.
    dtype : type
        Desired data of feature array. One-hot is most logically `bool`, but feature matrices are usually `float`.
    assume_valid : bool
        If true, assume all element of `X` are in the list `labels`. This saves the computational cost of verifying
        `X` are in `labels`. If true and a non-label `X` occurs this routine will silently give bogus result.

    Returns
    -------
    Y : :class:`numpy:numpy.ndarray` of shape (..., n)
        One-hot encoding of `X`. Extra dimension is appended at end for the one-hot vector. It has data type `dtype`.
    """
    X = np.asarray(X)
    labels = np.asarray(labels) if assume_sorted else np.unique(labels)
    check_array(labels, "labels", pre=True, ndim=1, min_size=1)

    idx = np.searchsorted(labels, X)
    # If x is not even in labels then this will fail. This is not ValueError
    # because the user explictly asked for this using argument assume_valid.
    assert assume_valid or np.all(np.asarray(labels[idx]) == X)

    # This is using some pro np indexing technique to vectorize across all
    # possible input dimensions for X in the same code.
    Y = np.zeros(X.shape + (len(labels),), dtype=dtype)
    Y[unravel_index(X.shape) + (idx.ravel(),)] = True
    return Y


def decode(Y, labels, assume_sorted=False):
    """Perform inverse of one-hot encoder `encode`.

    Parameters
    ----------
    Y : :class:`numpy:numpy.ndarray` of shape (..., n)
        One-hot encoding of categorical data `X`. Extra dimension is appended at end for the one-hot vector. Maximum
        element is taken if there is more than one non-zero entry in one-hot vector.
    labels : :class:`numpy:numpy.ndarray` of shape (n,)
        Complete list of all possible labels. List is flattened if it is not already 1-dimensional.
    assume_sorted : bool
        If true, assume labels is already sorted and unique. This saves the computational cost of calling
        :func:`numpy:numpy.unique`.

    Returns
    -------
    X : :class:`numpy:numpy.ndarray` of shape (...)
        Categorical values corresponding to one-hot encoded `Y`.
    """
    Y = np.asarray(Y)
    labels = np.asarray(labels) if assume_sorted else np.unique(labels)
    check_array(labels, "labels", pre=True, ndim=1, min_size=1)
    check_array(Y, "Y", pre=True, shape_endswith=(len(labels),))

    idx = np.argmax(Y, axis=-1)
    X = labels[idx]
    return X


def _error(msg, pre=False):  # pragma: validator
    """Helper routine for :func:`.check_array`.

    This could probably be made cleaner by using raise to create the assert.
    """
    if pre:
        raise ValueError(msg)
    else:
        assert False, msg


def check_array(
    X,
    name,
    pre=False,
    ndim=None,
    shape=None,
    shape_endswith=(),
    min_size=0,
    dtype=None,
    kind=None,
    allow_infinity=True,
    allow_nan=True,
    unsorted=True,
    whitelist=None,
):  # pragma: validator
    """Like :func:`sklearn:sklearn.utils.check_array` but better.

    Check specified property of input array `X`. If an argument is not specified it passes by default.

    Parameters
    ----------
    X : :class:`numpy:numpy.ndarray`
        `numpy` array we want to validate.
    name : str
        Human readable name of of variable to refer to it in error messages. Note this can include spaces unlike simply
        using the variable name.
    pre : bool
        If true, interpret this as check as validating pre-conditions to a function and will raise a `ValueError` if a
        check fails. If false, assumes we are checking post-conditions and will raise an assertion failure.
    ndim : int
        Expected value of ``X.ndim``.
    shape : tuple(int)
        Expected value of ``X.shape``.
    shape_endswith : tuple(int)
        Expected that ``X.shape`` ends with `shape_endswith`. This is useful in broadcasting where extra dimensions
        might be added on.
    min_size : int
        Minimum value for ``X.size``
    dtype : dtype
        Expected value of ``X.dtype``.
    kind : str
        Expected value of ``X.dtype.kind``. This is `'f'` for `float`, `'i'` for `int`, and so on.
    allow_infinity : bool
        If false, the check fails when `X` contains inf or ``-inf``.
    allow_nan : bool
        If false, the check fails when `X` contains a ``NaN``.
    unsorted : bool
        If false, the check fails when `X` is not in sorted order. This is designed to even work with string arrays.
    whitelist : :class:`numpy:numpy.ndarray`
        Array containing allowed values for `X`. If an element of `X` is not found in `whitelist`, the check fails.
    """
    if (ndim is not None) and X.ndim != ndim:
        _error("Expected %d dimensions for %s, got %d" % (ndim, name, X.ndim), pre)

    if (shape is not None) and X.shape != shape:
        _error("Expected shape %s for %s, got %s" % (str(shape), name, str(X.shape)), pre)

    if len(shape_endswith) > 0:
        if X.shape[-len(shape_endswith) :] != shape_endswith:
            if len(shape_endswith) == 1:
                _error("Expected shape (..., %d) for %s, got %s" % (shape_endswith[0], name, str(X.shape)), pre)
            else:
                _error("Expected shape (..., %s for %s, got %s" % (str(shape_endswith)[1:], name, str(X.shape)), pre)

    if (min_size > 0) and (X.size < min_size):
        _error("%s needs at least %d elements, it has %d" % (name, min_size, X.size), pre)

    if (dtype is not None) and X.dtype != np.dtype(dtype):
        _error("Expected dtype %s for %s, got %s" % (str(np.dtype(dtype)), name, str(X.dtype)), pre)

    if (kind is not None) and X.dtype.kind != kind:
        _error("Expected array with kind %s for %s, got %s" % (kind, name, str(X.dtype.kind)), pre)

    if (not allow_infinity) and np.any(np.abs(X) == np.inf):
        _error("Infinity is not allowed in %s" % name, pre)

    if (not allow_nan) and np.any(np.isnan(X)):
        _error("NaN is not allowed in %s" % name, pre)

    if whitelist is not None:
        ok = np.all([xx in whitelist for xx in np.nditer(X, ["zerosize_ok"])])
        if not ok:
            _error("Expected all elements of %s to be in %s" % (name, str(whitelist)), pre)

    # Only do this check in 1D
    if X.ndim == 1 and (not unsorted) and np.any(X[:-1] > X[1:]):
        _error("Expected sorted input for %s" % name, pre)


# ============================================================================
# Setup warping dictionaries
# ============================================================================


def identity(x):
    """Helper function that perform warping in linear space. Sort of a no-op.

    Parameters
    ----------
    x : scalar
        Input variable in linear space. Can be any numeric type and is vectorizable.

    Returns
    -------
    y : scalar
        Same as input `x`.
    """
    y = x
    return y


def bilog(x):
    """Bilog warping function. Extension of log to work with negative numbers.

    ``Bilog(x) ~= log(x)`` for large `x` or ``-log(abs(x))`` if `x` is negative. However, the bias term ensures good
    behavior near 0 and ``bilog(0) = 0``.

    Parameters
    ----------
    x : scalar
        Input variable in linear space. Can be any numeric type and is vectorizable.

    Returns
    -------
    y : float
        The bilog of `x`.
    """
    y = np.sign(x) * np.log(1.0 + np.abs(x))
    return y


def biexp(x):
    """Inverse of :func:`.bilog` function.

    Parameters
    ----------
    x : scalar
        Input variable in linear space. Can be any numeric type and is vectorizable.

    Returns
    -------
    y : float
        The biexp of `x`.
    """
    y = np.sign(x) * (np.exp(np.abs(x)) - 1.0)
    return y


WARP_DICT = {"linear": identity, "log": np.log, "logit": logit, "bilog": bilog}
UNWARP_DICT = {"linear": identity, "log": np.exp, "logit": logistic, "bilog": biexp}

# ============================================================================
# Setup spaces class hierarchy
# ============================================================================


class Space(object):
    """Base class for all types of variables.
    """

    def __init__(self, dtype, default_round, warp="linear", values=None, range_=None):
        """Generic constructor of `Space` class.

        Not intended to be called directly but instead by child classes. However, `Space` is not an abstract class and
        will not give an error when instantiated.
        """
        self.dtype = dtype
        assert warp in WARP_DICT, "invalid space %s, allowed spaces are: %s" % (str(warp), str(WARP_DICT.keys()))
        self.warp_f = WARP_DICT[warp]
        self.unwarp_f = UNWARP_DICT[warp]

        # Setup range and rounding if values is suplied
        assert (values is None) != (range_ is None)
        round_to_values = default_round
        if range_ is None:  # => value is not None
            # Debatable if unique should be done before or after cast. But I
            # think after is better, esp. when changing precisions.
            values = np.asarray(values, dtype=dtype)
            values = np.unique(values)  # values now 1D ndarray no matter what
            check_array(
                values,
                "unique values",
                pre=True,
                ndim=1,
                dtype=dtype,
                min_size=2,
                allow_infinity=False,
                allow_nan=False,
            )

            # Extrapolation might happen due to numerics in type conversions.
            # Bounds checking is still done in validate routines.
            round_to_values = interp1d(values, values, kind="nearest", fill_value="extrapolate")
            range_ = (values[0], values[-1])
        # Save values and rounding
        # Values is either None or was validated inside if statement
        self.values = values
        self.round_to_values = round_to_values

        # Note that if dtype=None that is the default for asarray.
        range_ = np.asarray(range_, dtype=dtype)
        check_array(range_, "range", pre=True, shape=(2,), dtype=dtype, unsorted=False)
        # Save range info, with input validation and post validation
        self.lower, self.upper = range_

        # Convert to warped bounds too with lots of post validation
        self.lower_warped, self.upper_warped = self.warp_f(range_[..., None]).astype(WARPED_DTYPE, copy=False)
        check_array(
            self.lower_warped,
            "warped lower bound %s(%.1f)" % (warp, self.lower),
            ndim=1,
            pre=True,
            dtype=WARPED_DTYPE,
            allow_infinity=False,
            allow_nan=False,
        )
        # Should never happen if warpers are strictly monotonic:
        assert np.all(self.lower_warped <= self.upper_warped)

        # Make sure a bit bigger to keep away from lower due to numerics
        self.upper_warped = np.maximum(self.upper_warped, np.nextafter(self.lower_warped, np.inf))
        check_array(
            self.upper_warped,
            "warped upper bound %s(%.1f)" % (warp, self.upper),
            pre=True,
            shape=self.lower_warped.shape,
            dtype=WARPED_DTYPE,
            allow_infinity=False,
            allow_nan=False,
        )
        # Should never happen if warpers are strictly monotonic:
        assert np.all(self.lower_warped < self.upper_warped)

    def validate(self, X, pre=False):
        """Routine to validate inputs to warp.

        This routine does not perform any checking on the dimensionality of `X` and is fully vectorized.
        """
        X = np.asarray(X, dtype=self.dtype)

        if self.values is None:
            X = clip_chk(X, self.lower, self.upper)
        else:
            check_array(X, "X", pre=pre, whitelist=self.values)

        return X

    def validate_warped(self, X, pre=False):
        """Routine to validate inputs to unwarp. This routine is vectorized, but `X` must have at least 1-dimension.
        """
        X = np.asarray(X, dtype=WARPED_DTYPE)
        check_array(X, "X", pre=pre, shape_endswith=(len(self.lower_warped),))

        X = clip_chk(X, self.lower_warped, self.upper_warped)
        return X

    def warp(self, X):
        """Warp inputs to a continuous space.

        Parameters
        ----------
        X : :class:`numpy:numpy.ndarray` of shape (...)
            Input variables to warp. This is vectorized to work in any dimension, but it must have the same type code
            as the class, which is in `self.type_code`.

        Returns
        -------
        X_w : :class:`numpy:numpy.ndarray` of shape (..., m)
            Warped version of input space. By convention there is an extra dimension on warped array.
            Currently, ``m=1`` for all warpers. `X_w` will have a `float` type.
        """
        X = self.validate(X, pre=True)

        X_w = self.warp_f(X)
        X_w = X_w[..., None]  # Convention is that warped has extra dim

        X_w = self.validate_warped(X_w)  # Ensures of WAPRED_DTYPE
        check_array(X_w, "X", ndim=X.ndim + 1, dtype=WARPED_DTYPE)
        return X_w

    def unwarp(self, X_w):
        """Inverse of `warp` function.

        Parameters
        ----------
        X_w : :class:`numpy:numpy.ndarray` of shape (..., m)
            Warped version of input space. This is vectorized to work in any dimension. But, by convention, there is an
            extra dimension on the warped array. Currently, the last dimension ``m=1`` for all warpers. `X_w` must be of
            a `float` type.

        Returns
        -------
        X : :class:`numpy:numpy.ndarray` of shape (...)
            Unwarped version of `X_w`. `X` will have the same type code as the class, which is in `self.type_code`.
        """
        X_w = self.validate_warped(X_w, pre=True)

        X = clip_chk(self.unwarp_f(X_w[..., 0]), self.lower, self.upper)
        X = self.round_to_values(X)

        X = self.validate(X)  # Ensures of dtype
        check_array(X, "X", ndim=X_w.ndim - 1, dtype=self.dtype)
        return X

    def get_bounds(self):
        """Get bounds of the warped space.

        Returns
        -------
        bounds : :class:`numpy:numpy.ndarray` of shape (D, 2)
            Bounds in the warped space. First column is the lower bound and the second column is the upper bound.
            Calling ``bounds.tolist()`` gives the bounds in the standard form expected by `scipy` optimizers:
            ``[(lower_1, upper_1), ..., (lower_n, upper_n)]``.
        """
        bounds = np.stack((self.lower_warped, self.upper_warped), axis=1)
        check_array(bounds, "bounds", shape=(len(self.lower_warped), 2), dtype=WARPED_DTYPE)
        return bounds

    def grid(self, max_interp=N_GRID_DEFAULT):
        """Return grid spanning the original (unwarped) space.

        Parameters
        ----------
        max_interp : int
            The number of points to use in grid space when a range and not values are used to define the space.
            Must be ``>= 0``.

        Returns
        -------
        values : list
            Grid spanning the original space. This is simply `self.values` if a grid has already been specified,
            otherwise it is just grid across the range.
        """
        values = self.values
        if values is None:
            vw = np.linspace(self.lower_warped, self.upper_warped, max_interp)
            # Some spaces like int make result in duplicates after unwarping
            # so we apply unique to avoid this. However this will usually be
            # wasted computation.
            values = np.unique(self.unwarp(vw[:, None]))
            check_array(values, "values", ndim=1, dtype=self.dtype)

        # Best to convert to list to make sure in native type
        values = values.tolist()
        return values


class Real(Space):
    """Space for transforming real variables to normalized space (after warping).
    """

    def __init__(self, warp="linear", values=None, range_=None):
        """Build Real space class.

        Parameters
        ----------
        warp : {'linear', 'log', 'logit', 'bilog'}
            Which warping type to apply to the space. The warping is applied in the original space. That is, in a space
            with ``warp='log'`` and ``range_=(2.0, 10.0)``, the value 2.0 warps to ``log(2)``, not ``-inf`` as in some
            other frameworks.
        values : None or list(float)
            Possible values for space to take. Values must be of `float` type.
        range_ : None or :class:`numpy:numpy.ndarray` of shape (2,)
            Array with (lower, upper) pair with limits of space. Note that one must specify `values` or `range_`, but
            not both. `range_` must be composed of `float`.
        """
        assert warp is not None, "warp/space not specified for real"
        Space.__init__(self, np.float_, identity, warp, values, range_)


class Integer(Space):
    """Space for transforming integer variables to continuous normalized space.
    """

    def __init__(self, warp="linear", values=None, range_=None):
        """Build Integer space class.

        Parameters
        ----------
        warp : {'linear', 'log', 'bilog'}
            Which warping type to apply to the space. The warping is applied in the original space. That is, in a space
            with ``warp='log'`` and ``range_=(2, 10)``, the value 2 warps to ``log(2)``, not ``-inf`` as in some other
            frameworks. There are no settings with integers that are compatible with the logit warp.
        values : None or list(float)
            Possible values for space to take. Values must be of `int` type.
        range_ : None or :class:`numpy:numpy.ndarray` of shape (2,)
            Array with (lower, upper) pair with limits of space. Note that one must specify `values` or `range_`, but
            not both. `range_` must be composed of `int`.
        """
        assert warp is not None, "warp/space not specified for int"
        Space.__init__(self, np.int_, np.round, warp, values, range_)


class Boolean(Space):
    """Space for transforming Boolean variables to continuous normalized space.
    """

    def __init__(self, warp=None, values=None, range_=None):
        """Build Boolean space class.

        Parameters
        ----------
        warp : None
            Must be omitted or None, provided for consitency with other types.
        values : None
            Must be omitted or None, provided for consitency with other types.
        range_ : None
            Must be omitted or None, provided for consitency with other types.
        """
        assert warp is None, "cannot warp bool"
        assert (values is None) and (range_ is None), "cannot pass in values or range for bool"
        self.dtype = np.bool_
        self.warp_f = identity
        self.unwarp_f = identity

        self.values = np.array([False, True], dtype=np.bool_)
        self.round_to_values = np.round

        self.lower, self.upper = self.dtype(False), self.dtype(True)
        self.lower_warped = np.array([0.0], dtype=WARPED_DTYPE)
        self.upper_warped = np.array([1.0], dtype=WARPED_DTYPE)


class Categorical(Space):
    """Space for transforming categorical variables to continuous normalized space.
    """

    def __init__(self, warp=None, values=None, range_=None):
        """Build Integer space class.

        Parameters
        ----------
        warp : None
            Must be omitted or None, provided for consitency with other types.
        values : list(str)
            Possible values for space to take. Values must be unicode strings. Requiring type unicode (``'U'``) rather
            than strings (``'S'``) corresponds to the native string type.
        range_ : None
            Must be omitted or None, provided for consitency with other types.
        """
        assert warp is None, "cannot warp cat"
        assert values is not None, "must pass in explicit values for cat"
        assert range_ is None, "cannot pass in range for cat"

        values = np.unique(values)  # values now 1D ndarray no matter what
        check_array(values, "values", pre=True, ndim=1, kind=CAT_KIND, min_size=2)
        self.values = values

        self.dtype = CAT_DTYPE
        # Debatable if decode should go in unwarp or round_to_values

        self.warp_f = self._encode
        self.unwarp_f = identity
        self.round_to_values = self._decode

        self.lower, self.upper = None, None  # Don't need them
        self.lower_warped = np.zeros(len(values), dtype=WARPED_DTYPE)
        self.upper_warped = np.ones(len(values), dtype=WARPED_DTYPE)

    def _encode(self, x):
        return encode(x, self.values, True, WARPED_DTYPE, True)

    def _decode(self, x):
        return decode(x, self.values, True)

    def warp(self, X):
        """Warp inputs to a continuous space.

        Parameters
        ----------
        X : :class:`numpy:numpy.ndarray` of shape (...)
            Input variables to warp. This is vectorized to work in any dimension, but it must have the same
            type code as the class, which is unicode (``'U'``) for the :class:`.Categorical` space.

        Returns
        -------
        X_w : :class:`numpy:numpy.ndarray` of shape (..., m)
            Warped version of input space. By convention there is an extra dimension on warped array. The warped space
            has a one-hot encoding and therefore `m` is the number of possible values in the space. `X_w` will have
            a `float` type.
        """
        X = self.validate(X, pre=True)

        X_w = self.warp_f(X)

        # Probably over kill to validate here too, but why not:
        X_w = self.validate_warped(X_w)
        check_array(X_w, "X", ndim=X.ndim + 1, dtype=WARPED_DTYPE)
        return X_w

    def unwarp(self, X_w):
        """Inverse of `warp` function.

        Parameters
        ----------
        X_w : :class:`numpy:numpy.ndarray` of shape (..., m)
            Warped version of input space. The warped space has a one-hot encoding and therefore `m` is the number of
            possible values in the space. `X_w` will have a `float` type. Non-zero/one values are allowed in `X_w`.
            The maximal element in the vector is taken as the encoded value.

        Returns
        -------
        X : :class:`numpy:numpy.ndarray` of shape (...)
            Unwarped version of `X_w`. `X` will have same type code as the :class:`.Categorical` class, which is
            unicode (``'U'``).
        """
        X_w = self.validate_warped(X_w, pre=True)

        X = self.round_to_values(self.unwarp_f(X_w))

        X = self.validate(X)
        check_array(X, "X", ndim=X_w.ndim - 1, kind=CAT_KIND)
        return X


# Treat ordinal identically to categorical for now
SPACE_DICT = {"real": Real, "int": Integer, "bool": Boolean, "cat": Categorical, "ordinal": Categorical}

# ============================================================================
# Setup code for joint spaces over multiple parameters with different configs
# ============================================================================


class JointSpace(object):
    """Combination of multiple :class:`.Space` objectives to transform multiple variables at the same time (jointly).
    """

    def __init__(self, meta):
        """Build Real space class.

        Parameters
        ----------
        meta : dict(str, dict)
            Configuration of variables in joint space. See API description.
        """
        assert len(meta) > 0  # Unclear what to do with empty space

        # Lock in an order if not ordered dict, sorted helps reproducibility
        self.param_list = sorted(meta.keys())

        # Might as well pre-validate a bit here
        for param, config in meta.items():
            assert config["type"] in SPACE_DICT, "invalid input type %s" % config["type"]

        spaces = {
            param: SPACE_DICT[config["type"]](
                config.get("space", None), config.get("values", None), config.get("range", None)
            )
            for param, config in meta.items()
        }
        self.spaces = spaces

        self.blocks = np.cumsum([len(spaces[param].get_bounds()) for param in self.param_list])

    def validate(self, X):
        """Raise `ValueError` if X does not match the format expected for a
        joint space."""
        for record in X:
            if self.param_list != sorted(record.keys()):
                raise ValueError("Expected joint space keys %s, but got %s", (self.param_list, sorted(record.keys())))
            for param in self.param_list:
                self.spaces[param].validate([record[param]], pre=True)
        # Return X back so we have option to cast it to list or whatever later
        return X

    def warp(self, X):
        """Warp inputs to a continuous space.

        Parameters
        ----------
        X : list(dict(str, object)) of shape (n,)
            List of `n` points in the joint space to warp. Each list element is a dictionary where each key corresponds
            to a variable in the joint space. Keys can be be missing in the records and the according warped variables
            will be ``nan``.

        Returns
        -------
        X_w : :class:`numpy:numpy.ndarray` of shape (n, m)
            Warped version of input space. Result is 2D `float` np array. `n` is the number of input points, length
            of `X`. `m` is the size of the joint warped space, which can be inferred by calling :func:`.get_bounds`.
        """
        # It would be nice to have cleaner way to deal with this corner case
        if len(X) == 0:
            return np.zeros((0, self.blocks[-1]), dtype=WARPED_DTYPE)

        X_w = [
            np.concatenate(
                [
                    self.spaces[param].warp(record[param])
                    if param in record
                    else np.full(len(self.spaces[param].get_bounds()), np.nan)
                    for param in self.param_list
                ]
            )
            for record in X
        ]
        X_w = np.stack(X_w, axis=0)
        check_array(X_w, "X", shape=(len(X), self.blocks[-1]), dtype=WARPED_DTYPE)
        return X_w

    def unwarp(self, X_w, fixed_vals={}):
        """Inverse of :func:`.warp`.

        Parameters
        ----------
        X_w : :class:`numpy:numpy.ndarray` of shape (n, m)
            Warped version of input space. Must be 2D `float` :class:`numpy:numpy.ndarray`. `n` is the number of
            separate points in the warped joint space. `m` is the size of the joint warped space, which can be inferred
            in advance by calling :func:`.get_bounds`.
        fixed_vals : dict
            Subset of variables we want to keep fixed in X. Unwarp checks that the unwarped version of `X_w` matches
            `fixed_vals` up to numerical error. Otherwise, an error is raised.

        Returns
        -------
        X : list(dict(str, object)) of shape (n,)
            List of `n` points in the joint space to warp. Each list element is a dictionary where each key corresponds
            to a variable in the joint space.
        """
        X_w = np.asarray(X_w)
        check_array(X_w, "X", ndim=2, shape_endswith=(self.blocks[-1],), dtype=WARPED_DTYPE)
        N = X_w.shape[0]

        # Use snap_to to make sure we get exact value (no-round off) for cases where we know expected answer
        X = {
            param: snap_to(self.spaces[param].unwarp(xx), fixed_vals.get(param, None))
            for param, xx in zip(self.param_list, np.hsplit(X_w, self.blocks[:-1]))
        }
        # Convert dict of arrays to list of dicts, this would not be needed if
        # we used pandas but we do not want to add it as a dep. np.asscalar and
        # .item() appear to be the same thing but asscalar seems more readable.
        X = [{param: X[param][ii].item() for param in self.param_list} for ii in range(N)]
        return X

    def get_bounds(self):
        """Get bounds of the warped joint space.

        Returns
        -------
        bounds : :class:`numpy:numpy.ndarray` of shape (m, 2)
            Bounds in the warped space. First column is the lower bound and the second column is the upper bound.
            ``bounds.tolist()`` gives the bounds in the standard form expected by scipy optimizers:
            ``[(lower_1, upper_1), ..., (lower_n, upper_n)]``.
        """
        bounds = np.concatenate([self.spaces[param].get_bounds() for param in self.param_list], axis=0)
        check_array(bounds, "bounds", shape_endswith=(2,), dtype=WARPED_DTYPE)
        return bounds

    def grid(self, max_interp=N_GRID_DEFAULT):
        """Return grid spanning the original (unwarped) space.

        Parameters
        ----------
        max_interp : int
            The number of points to use in grid space when a range and not values are used to define the space.
            Must be ``>= 0``.

        Returns
        -------
        axes : dict(str, list)
            Grids spanning the original spaces of each variable. For each variable, this is simply ``self.values``
            if a grid has already been specified, otherwise it is just grid across the range.
        """
        axes = {var_name: space.grid(max_interp=max_interp) for var_name, space in self.spaces.items()}
        return axes
