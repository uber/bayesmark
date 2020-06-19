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
"""Routines to compute and compare the "signatures" of objective functions. These are useful to make sure two different
studies were actually optimizing the same objective function (even if they say the same test case in the meta-data).
"""
import warnings

import numpy as np
import pandas as pd

import bayesmark.random_search as rs

# How many points to probe the function to get the signature
N_SUGGESTIONS = 5


def get_func_signature(f, api_config):
    """Get the function signature for an objective function in an experiment.

    Parameters
    ----------
    f : typing.Callable
        The objective function we want to compute the signature of. This function must take inputs in the form of
        ``dict(str, object)`` with one dictionary key per variable, and provide `float` as the output.
    api_config : dict(str, dict)
        Configuration of the optimization variables. See API description.

    Returns
    -------
    signature_x : list(dict(str, object)) of shape (n_suggest,)
        The input locations probed on signature call.
    signature_y : list(float) of shape (n_suggest,)
        The objective function values at the inputs points. This is the real signature.
    """
    # Make sure get same sequence on every call to be a signature
    random = np.random.RandomState(0)

    signature_x = rs.suggest_dict([], [], api_config, n_suggestions=N_SUGGESTIONS, random=random)

    # For now, we only take the first output as the signature. We can generalize this later.
    signature_y = [f(xx)[0] for xx in signature_x]
    assert np.all(np.isfinite(signature_y)), "non-finite values found in signature for function"
    return signature_x, signature_y


def analyze_signatures(signatures):
    """Analyze function signatures from the experiment.

    Parameters
    ----------
    signatures : dict(str, list(list(float)))
        The signatures should all be the same length, so it should be 2D array
        like.

    Returns
    -------
    sig_errs : :class:`pandas:pandas.DataFrame`
        rows are test cases, columns are test points.
    signatures_median : dict(str, list(float))
        Median signature across all repetition per test case.
    """
    sig_errs = {}
    signatures_median = {}
    for test_case, signature_y in signatures.items():
        assert len(signature_y) > 0, "signature with no cases found"
        assert np.all(np.isfinite(signature_y)), "non-finite values found in signature for function"

        minval = np.min(signature_y, axis=0)
        maxval = np.max(signature_y, axis=0)

        if not np.allclose(minval, maxval):
            # Arguably, the util should not raise the warning, and these should
            # be raised on the outside, but let's do this for simplicity.
            warnings.warn(
                "Signature diverged on %s betwen %s and %s" % (test_case, str(minval), str(maxval)), RuntimeWarning
            )
        sig_errs[test_case] = maxval - minval
        # ensure serializable using tolist
        signatures_median[test_case] = np.median(signature_y, axis=0).tolist()

    # Convert to pandas so easy to append margins with max, better for disp.
    # If we let the user convert to pandas then we don't need dep on pandas.
    sig_errs = pd.DataFrame(sig_errs).T
    sig_errs.loc["max", :] = sig_errs.max(axis=0)
    sig_errs.loc[:, "max"] = sig_errs.max(axis=1)

    return sig_errs, signatures_median


def analyze_signature_pair(signatures, signatures_ref):
    """Analyze a pair of signatures (often from two sets of experiments) and return the error between them.

    Parameters
    ----------
    signatures : dict(str, list(float))
        Signatures from set of experiments. The signatures must all be the same length, so it should be 2D array like.
    signatures_ref : dict(str, list(float))
        The signatures from a reference set of experiments. The keys in `signatures` must be a subset of the signatures
        in `signatures_ref`.

    Returns
    -------
    sig_errs : :class:`pandas:pandas.DataFrame`
        rows are test cases, columns are test points.
    signatures_median : dict(str, list(float))
        Median signature across all repetition per test case.
    """
    signatures_pair = {kk: [signatures[kk], signatures_ref[kk]] for kk in signatures}
    sig_errs, signatures_pair = analyze_signatures(signatures_pair)
    return sig_errs, signatures_pair
