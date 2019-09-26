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
"""Utilities handy for manipulating paths that have extra checks not included in `os.path`.
"""
import os.path
import warnings


def abspath(path, verify=True):  # pragma: io
    """Combo of :func:`os.path.abspath` and :func:`os.path.expanduser` that will also check existence of directory.

    Parameters
    ----------
    path : str
        Relative path string that can also contain home directories, e.g., ``"~/git/"``.
    verify : bool
        If true, verifies that the directory exists. Raises an assertion failure if it does not exist.

    Returns
    -------
    path : str
        Absolute version of input path.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if verify:
        assert os.path.isdir(path), "directory does not exist: %s" % path
    return path


def absopen(path, mode):  # pragma: io
    """Safe version of the built in :func:`open` that only opens absolute paths.

    Parameters
    ----------
    path : str
        Absolute path. An assertion failure is raised if it is not absolute.
    mode : str
        Open mode, any mode understood by the built in :func:`open`, e.g., ``"r"`` or ``"w"``.

    Returns
    -------
    f : file handle
        File handle open to use.
    """
    assert os.path.isabs(path), "Only allowing opening of absolute paths for safety."
    f = open(path, mode)
    return f


def _join_safe(*args):  # pragma: io
    """Helper routine with commonalities between `join_safe_r` and `join_safe_w`.
    """
    assert len(args) >= 2
    path, fname = args[:-1], args[-1]

    path = os.path.join(*path)  # Put together the dir
    path = abspath(path, verify=True)  # Make sure dir is abs, and exists

    assert os.path.basename(fname) == fname, "Expected basename got %s" % fname
    fname = os.path.join(path, fname)  # Put on the filename, must be abs
    # Could check abs again if really wanted to be safe
    return fname


def join_safe_r(*args):  # pragma: io
    """Safe version of :func:`os.path.join` that checks resulting path is absolute and the file exists for reading.

    Parameters
    ----------
    *args : str
        varargs for parts of path to combine. The last argument must be a file name.

    Returns
    -------
    fname : str
        Absolute path to filename.
    """
    fname = _join_safe(*args)
    assert os.path.isfile(fname)  # Check it exists
    return fname


def join_safe_w(*args):  # pragma: io
    """Safe version of :func:`os.path.join` that checks resulting path is absolute.

    Because this routine is for writing, if the file already exists, a warning is raised.

    Parameters
    ----------
    *args : str
        varargs for parts of path to combine. The last argument must be a file name.

    Returns
    -------
    fname : str
        Absolute path to filename.
    """
    fname = _join_safe(*args)
    # Give a warning if it exists
    if os.path.isfile(fname):
        warnings.warn("file already exists: %s" % fname, RuntimeWarning)
    return fname
