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
"""A serialization abstraction layer (SAL) to save and load experimental results. All IO of experimental results should
go through this module. This makes changing the backend (between different databases) transparent to the benchmark code.
"""
import json
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from tempfile import mkdtemp

import xarray as xr
from pathvalidate.argparse import validate_filename, validate_filepath

from bayesmark.path_util import join_safe_r, join_safe_w
from bayesmark.util import chomp, str_join_safe

NEWLINE = "\n"  # Just to be explicit, in case this ever gets run on Windows
PREFIX_FMT = "bo_%Y%m%d_%H%M%S_"  # The format we use for generating a new database name if none is specified

_XR_EXT = ".json"  # Extension we use for dumping xr.Dataset variables
_LOG_EXT = ".log"  # Extension to reccomend for logging files
_DERIVED_DIR = "derived"  # The folder for dervied variables (datasets)
_LOGGING_DIR = "log"  # The folder to reccomend for logging
_SETUP_STR = """
User must ensure
%s
exists, and setup folder using
mkdir %s
User must ensure equal reps of each optimizer for unbiased results."""


class Serializer(ABC):
    """Abstract base class for the serialization abstraction layer.
    """

    @staticmethod
    @abstractmethod
    def init_db(db_root, keys, db=None, exist_ok=True):
        """Initialize a "database" for storing data at the specified location.

        Parameters
        ----------
        db_root : str
            Absolute path to the database.
        keys : list(str)
            The variable names (or keys) we will store in the database for non-derived data.
        db : str
            The name of the database. If ``None``, a non-conflicting name will be generated.
        exist_ok : bool
            If true, do not raise an error if this database already exists.

        Returns
        -------
        db : str
            The name of the database.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_keys(db_root, db):
        """List the non-derived keys available in the database.

        Parameters
        ----------
        db_root : str
            Absolute path to the database.
        db : str
            The name of the database.

        Returns
        -------
        keys : list(str)
            The variable names (or keys) in the database for non-derived data.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_derived_keys(db_root, db):
        """List the derived keys currently available in the database.

        Parameters
        ----------
        db_root : str
            Absolute path to the database.
        db : str
            The name of the database.

        Returns
        -------
        keys : list(str)
            The variable names (or keys) in the database for derived data.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_uuids(db_root, db, key):
        """List the UUIDs for the versions of a variable (non-derived key) available in the database.

        Parameters
        ----------
        db_root : str
            Absolute path to the database.
        db : str
            The name of the database.
        keys : str
            The variable name in the database for non-derived data.

        Returns
        -------
        uuids : list(uuid.UUID)
            The UUIDs for the versions of this key.
        """
        pass

    @staticmethod
    @abstractmethod
    def save(data, meta, db_root, db, key, uuid_):
        """Abstract method for saving experimental data, details require the type of `data`.
        """
        pass

    @staticmethod
    @abstractmethod
    def load(db_root, db, key, uuid_):
        """Abstract method for loading experimental data, details require the type of `data`.
        """
        pass

    @staticmethod
    @abstractmethod
    def save_derived(data, meta, db_root, db, key):
        """Abstract method for saving derived data, details require the type of `data`.
        """
        pass

    @staticmethod
    @abstractmethod
    def load_derived(db_root, db, key):
        """Abstract method for loading derived data, details require the type of `data`.
        """
        pass


class XRSerializer(Serializer):
    """Serialization layer when saving and loading `xarray` datasets (currently) as `json`.
    """

    def init_db(db_root, keys, db=None, exist_ok=True):  # pragma: io
        XRSerializer._validate(db_root, keys, db)

        if db is None:
            folder_prefix = datetime.utcnow().strftime(PREFIX_FMT)
            exp_subdir = mkdtemp(prefix=folder_prefix, dir=db_root)
            db = os.path.basename(exp_subdir)
            assert db.startswith(folder_prefix)
            assert os.path.join(db_root, db) == exp_subdir
        else:
            exp_subdir = os.path.join(db_root, db)
            os.makedirs(exp_subdir, exist_ok=exist_ok)

        subdirs = [_DERIVED_DIR, _LOGGING_DIR] + list(keys)
        for subd in subdirs:
            os.makedirs(os.path.join(exp_subdir, subd), exist_ok=exist_ok)

        return db

    def init_db_manual(db_root, keys, db):
        """Instruction for how one would manually initialize the "database" on another system.

        Parameters
        ----------
        db_root : str
            Absolute path to the database.
        keys : list(str)
            The variable names (or keys) we will store in the database for non-derived data.
        db : str
            The name of the database.

        Returns
        -------
        manual_setup_info : str
            The setup instructions.
        """
        XRSerializer._validate(db_root, keys, db)
        assert db is not None, "Must specify db name to setup manually."

        exp_subdir = os.path.join(db_root, db)
        subdirs = [_DERIVED_DIR, _LOGGING_DIR] + list(keys)
        manual_setup_info = _SETUP_STR % (exp_subdir, str_join_safe(" ", subdirs))
        return manual_setup_info

    def get_keys(db_root, db):  # pragma: io
        XRSerializer._validate(db_root, keys=(), db=db)

        keys = sorted(os.listdir(os.path.join(db_root, db)))
        keys.remove(_DERIVED_DIR)
        keys.remove(_LOGGING_DIR)
        return keys

    def get_derived_keys(db_root, db):  # pragma: io
        XRSerializer._validate(db_root, keys=(), db=db)

        fnames = sorted(os.listdir(os.path.join(db_root, db, _DERIVED_DIR)))
        keys = [XRSerializer._fname_to_key(ff) for ff in fnames]
        return keys

    def get_uuids(db_root, db, key):  # pragma: io
        XRSerializer._validate(db_root, keys=[key], db=db)

        fnames = sorted(os.listdir(os.path.join(db_root, db, key)))
        uuids = [XRSerializer._fname_to_uuid(ff) for ff in fnames]
        return uuids

    def save(data, meta, db_root, db, key, uuid_):  # pragma: io
        """Save a dataset under a key name in the database.

        Parameters
        ----------
        data : :class:`xarray:xarray.Dataset`
            An :class:`xarray:xarray.Dataset` variable we would like to store as non-derived data from an experiment.
        meta : json-serializable
            Associated meta-data with the experiment. This can be anything json serializable.
        db_root : str
            Absolute path to the database.
        db : str
            The name of the database.
        key : str
            The variable name in the database for the data.
        uuid_ : uuid.UUID
            The UUID to represent the version of this variable we are storing.
        """
        XRSerializer._validate(db_root, keys=[key], db=db)

        fname = XRSerializer._uuid_to_fname(uuid_)
        path = (db_root, db, key, fname)
        with open(join_safe_w(*path), "w") as f:
            _dump_xr(f, ds=data, meta=meta)

    def load(db_root, db, key, uuid_):  # pragma: io
        """Load a dataset under a key name in the database. This is the inverse of :func:`.save`.

        Parameters
        ----------
        db_root : str
            Absolute path to the database.
        db : str
            The name of the database.
        key : str
            The variable name in the database for the data.
        uuid_ : uuid.UUID
            The UUID to represent the version of this variable we want to load.

        Returns
        -------
        data : :class:`xarray:xarray.Dataset`
            An :class:`xarray:xarray.Dataset` variable for the non-derived data from an experiment.
        meta : json-serializable
            Associated meta-data with the experiment. This can be anything json serializable.
        """
        XRSerializer._validate(db_root, keys=[key], db=db)

        fname = XRSerializer._uuid_to_fname(uuid_)
        path = (db_root, db, key, fname)
        with open(join_safe_r(*path), "r") as f:
            ds, meta = _load_xr(f)
        return ds, meta

    def save_derived(data, meta, db_root, db, key):  # pragma: io
        """Save a dataset under a key name in the database as derived data.

        Parameters
        ----------
        data : :class:`xarray:xarray.Dataset`
            An :class:`xarray:xarray.Dataset` variable we would like to store as derived data from experiments.
        meta : json-serializable
            Associated meta-data with the experiments. This can be anything json serializable.
        db_root : str
            Absolute path to the database.
        db : str
            The name of the database.
        key : str
            The variable name in the database for the data.
        """
        XRSerializer._validate(db_root, keys=[key], db=db)

        fname = XRSerializer._key_to_fname(key)
        path = (db_root, db, _DERIVED_DIR, fname)
        with open(join_safe_w(*path), "w") as f:
            _dump_xr(f, ds=data, meta=meta)

    def load_derived(db_root, db, key):  # pragma: io
        """Load a dataset under a key name in the database as derived data. This is the inverse of :func:`.save_derived`.

        Parameters
        ----------
        db_root : str
            Absolute path to the database.
        db : str
            The name of the database.
        key : str
            The variable name in the database for the data.

        Returns
        -------
        data : :class:`xarray:xarray.Dataset`
            An :class:`xarray:xarray.Dataset` variable for the derived data from experiments.
        meta : json-serializable
            Associated meta-data with the experiments. This can be anything json serializable.
        """
        XRSerializer._validate(db_root, keys=[key], db=db)

        fname = XRSerializer._key_to_fname(key)
        path = (db_root, db, _DERIVED_DIR, fname)
        with open(join_safe_r(*path), "r") as f:
            data, meta = _load_xr(f)
        return data, meta

    def logging_path(db_root, db, uuid_):  # pragma: io
        """Get an absolute path for logging from an experiment given its UUID.

        Parameters
        ----------
        db_root : str
            Absolute path to the database.
        db : str
            The name of the database.
        uuid_ : uuid.UUID
            The UUID to represent this experiment.

        Returns
        -------
        logfile : str
            Absolute path suitable for logging in this experiment.
        """
        XRSerializer._validate(db_root, keys=(), db=db)
        assert isinstance(uuid_, uuid.UUID)

        fname = uuid_.hex + _LOG_EXT
        logfile = join_safe_w(db_root, db, _LOGGING_DIR, fname)
        return logfile

    def _fname_to_uuid(fname):
        uuid_ = uuid.UUID(chomp(fname, _XR_EXT))
        return uuid_

    def _uuid_to_fname(uuid_):
        assert isinstance(uuid_, uuid.UUID)  # This can be eliminated once we use type hints

        fname = uuid_.hex + _XR_EXT
        return fname

    def _key_to_fname(key):
        fname = key + _XR_EXT
        return fname

    def _fname_to_key(fname):
        key = chomp(fname, _XR_EXT)
        return key

    def _validate(db_root, keys=(), db=None):
        validate_filepath(db_root, platform="auto")
        assert os.path.isabs(db_root), "db_root must be absolute path"

        if db is not None:
            validate_filename(db, platform="universal")

        for kk in keys:
            validate_filename(kk, platform="universal")


def _dump_xr(f, ds, meta):  # pragma: io
    """Helper routine to `XRSerializer.save` and `XRSerializer.save_derived`.
    """
    assert isinstance(ds, xr.Dataset)  # Requiring Dataset and not DataArray for now

    meta_json = json.dumps(meta)  # meta can be anything that json can handle
    # JSON dumps seems pretty good about escaping, but check to be sure
    assert NEWLINE not in meta_json

    # Built in json dumper doesn't allow us to only line break on top-level, so we manually do this for now
    f.write('{"meta": %s,' % meta_json)
    f.write(NEWLINE)
    f.write('"data": ')

    json.dump(ds.to_dict(), f)
    f.write("}")
    f.write(NEWLINE)


def _load_xr(f):  # pragma: io
    """Helper routine to `XRSerializer.load` and`XRSerializer.load_derived`.
    """
    all_json = json.load(f)
    meta = all_json.pop("meta")
    ds = xr.Dataset.from_dict(all_json.pop("data"))
    return ds, meta
