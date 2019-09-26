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
import os
from string import ascii_letters, digits

from hypothesis import given
from hypothesis.strategies import lists, text, uuids
from pathvalidate.argparse import validate_filename, validate_filepath

from bayesmark.serialize import XRSerializer


def filepaths():
    def valid(ss):
        try:
            validate_filepath(ss)
        except Exception:
            return False
        return True

    alphabet = ascii_letters + digits + "_.-~" + os.sep
    S = text(alphabet=alphabet, min_size=1).map(lambda ss: os.sep + ss).filter(valid)
    return S


def filenames(suffix=""):
    def valid(ss):
        try:
            validate_filename(ss)
        except Exception:
            return False
        return True

    alphabet = ascii_letters + digits + "_.-~"
    S = text(alphabet=alphabet, min_size=1).map(lambda ss: ss + suffix).filter(valid)
    return S


@given(filepaths(), lists(filenames()), filenames())
def test_init_db_manual(db_root, keys, db):
    XRSerializer.init_db_manual(db_root, keys, db)


@given(uuids())
def test_uuid_to_fname(uu):
    ff = XRSerializer._uuid_to_fname(uu)
    uu_ = XRSerializer._fname_to_uuid(ff)
    assert uu == uu_

    ff_ = XRSerializer._uuid_to_fname(uu_)
    assert ff == ff_


@given(filenames())
def test_key_to_fname(key):
    ff = XRSerializer._key_to_fname(key)
    kk = XRSerializer._fname_to_key(ff)
    assert key == kk


@given(filepaths(), lists(filenames()), filenames())
def test_validate(db_root, keys, db):
    XRSerializer._validate(db_root, keys, db)
