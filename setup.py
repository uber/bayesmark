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
from setuptools import find_packages, setup

CMD_NAME = "bayesmark"

# Strings to remove from README to make it PyPI friendly. See:
# https://packaging.python.org/guides/making-a-pypi-friendly-readme/#validating-restructuredtext-markup
REMOVE_FROM_RST = (":func:", ":ref:")

# Derive install requires from base.in first order requirements
with open("requirements/base.in") as f:
    requirements = f.read().strip()
requirements = requirements.replace("==", ">=").split()  # Convert to non-pinned for setup.py

with open("requirements/optimizers.in") as f:
    opt_requirements = f.read().strip()
opt_requirements = opt_requirements.replace("==", ">=").splitlines()  # Convert to non-pinned for setup.py
opt_requirements = [pp for pp in opt_requirements if pp[0].isalnum()]

with open("requirements/ipynb.in") as f:
    ipynb_requirements = f.read().strip()
ipynb_requirements = ipynb_requirements.replace("==", ">=").splitlines()  # Convert to non-pinned for setup.py
ipynb_requirements = [pp for pp in ipynb_requirements if pp[0].isalnum()]

with open("README.rst") as f:
    long_description = f.read()
# Probably more efficient way to do this with regex but good enough
for remove_word in REMOVE_FROM_RST:
    long_description = long_description.replace(remove_word, "")

setup(
    name="bayesmark",
    version="0.0.4rc5",
    packages=find_packages(),
    url="https://github.com/uber/bayesmark/",
    author="Ryan Turner",
    author_email=("rdturnermtl@github.com"),
    license="Apache v2",
    description="Bayesian optimization benchmark system",
    install_requires=requirements,
    extras_require={"optimizers": opt_requirements, "notebooks": ipynb_requirements},
    long_description=long_description,
    long_description_content_type="text/x-rst",
    platforms=["any"],
    entry_points={
        "console_scripts": [
            CMD_NAME + "-init = bayesmark.experiment_db_init:main",
            CMD_NAME + "-launch = bayesmark.experiment_launcher:main",
            CMD_NAME + "-agg = bayesmark.experiment_aggregate:main",
            CMD_NAME + "-baseline = bayesmark.experiment_baseline:main",
            CMD_NAME + "-anal = bayesmark.experiment_analysis:main",
            CMD_NAME + "-exp = bayesmark.experiment:main",
        ]
    },
)
