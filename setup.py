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


def read_requirements(name):
    with open("requirements/" + name + ".in") as f:
        requirements = f.read().strip()
    requirements = requirements.replace("==", ">=").splitlines()  # Loosen strict pins
    return [pp for pp in requirements if pp[0].isalnum()]


# Derive install requires from base.in first order requirements
requirements = read_requirements("base")
opt_requirements = read_requirements("optimizers")
ipynb_requirements = read_requirements("ipynb")

with open("README.rst") as f:
    long_description = f.read()
# Probably more efficient way to do this with regex but good enough
for remove_word in REMOVE_FROM_RST:
    long_description = long_description.replace(remove_word, "")

setup(
    name="bayesmark",
    version="0.0.7",
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
