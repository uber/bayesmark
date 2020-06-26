#!/bin/bash

set -ex
set -o pipefail

export PIP_REQUIRE_VIRTUALENV=false

# Handy to know what we are working with
git --version
python --version
pip freeze | sort

# Cleanup workspace, src for any old -e installs
git clean -x -f -d
rm -rf src/

# See if opentuner will work in env (but this command does not work on Mac)
# dpkg -l | grep libsqlite

# Simulate deployment with wheel
./build_wheel.sh
mv -v dist/bayesmark-* dist/bayesmark.tar.gz

# Install and run local optimizers
mkdir install_test
cp -r ./notebooks install_test
cp -r ./example_opt_root install_test

cd install_test
virtualenv bobm_ipynb --python=python3
source ./bobm_ipynb/bin/activate
python --version
pip freeze | sort

# Remove this if we want to make sure everything is compatible with latest
# pip install -r ../requirements/optimizers.txt

pip install ../dist/bayesmark.tar.gz[optimizers,notebooks]
../integration_test.sh

# wrap up
deactivate
cd ..

echo "success with setup wrapper too"
