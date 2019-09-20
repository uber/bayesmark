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

# See if opentuner will work in env
dpkg -l | grep libsqlite

# Simulate deployment with wheel
./build_wheel.sh
mv -v dist/bo_benchmark-* dist/bo_benchmark.tar.gz

# Install and run local optimizers
mkdir install_test
cd install_test

virtualenv bobm_ipynb --python=python3.6
source ./bobm_ipynb/bin/activate
python --version
pip freeze | sort

pip install ../dist/bo_benchmark.tar.gz[optimizers,notebooks]

# Be able to check if using version out of tar ball
which bob-launch
which bob-exp
which bob-agg
which bob-anal

cp -r ../notebooks .
DB_ROOT=./notebooks
DBID=bo_example_folder

bob-launch -n 15 -r 3 -dir $DB_ROOT -b $DBID -o RandomSearch PySOT -c SVM DT -d boston breast iris -v
bob-agg -dir $DB_ROOT -b $DBID
bob-anal -dir $DB_ROOT -b $DBID -v

# Try ipynb export
python -m ipykernel install --name=bobm_ipynb --user
jupyter nbconvert --to html --execute notebooks/plot_mean_score.ipynb

# Try dry run
bob-launch -n 15 -r 3 -dir $DB_ROOT -b $DBID -o RandomSearch PySOT -c SVM DT -nj 50 -v

# Try again but use the custom optimizers
mv $DB_ROOT/$DBID old
cp -r ../example_opt_root .
bob-launch -n 15 -r 3 -dir $DB_ROOT -b $DBID -o RandomSearch PySOT-New -c SVM DT --opt-root ./example_opt_root -d boston breast iris -v
bob-agg -dir $DB_ROOT -b $DBID
bob-anal -dir $DB_ROOT -b $DBID -v

# Export again
jupyter nbconvert --to html --execute notebooks/plot_mean_score.ipynb

# Try dry run
bob-launch -n 15 -r 3 -dir $DB_ROOT -b $DBID -o RandomSearch PySOT-New -c SVM DT --opt-root ./example_opt_root -nj 50 -v

# wrap up
deactivate
cd ..
