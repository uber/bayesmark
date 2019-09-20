#!/bin/bash

set -ex
set -o pipefail

# Set conda paths
export CONDA_PATH=./tmp/conda
export CONDA_ENVS=env

# Sometime pip PIP_REQUIRE_VIRTUALENV has issues with conda
export PIP_REQUIRE_VIRTUALENV=false

PY_VERSIONS=( "3.6" "3.7" )

# Handy to know what we are working with
git --version

# Cleanup workspace, src for any old -e installs
git clean -x -f -d
rm -rf src/

# Install miniconda
if command -v conda 2>/dev/null; then
    echo "Conda already installed"
else
    # We need to use miniconda since we can't figure out ho to install py3.6 in
    # this env image. We could also use Miniconda3-latest-Linux-x86_64.sh but
    # pinning version to make reprodicible.
    echo "Installing miniconda"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # In future let's also try, for reprodicibility:
        # curl -L -o miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.12-MacOSX-x86_64.sh;
        curl -L -o miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh;
    else
        # In future let's also try, for reprodicibility:
        # curl -L -o miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh;
        curl -L -o miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh;
    fi
    chmod +x ./miniconda.sh
    ./miniconda.sh -b -p $CONDA_PATH
    rm ./miniconda.sh
fi
export PATH=$CONDA_PATH/bin:$PATH

# Setup env just for installing pre-commit to run hooks on all files
rm -rf "$CONDA_ENVS"
ENV_PATH="${CONDA_ENVS}/bobm_commit_hooks"
conda create -y -q -p $ENV_PATH python=3.6
echo $ENV_PATH
source activate $ENV_PATH
python --version
pip freeze | sort
# not listing 2nd order deps here, but probably ok
pip install -r requirements/tools.txt
# Now run hooks on all files, don't need to install hooks since run directly
pre-commit run --all-files
# Now can leave env with  pre-commit
conda deactivate
# Also check no changes to files by hooks
test -z "$(git diff)"
# clean up for good measure, but need to keep miniconda tmp folder
git clean -x -f -d --exclude=tmp

# Tool to get compare only the package names in pip file
# On mac, sed -r needs to be seed -E
nameonly () { grep -i '^[a-z0-9]' | sed -E "s/([^=]*)==.*/\1/g" | tr _ - | sort -f; }
nameveronly () { grep -i '^[a-z0-9]' | awk '{print $1}' | tr _ - | sort -f; }
pipcheck () { cat $@ | grep -i '^[a-z0-9]' | awk '{print $1}' | sed -f requirements/pipreqs_edits.sed | sort -f | uniq >ask.log && pip freeze | sed -f requirements/pipreqs_edits.sed | sort -f >got.log && diff -i ask.log got.log; }

# Now test the deps
ENV_PATH="${CONDA_ENVS}/deps_test"
conda create -y -q -p $ENV_PATH python=3.6
echo $ENV_PATH
source activate $ENV_PATH
python --version
pip freeze | sort

# Install all requirements, make sure they are mutually compatible
pip install -r requirements/base.txt
pipcheck requirements/base.txt

# Install package
python setup.py install
pipcheck requirements/base.txt requirements/self.txt

pip install -r requirements/optimizers.txt
pipcheck requirements/base.txt requirements/self.txt requirements/optimizers.txt

pip install -r requirements/test.txt
pipcheck requirements/base.txt requirements/self.txt requirements/optimizers.txt requirements/test.txt

pip install -r requirements/ipynb.txt
pipcheck requirements/base.txt requirements/self.txt requirements/test.txt requirements/optimizers.txt requirements/ipynb.txt
pip install -r requirements/docs.txt
pipcheck requirements/base.txt requirements/self.txt requirements/test.txt requirements/optimizers.txt requirements/ipynb.txt requirements/docs.txt

pip install -r requirements/tools.txt

# Make sure .in file corresponds to what is imported
nameonly <requirements/base.in >ask.log
pipreqs bo_benchmark/  --ignore bo_benchmark/builtin_opt/ --savepath requirement_chk.in
sed -f requirements/pipreqs_edits.sed requirement_chk.in | nameonly >got.log
diff ask.log got.log

nameonly <requirements/test.in >ask.log
pipreqs test/ --savepath requirement_chk.in
sed -f requirements/pipreqs_edits.sed requirement_chk.in | nameonly >got.log
diff ask.log got.log

nameonly <requirements/optimizers.in >ask.log
pipreqs bo_benchmark/builtin_opt/ --savepath requirement_chk.in
sed -f requirements/pipreqs_edits.sed requirement_chk.in | nameonly >got.log
diff ask.log got.log

nameonly <requirements/docs.in >ask.log
pipreqs docs/ --savepath requirement_chk.in
sed -f requirements/pipreqs_edits.sed requirement_chk.in | nameonly >got.log
diff ask.log got.log

nameonly <requirements/ipynb.in >ask.log
jupyter nbconvert --to script notebooks/*.ipynb
pipreqs notebooks/ --savepath requirement_chk.in
sed -f requirements/pipreqs_edits.sed requirement_chk.in | nameonly >got.log
diff ask.log got.log

# Make sure txt file corresponds to pip compile
# First copy the originals
for f in requirements/*.txt; do cp -- "$f" "${f%.txt}.chk"; done
# Now re-compile
# no-upgrade means that by default it keeps the 2nd order dependency versions already in the requirements txt file
# (otherwise it brings it to the very latest available version which often causes issues).
pip-compile-multi -o txt --no-upgrade

nameveronly <requirements/base.chk >ask.log
sed -f requirements/pipreqs_edits.sed requirements/base.txt | nameveronly >got.log
diff ask.log got.log

nameveronly <requirements/test.chk >ask.log
sed -f requirements/pipreqs_edits.sed requirements/test.txt | nameveronly >got.log
diff ask.log got.log

nameveronly <requirements/optimizers.chk | sed -f requirements/pipreqs_edits.sed >ask.log
sed -f requirements/pipreqs_edits.sed requirements/optimizers.txt | nameveronly >got.log
diff ask.log got.log

nameveronly <requirements/ipynb.chk | sed -f requirements/pipreqs_edits.sed >ask.log
sed -f requirements/pipreqs_edits.sed requirements/ipynb.txt | nameveronly >got.log
diff ask.log got.log

nameveronly <requirements/docs.chk | sed -f requirements/pipreqs_edits.sed >ask.log
sed -f requirements/pipreqs_edits.sed requirements/docs.txt | nameveronly >got.log
diff ask.log got.log

nameveronly <requirements/tools.chk | sed -f requirements/pipreqs_edits.sed >ask.log
sed -f requirements/pipreqs_edits.sed requirements/tools.txt | nameveronly >got.log
diff ask.log got.log

# Deactivate virtual environment
conda deactivate

# Set up environments for all Python versions and loop over them
rm -rf "$CONDA_ENVS"
for i in "${PY_VERSIONS[@]}"
do
    # Now test the deps
    ENV_PATH="${CONDA_ENVS}/unit_test"
    conda create -y -q -p $ENV_PATH python=$i
    echo $ENV_PATH
    source activate $ENV_PATH
    python --version
    pip freeze | sort

    # Install all requirements
    pip install -r requirements/test.txt

    # Install package
    python setup.py install

    # Run tests
    pytest test/ -s -v --hypothesis-seed=0 --disable-pytest-warnings --cov=bo_benchmark --cov-report html

    conda deactivate
done
