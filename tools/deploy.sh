#!/bin/bash
#
# Note that
# UUID=$(uuidgen)
# works on Mac OS, on linux change this too:
# UUID=$(cat /proc/sys/kernel/random/uuid)

set -ex
set -o pipefail

# Script arguments
REMOTE=$1
BRANCH=$2
PACKAGE=$3
VERSION=$4

# Check versions are there, this is a crude way to do it but it works
grep "^$PACKAGE==$VERSION\$" requirements/self.txt
grep '^__version__ = "'$VERSION'"$' bayesmark/__init__.py
grep 'version="'$VERSION'",$' setup.py

# Where envs go
ENVS=~/envs
# Which python version this uses
PY=python3.6
# Which env contains twine and py version we use
TWINE_ENV=twine_env
# Where to run tar ball tests from
TEST_DIR=~/tmp/deploy_tests

mkdir -p $TEST_DIR

# Get the dir
REPO_DIR=$(pwd)
git checkout $BRANCH

# Fail if untracked files and clean
test -z "$(git status --porcelain)"
git clean -x -ff -d

# Run tests locally and cleanup
./integration_test.sh
./test.sh
git reset --hard HEAD
git clean -x -ff -d
test -z "$(git status --porcelain)"

# push to remote and check
git push -u $REMOTE $BRANCH
git diff $BRANCH $REMOTE/$BRANCH --quiet

# See if tests pass remote, TODO use travis CLI
read -t 1 -n 10000 discard || true
read -p "Travis tests pass [y/n]? " -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# test tar ball
source $ENVS/$TWINE_ENV/bin/activate
python setup.py sdist
twine check dist/*
deactivate
cd $TEST_DIR
UUID=$(uuidgen)
mkdir $UUID
cd $UUID
virtualenv env --python=$PY
source ./env/bin/activate
pip install -r $REPO_DIR/requirements/test.txt
pip install $REPO_DIR/dist/*.tar.gz
cp -r $REPO_DIR/test .
pytest test/ -s -v --hypothesis-seed=0 --disable-pytest-warnings
deactivate
cd $REPO_DIR
# Cleanup since we will build again
git clean -x -ff -d
test -z "$(git status --porcelain)"

# merge master
# Fail if origin and local differ
git checkout $BRANCH
git diff master $REMOTE/master --quiet
git merge master --no-commit
# Fail if not clean
test -z "$(git status --porcelain)"

# merge to master
git checkout master
git merge $BRANCH --squash --no-commit
git status
read -t 1 -n 10000 discard || true
read -p "Commit message (CTRL-C to abort): "
git commit -m "$REPLY"
# Fail if not clean
test -z "$(git status --porcelain)"

# Run tests locally and cleanup
./integration_test.sh
./test.sh
git reset --hard HEAD
git clean -x -ff -d
test -z "$(git status --porcelain)"

# test tar ball
source $ENVS/$TWINE_ENV/bin/activate
python setup.py sdist
twine check dist/*
deactivate
cd $TEST_DIR
UUID=$(uuidgen)
mkdir $UUID
cd $UUID
virtualenv env --python=$PY
source ./env/bin/activate
pip install -r $REPO_DIR/requirements/test.txt
pip install $REPO_DIR/dist/*.tar.gz
cp -r $REPO_DIR/test .
pytest test/ -s -v --hypothesis-seed=0 --disable-pytest-warnings
deactivate
cd $REPO_DIR

# push to test pypi
source $ENVS/$TWINE_ENV/bin/activate
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
deactivate

# install and test
cd $TEST_DIR
UUID=$(uuidgen)
mkdir $UUID
cd $UUID
virtualenv env --python=$PY
source ./env/bin/activate
pip install -r $REPO_DIR/requirements/test.txt
pip install $PACKAGE==$VERSION --index-url https://test.pypi.org/simple/
cp -r $REPO_DIR/test .
pytest test/ -s -v --hypothesis-seed=0 --disable-pytest-warnings
deactivate
cd $REPO_DIR

# push to remote and check
git push $REMOTE master
git diff master $REMOTE/master --quiet

# Show sha256sum in case we want to check against PyPI test
sha256sum dist/*

# See if tests pass remote, TODO use travis CLI
read -t 1 -n 10000 discard || true
read -p "Travis tests pass, and push to PyPI? This cannot be undone. [y/n]" -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# push to full pypi
source $ENVS/$TWINE_ENV/bin/activate
twine upload dist/*
deactivate

# install and test
cd $TEST_DIR
UUID=$(uuidgen)
mkdir $UUID
cd $UUID
virtualenv env --python=$PY
source ./env/bin/activate
pip install -r $REPO_DIR/requirements/test.txt
pip install $PACKAGE==$VERSION
cp -r $REPO_DIR/test .
pytest test/ -s -v --hypothesis-seed=0 --disable-pytest-warnings
deactivate
cd $REPO_DIR

# clean and tag
git clean -x -ff -d
test -z "$(git status --porcelain)"
git tag -a v$VERSION -m "$PACKAGE version $VERSION"
git push $REMOTE v$VERSION

# remind user to archive/delete branch
echo "remember to delete branch $BRANCH, and update readthedocs.io"
echo "done"
