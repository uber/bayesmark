#!/bin/bash

set -ex
set -o pipefail

# Fail if untracked files so we don't delete them in next step
test -z "$(git status --porcelain)"

# Build from clean repo, delete all ignored files
git clean -x -f -d

# Get everything in place to put inside the wheel
SHA_LONG=$(git rev-parse HEAD)
echo VERSION=\"$SHA_LONG\" >bayesmark/version.py

# Now the actual build
python3.6 setup.py sdist
