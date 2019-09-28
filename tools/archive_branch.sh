#!/bin/bash

set -ex
set -o pipefail

DATE=$(date +"%Y%m%d")
TAGNAME=archive/$DATE-$1

# Fail if untracked files
test -z "$(git status --porcelain)"

# Fail if origin and local differ
git diff $1 origin/$1 --quiet

# Prune remotes for good measure
git remote prune origin

git checkout $1
git tag -a $TAGNAME -m "archived branch $1 on $DATE"
git checkout master
git push origin $TAGNAME

# Make sure we tagged correctly for good measure
diff <(git rev-list $TAGNAME -n 1) <(git rev-parse $1)
git ls-remote --tags origin | grep $(git rev-parse $1)

git branch -D $1
git push origin --delete $1

echo "cleaned up"
