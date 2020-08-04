#!/bin/bash

set -ex
set -o pipefail

# Be able to check if using version out of tar ball
which bayesmark-launch
which bayesmark-exp
which bayesmark-agg
which bayesmark-anal

DB_ROOT=./notebooks
DBID=bo_example_folder

bayesmark-launch -n 15 -r 2 -dir $DB_ROOT -b $DBID -o RandomSearch PySOT OpenTuner-BanditA -c SVM DT -d boston breast -v
bayesmark-agg -dir $DB_ROOT -b $DBID
bayesmark-anal -dir $DB_ROOT -b $DBID -v

# Try ipynb export
python -m ipykernel install --name=bobm_ipynb --user
jupyter nbconvert --to html --execute notebooks/plot_mean_score.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to html --execute notebooks/plot_test_case.ipynb --ExecutePreprocessor.timeout=-1

# Try dry run
bayesmark-launch -n 15 -r 3 -dir $DB_ROOT -b $DBID -o RandomSearch PySOT OpenTuner-BanditA -c SVM DT -nj 50 -v

# Try again but use the custom optimizers
mv $DB_ROOT/$DBID old
bayesmark-launch -n 15 -r 1 -dir $DB_ROOT -b $DBID -o RandomSearch PySOT-New OpenTuner-BanditA-New -c SVM DT --opt-root ./example_opt_root -d boston breast -v
bayesmark-agg -dir $DB_ROOT -b $DBID
bayesmark-anal -dir $DB_ROOT -b $DBID -v

# Export again
jupyter nbconvert --to html --execute notebooks/plot_mean_score.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to html --execute notebooks/plot_test_case.ipynb --ExecutePreprocessor.timeout=-1

# Try dry run
bayesmark-launch -n 15 -r 2 -dir $DB_ROOT -b $DBID -o RandomSearch PySOT-New OpenTuner-BanditA-New -c SVM DT --opt-root ./example_opt_root -nj 50 -v

echo "success"
