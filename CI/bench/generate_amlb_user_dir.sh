#!/usr/bin/env bash

MODULE=$1
REPOSITORY=$2
BRANCH=$3
SHORT_SHA=$4
PR_NUMBER=$5
#Parameter Called Module will come here which will pick the specific module to prepare user dir and will replace tabular
#Move the generate framework up one directory
#Master run will generate user_dir for both tabular and timeseries

# generate tabular configs
python $(dirname "$0")/generate_framework.py --repository https://github.com/$REPOSITORY.git --branch $BRANCH --module $MODULE
if [ -n "$PR_NUMBER" ]
then
    CONFIG_PATH=$MODULE/$PR_NUMBER
else
    CONFIG_PATH=$MODULE/$BRANCH
fi

# keep commit sha for future reference
aws s3 cp --recursive $(dirname "$0")/$MODULE/amlb_user_dir/ s3://autogluon-ci-benchmark/configs/$CONFIG_PATH/$SHORT_SHA/
aws s3 rm --recursive s3://autogluon-ci-benchmark/configs/$CONFIG_PATH/latest/
aws s3 cp --recursive $(dirname "$0")/$MODULE/amlb_user_dir/ s3://autogluon-ci-benchmark/configs/$CONFIG_PATH/latest/
