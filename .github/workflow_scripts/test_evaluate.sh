#!/bin/bash

set -ex

MODULE=$1
TIME_LIMIT=$2

#pip install 
# pip install -U autogluon.bench

#source install
git clone -b v0.4.0 https://github.com/autogluon/autogluon-bench.git
pip install -e ./autogluon-bench

#copy from s3
aws s3 cp --recursive s3://autogluon-ci-benchmark/cleaned/tabular/master/latest/ ./results
aws s3 cp --recursive s3://autogluon-ci-benchmark/cleaned/tabular/timeseries_dev/latest/ ./results

#run evaluation
python CI/bench/test_bench_eval.py --config_path ./ag_bench_runs/$MODULE/ --module_name $MODULE --time_limit $TIME_LIMIT