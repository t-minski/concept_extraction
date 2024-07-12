#!/bin/bash

datasets=(
    "kp20k"
    "semeval2010"
    "semeval2017"
)

for data in "${datasets[@]}"
do
	sbatch --job-name "transfer-${data}" run_fs_fixed_LLM_transfer-job.sh inspec ${data}
done