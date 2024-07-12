#!/bin/bash

datasets=(
    "inspec"
    "kp20k"
    "semeval2010"
    "semeval2017"
)

for data in "${datasets[@]}"
do
	# Run the command and log the output, continue to next command even if there is an error
	sbatch --job-name "fs-${data}" run_fs_fixed_LLM-job.sh LLMClosestTraining 1 ${data}
done