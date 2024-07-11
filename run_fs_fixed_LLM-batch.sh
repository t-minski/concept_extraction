#!/bin/bash

# Define templates
class_models=(
    "LLMRandomTraining"
    "LLMRandomButFixedTraining"
    "LLMClosestTraining"
)

# Define numbers
numbers=(
    "1"
    "3"
    "5"
)


for class_model in "${class_models[@]}"
do
	for number in "${numbers[@]}"
	do
		# Run the command and log the output, continue to next command even if there is an error
		sbatch --job-name "fs-${class_model}-${number}" run_fs_fixed_LLM-job.sh ${class_model} ${number} inspec
	done
done