#!/bin/bash

# Define the models and datasets
small_models=(
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-13b-chat-hf"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
)
medium_models=(
    "meta-llama/Llama-2-70b-chat-hf"
    "meta-llama/Meta-Llama-3-70B-Instruct"
)
large_models=(
    "mistralai/Mixtral-8x22B-v0.1"
)


datasets=(
    "inspec"
#    "kp20k"
#    "semeval2010"
#    "semeval2017"
)

# Define prompts
prompts=(
    "simple_keywords"
    "simple_keyphrases"
    "simple_concepts"
    "simple_classes"
    "simple_entities"
    "simple_topics"
)

# Loop over each model and dataset combination


for dataset in "${datasets[@]}"
do
	for prompt in "${prompts[@]}"
	do
		for model in "${small_models[@]}"
		do
			sbatch --job-name "${model}-${prompt}-${dataset}" batch_job_1gpu.sh ${model} ${prompt} ${dataset}
		done
		
		for model in "${medium_models[@]}"
		do
			sbatch --job-name "${model}-${prompt}-${dataset}" batch_job_2gpu.sh ${model} ${prompt} ${dataset}
		done
		
		for model in "${large_models[@]}"
		do
			sbatch --job-name "${model}-${prompt}-${dataset}" batch_job_3gpu.sh ${model} ${prompt} ${dataset}
		done
	done
done
