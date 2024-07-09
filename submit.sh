#!/bin/bash

# Define the models and datasets
models=(
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-70b-chat-hf"
    "meta-llama/Llama-2-13b-chat-hf"

    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Meta-Llama-3-70B-Instruct"

    "mistralai/Mistral-7B-Instruct-v0.3"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
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
for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for prompt in "${prompts[@]}"
        do
			sbatch --job-name "${model}-${prompt}-${dataset}" run_job_2gpus.sh ${model} ${prompt} ${dataset}
        done
    done
done