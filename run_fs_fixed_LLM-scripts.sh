#!/bin/bash

# Define the models and datasets
models=(
    "meta-llama/Llama-2-7b-chat-hf"
    # "meta-llama/Llama-2-70b-chat-hf"
    # "meta-llama/Llama-2-13b-chat-hf"

    # "meta-llama/Meta-Llama-3-8B-Instruct"
    # "meta-llama/Meta-Llama-3-70B-Instruct"

    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # "mistralai/Mixtral-8x22B-v0.1"

    # "gpt-3.5-turbo"

)

datasets=(
    "inspec"
    # "kp20k"
    # "semeval2010"
    # "semeval2017"
)

output_folder="output"
log_folder="logs"

# Create the folders if it does not exist
mkdir -p ${log_folder}

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

# Loop over each model and dataset combination
for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for class_model in "${class_models[@]}"
        do
            for number in "${numbers[@]}"
            do
                log_file="${log_folder}/${model}_${dataset}_${template}_${number}.log"
                
                echo "Running model ${model} on dataset ${dataset} with template ${template}_${number}, output will be logged to ${log_file}"
                
                # Run the command and log the output, continue to next command even if there is an error
                python3 main.py --models class=${class_model},prompt=fewshot_keyword,model_name=${model},number_of_examples=${number},with_confidence=False,batched_generation=False  --datasets ${dataset} --output ${output_folder} --gpu 1> ${log_file} 2>&1
                if [ $? -ne 0 ]; then
                    echo "Error encountered with model ${model} on dataset ${dataset} using template ${template}_${number}. Check ${log_file} for details."
                else
                    echo "Successfully completed model ${model} on dataset ${dataset} using template ${template}_${number}. Log saved to ${log_file}."
                fi
            done
        done
    done
done

echo "All tasks have been submitted."            