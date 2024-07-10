#!/bin/bash

# Define the models and datasets
models=(
    "SpacyEntities"
    "SpacyNounChunks"
    "TfIdfEntities"
    "YakeEntities"
    "SummaEntities"
    "RakeEntities"
    "PyateBasicsEntities"
    "PyateComboBasicEntities"
    "PyateCvaluesEntities"
    "LSAEntities"
    "LDAEntities"
    "pke_FirstPhrases"
    "pke_TextRank"
    "pke_SingleRank"
    "pke_TopicRank"
    "pke_MultipartiteRank"
    "pke_TfIdf"
    "pke_TopicalPageRank"
    "pke_YAKE"
    "pke_KPMiner"
    "pke_Kea"
    "KeyBERTEntities"
)

datasets=(
    "inspec"
    "kp20k"
    "semeval2010"
    "semeval2017"
)

output_folder="output"
log_folder="logs"

# Create the folders if it does not exist
mkdir -p ${log_folder}

# Loop over each model and dataset combination
for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        
        log_file="${log_folder}/${model}_${dataset}.log"
        
        echo "Running model ${model} on dataset ${dataset}, output will be logged to ${log_file}"
        
        # Run the command and log the output, continue to next command even if there is an error
        python3 main.py --models ${model} --datasets ${dataset} --output ${output_folder} -v > ${log_file} 2>&1

        if [ $? -ne 0 ]; then
            echo "Error encountered with model ${model} on dataset ${dataset}. Check ${log_file} for details."
        else
            echo "Successfully completed model ${model} on dataset ${dataset}. Log saved to ${log_file}."
        
    done
done

echo "All tasks have been submitted."            