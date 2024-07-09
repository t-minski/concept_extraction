#!/bin/bash

# Define the models and datasets
models=(
    # "SpacyEntities"
    # "SpacyNounChunks"
    # "TfIdfEntities"
    # "YakeEntities"
    # "SummaEntities"
    # "RakeEntities"
    # "PyateBasicsEntities"
    # "PyateComboBasicEntities"
    # "PyateCvaluesEntities"
    # "LSAEntities"
    # "LDAEntities"
    # "pke_FirstPhrases"
    # "pke_TextRank"
    # "pke_SingleRank"
    # "pke_TopicRank"
    # "pke_MultipartiteRank"
    # "pke_TfIdf"
    # "pke_TopicalPageRank"
    # "pke_YAKE"
    # "pke_KPMiner"
    # "pke_Kea"
    # "KeyBERTEntities"
    # "Llama2_7b_ZeroShotEntities"
    # "Llama2_7b_OneShotEntities"
    # "Llama3_8b_ZeroShotEntities"
    # "Llama3_8b_OneShotEntities"
    # "Mistral_7b_ZeroShotEntities"
    # "Mistral_7b_OneShotEntities"
    # "Mixtral_7b_ZeroShotEntities"
    # "Mixtral_7b_OneShotEntities"
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-70b-chat-hf"
    "meta-llama/Llama-2-13b-chat-hf"

    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Meta-Llama-3-70B-Instruct"

    "mistralai/Mistral-7B-Instruct-v0.3"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "mistralai/Mixtral-8x22B-v0.1"

    # "gpt-3.5-turbo"

)

datasets=(
    "inspec"
    "kp20k"
    "semeval2010"
    "semeval2017"
)

output_folder="output2"
log_folder="logs2"

# Create the folders if it does not exist
mkdir -p ${log_folder}

# Define templates
templates=(
    #"template_1"
    #"template_2"
    #"template_3"
    #"template_4"
    #"template_5"
    #"template_6"
    "simple_keywords"
    "simple_keyphrases"
    "simple_concepts"
    "simple_classes"
    "simple_entities"
    "simple_topics"
)
#python3 con/main.py --models AdvancedConceptExtractor --datasets inspec --template template_1 --output con_output2 -v > con_logs2/AdvancedConceptExtractor_inspec_template_1.log 2>&1

# Loop over each model and dataset combination
for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for template in "${templates[@]}"
        do
            log_file="${log_folder}/${model}_${dataset}_${template}.log"
            
            echo "Running model ${model} on dataset ${dataset} with template ${template}, output will be logged to ${log_file}"
            
            # Run the command and log the output, continue to next command even if there is an error
            #python3 main.py --models ${model} --datasets ${dataset} --template ${template} --output ${output_folder} -v > ${log_file} 2>&1
            python3 main.py --models class=LLMBaseModel,prompt=${template},model_name=${model},with_confidence=False,batched_generation=False  --datasets ${dataset} --output ${output_folder} -v > ${log_file} 2>&1

            if [ $? -ne 0 ]; then
                echo "Error encountered with model ${model} on dataset ${dataset} using template ${template}. Check ${log_file} for details."
            else
                echo "Successfully completed model ${model} on dataset ${dataset} using template ${template}. Log saved to ${log_file}."
            fi
        done
    done
done

echo "All tasks have been submitted."            