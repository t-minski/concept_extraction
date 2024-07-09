#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=15
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A100:2


log_file="./output/${1////*}_${3}_${2}.log"
mkdir -p ${log_file}
python main.py --models class=LLMBaseModel,prompt=${2},model_name=${1},with_confidence=False,batched_generation=False --datasets ${3} --output ./output -v > ${log_file} 2>&1