#!/bin/bash
#SBATCH --partition=devel
#SBATCH --ntasks=15
#SBATCH --time=00:29:00
#SBATCH --gres=gpu:A100:1

source activate conexion

log_file="./output/${1////*}_${3}_${2}.log"
python main.py --models class=LLMBaseModel,prompt=${2},model_name=${1},with_confidence=False,batched_generation=False --datasets ${3} --output ./output -v > ${log_file} 2>&1
