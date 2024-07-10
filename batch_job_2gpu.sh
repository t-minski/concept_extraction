#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=15
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A100:2

source activate conexion

log_file="./output/${1////*}_${3}_${2}.log"
python main.py --models class=LLMBaseModel,prompt=${2},model_name=${1},load_in_4bit=True --datasets ${3} --output ./output -v > ${log_file} 2>&1
