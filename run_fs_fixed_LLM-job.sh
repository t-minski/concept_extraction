#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=15
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A100:2

source activate conexion

log_file="./output/fs_fixed_Meta-Llama-3-70B-Instruct_${1}_${2}.log"
python main.py --models class=${1},prompt=fs_keyphrases,model_name=meta-llama/Meta-Llama-3-70B-Instruct,number_of_examples=${2},load_in_4bit=True --datasets ${3} --output ./output -v > ${log_file} 2>&1
