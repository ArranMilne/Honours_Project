#!/bin/bash
# Options SBATCH :

#SBATCH --job-name=a_paper13
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --output=logs/paper13.out
#SBATCH --error=logs/paper13.err
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:1 



#Using a conda virtual environment which already has all the 'pip installs'.

source ~/.bashrc
conda activate honours

model_dir="localscratch/llama/models/llama-2-7b"
raw_data_dir="paper_13/SCESC-LLM-skill-extraction/data/annotated/raw/"
output_dir="paper_13/SCESC-LLM-skill-extraction/output/"

cd /users/40624421/paper_13/SCESC-LLM-skill-extraction


#Running this as starting point.
sh run.sh $model_dir $raw_data_dir $output_dir


