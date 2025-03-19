#!/bin/bash
#SBATCH --job-name=s_llama_ner
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192M
#SBATCH --output=logs/s_llama_ner.out
#SBATCH --error=logs/s_llama_ner.err


source ~/.bashrc
conda activate honours


python /users/40624421/sharedscratch/python_scripts/s_llama_ner.py



