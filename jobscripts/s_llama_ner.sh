#!/bin/bash
#SBATCH --job-name=llama_ner_skillspan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/s_llama_ner.out
#SBATCH --error=logs/s_llama_ner.err


source ~/.bashrc
conda activate honours


python /users/40624421/sharedscratch/python_scripts/s_llama_ner.py

