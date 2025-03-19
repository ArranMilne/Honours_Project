#!/bin/bash
#SBATCH --job-name=s_bert_ner
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --output=logs/s_bert_ner_hp.out
#SBATCH --error=logs/s_bert_ner_hp.err




source ~/.bashrc
conda activate honours



python /users/40624421/sharedscratch/python_scripts/s_bert_ner_hp.py








