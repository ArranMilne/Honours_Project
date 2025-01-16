#!/bin/bash
# Options SBATCH :

#SBATCH --job-name=a_llama
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --output=logs/a_llama.out
#SBATCH --error=logs/a_llama.err
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:1





source ~/.bashrc
conda activate honours


python <<EOF


from transformers import LlamaForCausalLM, LlamaTokenizer


model_name = "meta-llama/Llama-2-7b-hf"
save_directory = "/users/40624421/models/llama-2-7b"


tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.save_pretrained(save_directory)

model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=True)
model.save_pretrained(save_directory)


EOF

