#!/bin/bash
module load cuda/12.6.1
module load conda

conda activate pupa

cd /ocean/projects/cis250134p/shared/DP_PAPILLON/papillon
export CUDA_VISIBLE_DEVICES=0,1
python compute_ifs.py --model_name Qwen/Qwen2.5-7B-Instruct --task GSM8k --use_vllm --logprob_batch_size 128
python compute_ifs.py --model_name Qwen/Qwen3-4B-Instruct-2507 --task GSM8k --use_vllm --logprob_batch_size 128
python compute_ifs.py --model_name google/gemma-3-4b-it --task GSM8k --use_vllm --logprob_batch_size 128
