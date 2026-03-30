  # python train_ifs_grpo.py \
  #     --model_name Qwen/Qwen2.5-1.5B-Instruct \
  #     --data_file ../pupa_nemotron/gsm8k_train_gliner_pii.csv \
  #     --output_dir ./grpo_ifs_debug \
  #     --num_variants 4 --use_rdd_reward --per_device_batch_size 8 --num_generations 8
  python train_ifs_grpo.py \
      --model_name Qwen/Qwen2.5-1.5B-Instruct \
      --data_file ../pupa_nemotron/gsm8k_train_gliner_pii.csv \
      --output_dir ./grpo_ifs_qwen_15b_inst_correctness \
      --num_variants 4 --use_rdd_reward --use_correctness_reward --per_device_batch_size 8 --num_generations 8 --num_epochs 1 --resume_from_checkpoint last


  python compute_ifs.py --model_name /local-storage/interaction/siyanli/DP_PAPILLON/papillon/grpo_ifs_qwen_15b_inst_correctness --task GSM8k --use_vllm --logprob_batch_size 128