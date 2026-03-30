  # python train_ifs_grpo.py \
  #     --model_name Qwen/Qwen2.5-1.5B-Instruct \
  #     --data_file ../pupa_nemotron/gsm8k_train_gliner_pii.csv \
  #     --output_dir ./grpo_ifs_debug \
  #     --num_variants 4 --use_rdd_reward --per_device_batch_size 8 --num_generations 8
  python train_ifs_grpo_v2.py \
      --model_name Qwen/Qwen2.5-0.5B-Instruct \
      --data_file ../pupa_nemotron/gsm8k_train_gliner_pii.csv \
      --output_dir ./grpo_ifs_qwen_05b_all \
      --num_variants 4 --use_rdd_reward --use_correctness_reward --per_device_batch_size 8 --num_generations 4 --num_epochs 1 --eval_steps 50 --report_to wandb


  python compute_ifs.py --model_name /local-storage/interaction/siyanli/DP_PAPILLON/papillon/grpo_ifs_qwen_05b_all --task GSM8k --use_vllm --logprob_batch_size 128




# # train correctness only, eval both
# python train_ifs_grpo_v2.py --use_correctness_reward --eval_rdd_reward ...

# # train RDD only, eval both (rdd_fn reused, no double load)
# python train_ifs_grpo_v2.py --use_rdd_reward ...

# # train both, eval both
# python train_ifs_grpo_v2.py --use_rdd_reward --use_correctness_reward ...

# # train correctness only, eval correctness only (no ref model loaded)
# python train_ifs_grpo_v2.py --use_correctness_reward ...
