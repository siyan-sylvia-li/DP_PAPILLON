#!/usr/bin/env python3
"""
train_ifs_grpo.py — GRPO training to reduce Invariance Failure Score (IFS).

For each problem that contains a name, we create N variants by substituting
different names (from the pre-filtered gsm8k_train_gliner_pii.csv or a general
PUPA-style CSV).  GRPO trains the model to produce invariant outputs regardless
of which name appears in the prompt.

Reward functions (choose one or combine):

  A. correctness_reward   (--use_correctness_reward, task-specific)
       +1.0 if extracted numeric answer matches ground truth, -1.0 if wrong.
       Only applicable to GSM8K-style tasks with a ground-truth answer_key.

  B. rdd_reward           (--use_rdd_reward, task-agnostic)  ← OPTION B
       reward_i = -RDD(i)
       where RDD(i) = |log p(c_i | p_i) − mean_{j≠i} log p(c_i | p_j)|
       Uses a frozen reference model for all log-prob scoring.
       Completely task-agnostic: no ground-truth labels needed.
       Works for GSM8K, PAPILLON, TruthfulQA, or any name-variant task.

  C. invariance_reward    (--use_invariance_reward, lightweight task-agnostic)
       Within each batch, groups completions by group_id and rewards agreement
       with the modal extracted answer. Cheaper than rdd_reward but less
       principled (requires answer extraction).

Memory note for rdd_reward:
  The reference model is loaded separately from the TRL-managed reference copy.
  With --use_4bit, each model copy is ~4 GB.  A 7B setup requires ~12 GB total
  (training model + TRL reference + RDD scoring reference).

Usage:

  # Option B only (task-agnostic, no ground truth):
  python train_ifs_grpo.py \\
      --model_name Qwen/Qwen2.5-7B-Instruct \\
      --data_file ../pupa_nemotron/gsm8k_train_gliner_pii.csv \\
      --output_dir ./grpo_ifs_rdd \\
      --num_variants 8 --use_rdd_reward --use_lora --use_4bit

  # Option B + correctness (GSM8K only):
  python train_ifs_grpo.py \\
      --model_name Qwen/Qwen2.5-7B-Instruct \\
      --data_file ../pupa_nemotron/gsm8k_train_gliner_pii.csv \\
      --output_dir ./grpo_ifs_combined \\
      --num_variants 8 --use_rdd_reward --use_correctness_reward \\
      --use_lora --use_4bit

  # Quick debug run:
  python train_ifs_grpo.py \\
      --model_name Qwen/Qwen2.5-1.5B-Instruct \\
      --data_file ../pupa_nemotron/gsm8k_train_gliner_pii.csv \\
      --output_dir ./grpo_ifs_debug \\
      --num_variants 4 --max_examples 50 --use_rdd_reward

Requirements (beyond environment.yml):
    pip install trl>=0.12 peft accelerate
"""

import argparse
import logging
import os
import re
import sys
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Few-shot prompt prefix (mirrors GSM8kCustomizer in task_customizer.py)
# ---------------------------------------------------------------------------

PROMPT_PREFIX = """\
As an expert problem solver, solve step by step the following mathematical questions.

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. The final answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. The final answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. The final answer is 39

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8. The final answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9. The final answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29. The final answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33. The final answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8. The final answer is 8.

Q: {question}
A: Let's think step by step."""


def format_prompt(question: str) -> list[dict]:
    return [{"role": "user", "content": PROMPT_PREFIX.format(question=question)}]


# ---------------------------------------------------------------------------
# Grouped batch sampler — ensures same-group variants co-occur in each batch
# ---------------------------------------------------------------------------

class GroupedBatchSampler(Sampler):
    """
    Yields batches where every index in a batch shares the same group_id.

    Within each group, indices are shuffled. Groups themselves are also
    shuffled each epoch.  If a group has fewer than batch_size variants,
    the batch is padded by repeating indices from the same group so that
    each batch is exactly batch_size.  This guarantees that rdd_reward
    always sees at least two distinct prompts per group_id in the batch,
    making reward_std > 0 and GRPO loss > 0.
    """

    def __init__(self, group_ids: list[int], batch_size: int, seed: int = 42):
        from collections import defaultdict
        self.batch_size = batch_size
        self.seed = seed
        buckets: dict[int, list[int]] = defaultdict(list)
        for idx, gid in enumerate(group_ids):
            buckets[gid].append(idx)
        self.buckets = list(buckets.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        groups = [rng.permutation(b).tolist() for b in self.buckets]
        rng.shuffle(groups)
        for group in groups:
            # Tile the group so we always emit full batches
            tiled = (group * ((self.batch_size // len(group)) + 1))[: self.batch_size]
            yield tiled

    def __len__(self):
        return len(self.buckets)


# ---------------------------------------------------------------------------
# Answer extraction  (used by correctness_reward and invariance_reward)
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> Optional[int]:
    """
    Extract the final numeric answer from a model completion.

    Tries three patterns in order:
      1. "The final answer is <N>"
      2. "#### <N>"  (GSM8K ground-truth format)
      3. Last integer-like token in the text
    Returns None if no number is found.
    """
    m = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is\s+(-?[\d,]+)", text)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            pass

    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            pass

    tokens = text.split()
    for tok in reversed(tokens):
        cleaned = re.sub(r"[^\d\-]", "", tok)
        if cleaned and cleaned != "-":
            try:
                return int(cleaned)
            except ValueError:
                continue

    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gsm8k_variants(
    data_file: str,
    num_variants: int,
    max_examples: Optional[int],
    tokenizer: AutoTokenizer,
    seed: int = 42,
) -> Dataset:
    """
    Build the training dataset from gsm8k_train_gliner_pii.csv.

    Each row describes one math problem with a detected name (pii) and a list
    of substitute names (substitutes, '||'-separated).  We produce:
      - the original question  (variant index 0)
      - up to (num_variants - 1) name-substituted copies

    Dataset columns:
      prompt       : fully-formatted chat string consumed by GRPOTrainer
      prompt_str   : same string, passed as a kwarg to reward functions
      group_id     : int identifying the base question across all variants
      answer_key   : correct numeric answer (int); -1 if unavailable
      variant_name : which name is used in this variant (for debugging)
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(data_file)
    if max_examples is not None:
        df = df.iloc[:max_examples]

    records = []
    for group_id, (_, row) in enumerate(df.iterrows()):
        original_question = str(row["original_question"])
        pii               = str(row["pii"])
        substitutes       = [s for s in str(row["substitutes"]).split("||") if s.strip()]

        # Parse the ground-truth answer (format: "step ... #### 42").
        # answer_key = -1 signals "unavailable" to reward functions.
        raw_answer = str(row.get("original_answer", ""))
        try:
            answer_key = int(raw_answer.split("####")[-1].strip().replace(",", "").replace(".", ""))
        except (ValueError, IndexError):
            answer_key = -1

        variants: list[tuple[str, str]] = [(original_question, pii)]
        if substitutes:
            n_sub = min(num_variants - 1, len(substitutes))
            chosen = rng.choice(substitutes, size=n_sub, replace=False)
            for name in chosen:
                variants.append((original_question.replace(pii, name), name))

        for question, variant_name in variants:
            msgs = format_prompt(question)
            prompt_str = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            records.append({
                "prompt":       prompt_str,
                "prompt_str":   prompt_str,   # explicit kwarg for reward fns
                "group_id":     group_id,
                "answer_key":   answer_key,
                "variant_name": variant_name,
            })

    logger.info("Dataset built: %d total examples from %d base questions.", len(records), len(df))
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Reward A: correctness  (task-specific, GSM8K)
# ---------------------------------------------------------------------------

def reward_correctness(
    completions: list[str],
    answer_key: list[int],
    **kwargs,
) -> list[float]:
    """
    +1.0  if extracted answer matches ground-truth answer_key
    -1.0  if extracted but wrong
     0.0  if unparseable OR answer_key == -1 (unavailable)
    """
    rewards = []
    for comp, key in zip(completions, answer_key):
        if int(key) == -1:
            rewards.append(0.0)
            continue
        pred = extract_answer(comp)
        if pred is None:
            rewards.append(0.0)
        elif pred == int(key):
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


# ---------------------------------------------------------------------------
# Reward B: RDD  (task-agnostic, Option B)
# ---------------------------------------------------------------------------

def make_rdd_reward(ref_model: AutoModelForCausalLM, ref_tokenizer: AutoTokenizer):
    """
    Factory for the RDD-based invariance reward.

    For each completion c_i generated from prompt variant p_i:

        RDD(i) = |log p(c_i | p_i) - mean_{j≠i} log p(c_i | p_j)|
        reward_i = -RDD(i)

    where p_j are the OTHER name-variant prompts sharing the same group_id
    in the current batch.  The reward is 0.0 when a group has only one
    variant visible in the batch (no cross-variant scoring possible).

    All log-probs are length-normalised to avoid penalising longer completions.

    Uses ref_model (frozen) for all scoring, so reward hacking through the
    policy's own log-probs is not possible.
    """
    ref_model.eval()
    _device = next(ref_model.parameters()).device

    @torch.no_grad()
    def _norm_logprob(prompt: str, completion: str) -> float:
        """Length-normalised log p(completion | prompt) under ref_model."""
        p_ids = ref_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        )["input_ids"].to(_device)
        c_ids = ref_tokenizer(
            completion, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(_device)

        input_ids = torch.cat([p_ids, c_ids], dim=1)
        logits    = ref_model(input_ids=input_ids).logits           # (1, L, V)
        lp        = F.log_softmax(logits, dim=-1)

        # Shift: lp[:, t, :] predicts token at t+1
        shift_lp  = lp[:, :-1, :]                                  # (1, L-1, V)
        shift_ids = input_ids[:, 1:]                                # (1, L-1)

        pl = p_ids.shape[1]
        cl = c_ids.shape[1]
        positions = torch.arange(pl - 1, pl - 1 + cl, device=_device)

        tok_lp = shift_lp[0, positions, shift_ids[0, positions]]   # (cl,)
        return (tok_lp.sum() / cl).item()

    def rdd_reward(
        completions: list[str],
        group_id: list[int],
        prompt_str: list[str],
        **kwargs,
    ) -> list[float]:
        """
        Compute -RDD for each completion.

        TRL repeats each dataset row G times (once per generation), so
        group_id and prompt_str are already length len(completions).

        Within a batch, multiple prompts may share the same group_id —
        these are different name variants of the same base question.
        We collect the unique prompt string for each variant and compute
        cross-variant log-probs for every completion in the group.

        A per-batch cache avoids redundant forward passes: identical
        (prompt, completion) pairs (across G rollouts of the same variant)
        are scored only once.
        """
        gids    = list(group_id)
        prompts = list(prompt_str)

        # Unique prompts per group_id visible in this batch
        group_unique_prompts: dict[int, list[str]] = {}
        for gid, p in zip(gids, prompts):
            bucket = group_unique_prompts.setdefault(gid, [])
            if p not in bucket:
                bucket.append(p)

        # Score cache: (hash(prompt), hash(completion)) -> norm_logprob
        # Using hash() for speed; collisions are astronomically unlikely here.
        _cache: dict[tuple[int, int], float] = {}

        def cached_score(p: str, c: str) -> float:
            key = (hash(p), hash(c))
            if key not in _cache:
                _cache[key] = _norm_logprob(p, c)
            return _cache[key]

        rewards: list[float] = []
        for comp, gid, self_p in zip(completions, gids, prompts):
            other_ps = [p for p in group_unique_prompts[gid] if p != self_p]

            if not other_ps:
                # Only one variant present in this batch for this group.
                # Cannot compute RDD — assign neutral reward.
                rewards.append(0.0)
                continue

            lp_self  = cached_score(self_p, comp)
            lp_cross = [cached_score(p, comp) for p in other_ps]

            rdd = abs(lp_self - float(np.mean(lp_cross)))
            rewards.append(-rdd)

        return rewards

    rdd_reward.__name__ = "rdd_reward"
    return rdd_reward


# ---------------------------------------------------------------------------
# Reward C: invariance  (lightweight task-agnostic, answer-extraction based)
# ---------------------------------------------------------------------------

def make_invariance_reward():
    """
    Within each batch, group completions by group_id and reward agreement
    with the within-group modal extracted answer.

    +0.5  if the extracted answer matches the modal answer for this group
    -0.5  if it disagrees
     0.0  if unparseable OR only one completion in the group
    """
    def invariance_reward(
        completions: list[str],
        group_id: list[int],
        **kwargs,
    ) -> list[float]:
        extracted = [extract_answer(c) for c in completions]
        gids      = list(group_id)

        group_answers: dict[int, list] = {}
        for gid, ans in zip(gids, extracted):
            group_answers.setdefault(gid, []).append(ans)

        group_modal: dict[int, Optional[int]] = {}
        for gid, answers in group_answers.items():
            valid = [a for a in answers if a is not None]
            group_modal[gid] = Counter(valid).most_common(1)[0][0] if valid else None

        rewards = []
        for gid, ans in zip(gids, extracted):
            modal = group_modal.get(gid)
            if ans is None or modal is None or len(group_answers[gid]) <= 1:
                rewards.append(0.0)
            elif ans == modal:
                rewards.append(0.5)
            else:
                rewards.append(-0.5)
        return rewards

    invariance_reward.__name__ = "invariance_reward"
    return invariance_reward


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO training to reduce IFS via name-variant groups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Model")
    g.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    g.add_argument("--use_lora", action="store_true")
    g.add_argument("--lora_rank", type=int, default=16)
    g.add_argument("--lora_alpha", type=int, default=32)
    g.add_argument("--use_4bit", action="store_true")
    g.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])

    g = p.add_argument_group("Data")
    g.add_argument("--data_file", required=True,
                   help="Path to gsm8k_train_gliner_pii.csv.")
    g.add_argument("--num_variants", type=int, default=8,
                   help="Name variants per question (original + substitutes).")
    g.add_argument("--max_examples", type=int, default=None,
                   help="Cap base questions (for debugging).")
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("Reward")
    g.add_argument("--use_rdd_reward", action="store_true",
                   help="[Option B] Task-agnostic RDD reward using cross-variant "
                        "log-probs from a frozen reference model.")
    g.add_argument("--ref_model_name", type=str, default=None,
                   help="Reference model for RDD scoring. Defaults to --model_name.")
    g.add_argument("--use_correctness_reward", action="store_true",
                   help="[Option A] Task-specific correctness reward (GSM8K only).")
    g.add_argument("--use_invariance_reward", action="store_true",
                   help="[Option C] Lightweight modal-answer consistency reward.")

    g = p.add_argument_group("Training")
    g.add_argument("--output_dir", default="./grpo_ifs_output")
    g.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to a checkpoint directory, or 'last' to resume "
                        "from the most recent checkpoint in --output_dir.")
    g.add_argument("--num_epochs", type=int, default=1)
    g.add_argument("--per_device_batch_size", type=int, default=2)
    g.add_argument("--grad_accum_steps", type=int, default=4)
    g.add_argument("--num_generations", type=int, default=8,
                   help="GRPO group size G: rollout completions per prompt.")
    g.add_argument("--max_new_tokens", type=int, default=512)
    g.add_argument("--max_prompt_length", type=int, default=1024)
    g.add_argument("--learning_rate", type=float, default=1e-6)
    g.add_argument("--beta", type=float, default=0.01,
                   help="KL-penalty β in the GRPO objective.")
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--top_p", type=float, default=0.95)
    g.add_argument("--logging_steps", type=int, default=10)
    g.add_argument("--save_steps", type=int, default=50)

    g = p.add_argument_group("vLLM rollouts (TRL >= 0.14)")
    g.add_argument("--use_vllm", action="store_true")
    g.add_argument("--vllm_server_host", default="0.0.0.0")
    g.add_argument("--vllm_server_port", type=int, default=8000)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not any([args.use_rdd_reward, args.use_correctness_reward, args.use_invariance_reward]):
        logger.warning(
            "No reward function selected. Pass at least one of "
            "--use_rdd_reward, --use_correctness_reward, --use_invariance_reward."
        )

    # Tokeniser
    logger.info("Loading tokeniser: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=os.path.isdir(args.model_name),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    logger.info("Building dataset from: %s", args.data_file)
    train_dataset = load_gsm8k_variants(
        data_file=args.data_file,
        num_variants=args.num_variants,
        max_examples=args.max_examples,
        tokenizer=tokenizer,
        seed=args.seed,
    )
    logger.info("Total training examples: %d", len(train_dataset))

    # Reward functions
    reward_fns = []

    if args.use_rdd_reward:
        ref_name = args.ref_model_name or args.model_name
        logger.info("Loading reference model for RDD scoring: %s", ref_name)
        ref_kwargs: dict = {"dtype": getattr(torch, args.dtype)}
        if args.use_4bit:
            from transformers import BitsAndBytesConfig
            ref_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_name, local_files_only=os.path.isdir(ref_name), **ref_kwargs
        )
        ref_model = ref_model.to("cuda" if torch.cuda.is_available() else "cpu")
        ref_model.eval()
        reward_fns.append(make_rdd_reward(ref_model, tokenizer))
        logger.info("RDD reward enabled (ref model: %s).", ref_name)

    if args.use_correctness_reward:
        reward_fns.append(reward_correctness)
        logger.info("Correctness reward enabled.")

    if args.use_invariance_reward:
        reward_fns.append(make_invariance_reward())
        logger.info("Invariance reward enabled.")

    # LoRA
    peft_config = None
    if args.use_lora:
        from peft import LoraConfig, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        logger.info("LoRA enabled (r=%d, α=%d).", args.lora_rank, args.lora_alpha)

    # Model kwargs
    torch_dtype = getattr(torch, args.dtype)
    model_kwargs: dict = {}
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # GRPO config
    grpo_kwargs: dict = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        # max_prompt_length=args.max_prompt_length,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    if args.use_vllm:
        grpo_kwargs.update(
            use_vllm=True,
            vllm_server_host=args.vllm_server_host,
            vllm_server_port=args.vllm_server_port,
        )
        logger.info("vLLM server: %s:%d", args.vllm_server_host, args.vllm_server_port)

    grpo_config = GRPOConfig(**grpo_kwargs)

    # Subclass to inject grouped batch sampler so same-group variants
    # always co-occur in a batch (required for rdd_reward to be non-zero).
    class GroupedGRPOTrainer(GRPOTrainer):
        def get_train_dataloader(self):
            if args.use_rdd_reward:
                from torch.utils.data import DataLoader
                sampler = GroupedBatchSampler(
                    group_ids=self.train_dataset["group_id"],
                    batch_size=self.args.per_device_train_batch_size * self.args.num_generations,
                    seed=self.args.seed,
                )
                return DataLoader(
                    self.train_dataset,
                    batch_sampler=sampler,
                    collate_fn=self.data_collator,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                )
            return super().get_train_dataloader()

    trainer = GroupedGRPOTrainer(
        model=args.model_name,
        reward_funcs=reward_fns,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        **model_kwargs,
    )

    resume = args.resume_from_checkpoint
    if resume == "last":
        resume = True  # HF Trainer will find the latest checkpoint in output_dir

    logger.info("Starting GRPO training…")
    trainer.train(resume_from_checkpoint=resume)

    logger.info("Saving model to: %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
