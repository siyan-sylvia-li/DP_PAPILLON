#!/usr/bin/env python3
"""
train_grpo.py — GRPO-based training for PAPILLON privacy-preserving prompts.

Trains a causal LM (e.g. Llama-3.2-1B/3B-Instruct) to generate privacy-aware
prompts via Group Relative Policy Optimization (GRPO).  The training signal is
a combination of two cheap heuristic rewards (no PII leakage, prompt quality)
and, optionally, the full PAPILLON LLM-judge metric that mirrors the score used
in run_dspy_optimization_llama.py.

DSPy's ChatAdapter is used to format every training prompt identically to how
PAPILLON calls the model at inference time, so the model sees the same chat
template structure during both training and evaluation.

Additional requirements (not in environment.yml):
    pip install trl>=0.12 peft accelerate

Usage:
    # Minimal run (heuristic rewards only)
    python train_grpo.py \\
        --model_name meta-llama/Llama-3.2-1B-Instruct \\
        --data_file ../pupa/pupa_data.csv \\
        --output_dir ./grpo_output

    # With LoRA + LLM-judge reward
    python train_grpo.py \\
        --model_name meta-llama/Llama-3.2-3B-Instruct \\
        --data_file ../pupa/pupa_data.csv \\
        --output_dir ./grpo_output_3b \\
        --use_lora --use_4bit \\
        --use_llm_judge --untrusted_port 8001

    # With vLLM rollout server (TRL >= 0.14)
    python train_grpo.py \\
        --model_name meta-llama/Llama-3.2-1B-Instruct \\
        --data_file ../pupa/pupa_data.csv \\
        --output_dir ./grpo_output \\
        --use_vllm --vllm_server_port 8000
"""

import argparse
import logging
import os
import re
import sys
from typing import Optional

import pandas as pd
import torch
import dspy
from dspy.adapters import ChatAdapter
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Allow running from both papillon/ and the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_llama_dspy import PAPILLON  # noqa: E402
from llm_judge import LLMJudge  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("DSPY_CACHEDIR", os.path.join(os.getcwd(), "cache"))

adapter = ChatAdapter()


# ── DSPy / ChatAdapter helpers ────────────────────────────────────────────────

def extract_created_prompt(completion: str):
    outputs = adapter.parse(opt_papillon.prompt_creater.signature, completion)
    assert type(outputs) == dict
    return outputs.get("createdPrompt", "").strip()

def extract_final_output(completion: str):
    outputs = adapter.parse(opt_papillon.info_aggregator.signature, completion)
    assert type(outputs) == dict
    return outputs.get("finalOutput", "").strip()


def format_chat_messages(user_query: str) -> list[dict]:
    formatted_msgs = adapter.format()

# ── Data loading ───────────────────────────────────────────────────────────────

def load_pupa_split(
    data_file: str,
    split: str = "train",
    max_examples: Optional[int] = None,
) -> list[dict]:
    """
    Load a PUPA-format CSV and return examples for the requested split.
    Split boundaries mirror run_dspy_optimization_llama.py:
        train  → rows 0–149
        val    → rows 150–299
        test   → rows 300+

    Required CSV columns: user_query, target_response, pii_units
    """
    df = pd.read_csv(data_file, index_col=False)

    # Drop rows with missing / empty PII annotations
    valid = df["pii_units"].apply(
        lambda x: isinstance(x, str) and len(x.strip()) > 0
    )
    df = df[valid].reset_index(drop=True)

    if split == "train":
        df = df.iloc[:150]
    elif split == "val":
        df = df.iloc[150:300]
    else:
        df = df.iloc[300:]

    if max_examples is not None:
        df = df.iloc[:max_examples]

    return [
        {
            "user_query": str(row["user_query"]),
            "pii_str": str(row["pii_units"]),
            "target_response": str(row.get("target_response", "")),
        }
        for _, row in df.iterrows()
    ]


def make_hf_dataset(
    examples: list[dict],
    tokenizer: AutoTokenizer,
) -> Dataset:
    """
    Build a HuggingFace Dataset for GRPOTrainer.

    The "prompt" column holds the fully-formatted chat string (after applying
    the tokenizer's chat template), ready for the model to complete.
    Extra columns (user_query, pii_str, target_response) are forwarded as
    **kwargs to each reward function by GRPOTrainer.
    """
    records = []
    for ex in examples:
        messages = format_chat_messages(ex["user_query"])
        # Apply chat template to obtain a plain string prompt; this is the
        # same transformation the model saw during supervised fine-tuning.
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        records.append(
            {
                "prompt": prompt_str,
                "user_query": ex["user_query"],
                "pii_str": ex["pii_str"],
                "target_response": ex["target_response"],
            }
        )
    return Dataset.from_list(records)


# ── Reward functions ───────────────────────────────────────────────────────────
#
# GRPOTrainer calls every reward function with:
#   completions : list[str]  — one generated completion per rollout per prompt
#   **kwargs                 — one list per extra dataset column
#
# Each function must return list[float] of the same length as `completions`.


def reward_no_leakage(
    completions: list[str],
    pii_str: list[str],
    **kwargs,
) -> list[float]:
    """
    Privacy reward: +1.0 if the generated prompt leaks none of the PII tokens
    (case-insensitive substring match), 0.0 if any token leaks, 0.5 if PII
    annotations are unavailable (neutral).

    This is a fast approximation of LLMJudge.JudgeLeakage that avoids LLM
    calls during training rollouts.
    """
    rewards = []
    for comp, pii in zip(completions, pii_str):
        if not isinstance(pii, str) or not pii.strip():
            rewards.append(0.5)
            continue
        generated = extract_created_prompt(comp).lower()
        pii_pieces = [p.strip().lower() for p in pii.split("||") if p.strip()]
        leaked = any(p in generated for p in pii_pieces)
        rewards.append(0.0 if leaked else 1.0)
    return rewards


def make_cosine_similarity_reward(
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Factory returning a reward function that scores the generated prompt by its
    cosine similarity to the original user query.

    High similarity → the privacy-aware prompt preserves the task intent.
    Low  similarity → the prompt has drifted away from what the user wanted.

    The encoder runs on CPU so it doesn't contend with the training model on GPU.
    Requires: pip install sentence-transformers

    Reward ∈ [0, 1]  (cosine similarity linearly scaled from [−1, 1]).
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence encoder: %s (CPU)", encoder_name)
    encoder = SentenceTransformer(encoder_name, device="cpu")

    def cosine_similarity_reward(
        completions: list[str],
        user_query: list[str],
        **kwargs,
    ) -> list[float]:
        generated = [extract_created_prompt(c) for c in completions]
        queries = list(user_query)

        # encode() returns normalised embeddings when normalize_embeddings=True,
        # so their dot product equals cosine similarity
        gen_emb = encoder.encode(generated, convert_to_tensor=True, normalize_embeddings=True)
        q_emb   = encoder.encode(queries,   convert_to_tensor=True, normalize_embeddings=True)

        sims    = (gen_emb * q_emb).sum(dim=-1)   # (B,)  ∈ [−1, 1]
        rewards = ((sims + 1.0) / 2.0).tolist()   # rescale to [0, 1]
        return rewards

    cosine_similarity_reward.__name__ = "cosine_similarity_reward"
    return cosine_similarity_reward


def make_llm_judge_reward(judge_lm: dspy.LM, untrusted_lm: dspy.LM):
    """
    Factory returning a reward function that runs the full PAPILLON evaluation
    pipeline (LLM-as-judge for quality + leakage + prompt validity).

    Reward = (quality − leakage/num_pii + prompt_quality) / 2  ∈ [−1, 1]

    This mirrors the metric() in run_dspy_optimization_llama.py.  It is
    expensive (one untrusted-LM call + three judge calls per completion) so
    use it only when the cheap heuristics are insufficient.
    """
    judge = LLMJudge()

    def llm_judge_reward(
        completions: list[str],
        user_query: list[str],
        pii_str: list[str],
        target_response: list[str],
        **kwargs,
    ) -> list[float]:
        rewards = []
        with dspy.context(lm=judge_lm):
            for comp, query, pii, target in zip(
                completions, user_query, pii_str, target_response
            ):
                generated = extract_created_prompt(comp).strip()
                if not generated:
                    rewards.append(-1.0)
                    continue
                try:
                    # Send the privacy-aware prompt to the untrusted LM
                    model_resp = untrusted_lm(generated)[0]
                    scores = judge(
                        user_query=query,
                        new_resp=model_resp,
                        og_resp=target,
                        updated_query=generated,
                        pii_str=pii,
                    )
                    num_pii = (
                        max(1, len(set(pii.split("||"))))
                        if isinstance(pii, str)
                        else 1
                    )
                    reward = (
                        scores.quality - scores.leakage / num_pii + scores.prompt
                    ) / 2.0
                    rewards.append(float(reward))
                except Exception as exc:
                    logger.warning("LLM judge failed: %s", exc)
                    rewards.append(0.0)
        return rewards

    llm_judge_reward.__name__ = "llm_judge_reward"
    return llm_judge_reward


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO training for PAPILLON privacy-preserving prompt generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("PAPILLON-specific")
    g.add_argument("--opt_file_path", type=str)

    # — Model —
    g = p.add_argument_group("Model")
    g.add_argument(
        "--model_name",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID to train.",
    )
    g.add_argument(
        "--use_lora",
        action="store_true",
        help="Apply LoRA adapters (requires peft).",
    )
    g.add_argument("--lora_rank", type=int, default=16)
    g.add_argument("--lora_alpha", type=int, default=32)
    g.add_argument(
        "--use_4bit",
        action="store_true",
        help="Load base model in 4-bit (requires bitsandbytes).",
    )

    # — Data —
    g = p.add_argument_group("Data")
    g.add_argument(
        "--data_file",
        required=True,
        help="PUPA-format CSV with columns: user_query, target_response, pii_units.",
    )
    g.add_argument(
        "--max_train_examples",
        type=int,
        default=None,
        help="Cap the training set (useful for debugging).",
    )

    # — Reward —
    g = p.add_argument_group("Reward")
    g.add_argument(
        "--similarity_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model for the cosine-similarity quality reward.",
    )
    g.add_argument(
        "--use_llm_judge",
        action="store_true",
        help="Enable the full LLM-judge reward (requires OpenAI API key).",
    )
    g.add_argument(
        "--judge_model",
        default="gpt-4o-mini",
        help="OpenAI model used as LLM judge.",
    )
    g.add_argument(
        "--untrusted_port",
        type=int,
        default=None,
        help=(
            "Port of a local vLLM/sglang server used as the untrusted LM "
            "inside the LLM-judge reward. Falls back to the judge LM if omitted."
        ),
    )

    # — Training —
    g = p.add_argument_group("Training")
    g.add_argument("--output_dir", default="./grpo_papillon_output")
    g.add_argument("--num_epochs", type=int, default=1)
    g.add_argument("--per_device_batch_size", type=int, default=2)
    g.add_argument("--grad_accum_steps", type=int, default=4)
    g.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="GRPO group size G: rollout completions sampled per training prompt.",
    )
    g.add_argument("--max_new_tokens", type=int, default=512)
    g.add_argument("--max_prompt_length", type=int, default=1024)
    g.add_argument("--learning_rate", type=float, default=1e-6)
    g.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="KL-penalty coefficient β in the GRPO objective.",
    )
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--top_p", type=float, default=0.95)
    g.add_argument("--logging_steps", type=int, default=10)
    g.add_argument("--save_steps", type=int, default=50)

    # — vLLM (optional fast rollouts, TRL >= 0.14) —
    g = p.add_argument_group("vLLM rollouts (TRL >= 0.14)")
    g.add_argument(
        "--use_vllm",
        action="store_true",
        help="Offload rollout generation to a running vLLM server.",
    )
    g.add_argument("--vllm_server_host", default="0.0.0.0")
    g.add_argument("--vllm_server_port", type=int, default=8000)

    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    logger.info("Loading tokeniser: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── DSPy — needed only for ChatAdapter prompt formatting ──────────────────
    # We never call a real LM through DSPy here; a dummy LM is sufficient.
    _dummy_lm = dspy.LM("openai/gpt-4o-mini", api_key="dummy", cache=False)
    dspy.configure(lm=_dummy_lm)

    opt_papillon = PAPILLON(_dummy_lm)
    opt_papillon.load(args.opt_file_path)

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info("Loading training data from: %s", args.data_file)
    train_examples = load_pupa_split(
        args.data_file,
        split="train",
        max_examples=args.max_train_examples,
    )
    train_dataset = make_hf_dataset(train_examples, tokenizer)
    logger.info("Training examples: %d", len(train_dataset))

    # ── Reward functions ──────────────────────────────────────────────────────
    cosine_reward = make_cosine_similarity_reward(args.similarity_model)
    reward_fns = [reward_no_leakage, cosine_reward]

    if args.use_llm_judge:
        judge_lm = dspy.LM(f"openai/{args.judge_model}", max_tokens=1000)
        if args.untrusted_port:
            untrusted_lm = dspy.LM(
                "openai/default",
                api_base=f"http://127.0.0.1:{args.untrusted_port}/v1",
                api_key="",
                max_tokens=2000,
            )
        else:
            untrusted_lm = judge_lm
            logger.warning(
                "--untrusted_port not set; using the judge LM as the untrusted LM."
            )
        reward_fns.append(make_llm_judge_reward(judge_lm, untrusted_lm))
        logger.info("LLM-judge reward enabled (judge=%s).", args.judge_model)

    # ── LoRA (optional) ───────────────────────────────────────────────────────
    peft_config = None
    if args.use_lora:
        from peft import LoraConfig, TaskType  # noqa: PLC0415

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        logger.info("LoRA enabled (r=%d, α=%d).", args.lora_rank, args.lora_alpha)

    # ── Model kwargs (quantisation) ───────────────────────────────────────────
    model_kwargs: dict = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32
    }
    if args.use_4bit:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("4-bit quantisation enabled.")

    # ── GRPO config ───────────────────────────────────────────────────────────
    grpo_kwargs: dict = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        # GRPO-specific
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
        logger.info(
            "vLLM generation server: %s:%d",
            args.vllm_server_host,
            args.vllm_server_port,
        )

    grpo_config = GRPOConfig(**grpo_kwargs)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=reward_fns,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        model_init_kwargs=model_kwargs,
    )

    logger.info("Starting GRPO training…")
    trainer.train()

    logger.info("Saving model to: %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
