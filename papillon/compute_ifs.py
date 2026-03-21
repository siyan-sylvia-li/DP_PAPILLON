import pandas
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from evaluate_papillon import parse_model_prompt
import dspy
from run_llama_dspy import PAPILLON
from dspy.adapters import ChatAdapter
import transformers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from argparse import ArgumentParser
import tqdm

torch.cuda.empty_cache()

import random
random.seed(42)


# ---------------------------------------------------------------------------
# Core log-probability computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def logprob_completion_causal(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    *,
    add_special_tokens_to_prompt: bool = True,
):
    """
    Compute the log-probability of a completion given a prompt under a causal LM.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: The prompt string.
        completion: The completion string whose likelihood we measure.
        add_special_tokens_to_prompt: Whether to prepend BOS etc. to the prompt.

    Returns:
        total_logprob: float  — sum of log p(token_i | prompt, prev completion tokens)
        normalized_logprob: float — total_logprob / number of completion tokens
        token_logprobs: list[float] — per-completion-token log-probs
        completion_token_ids: list[int]
    """

    # FIX 1: Accept strings directly — no decode/re-encode round trip.
    prompt_enc = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=add_special_tokens_to_prompt,
    )
    comp_enc = tokenizer(
        completion,
        return_tensors="pt",
        add_special_tokens=False,  # IMPORTANT: don't add BOS/EOS to the completion
    )

    prompt_ids  = prompt_enc["input_ids"]
    prompt_mask = prompt_enc["attention_mask"]
    comp_ids    = comp_enc["input_ids"]
    comp_mask   = comp_enc["attention_mask"]

    # Concatenate into one sequence: [prompt][completion]
    input_ids      = torch.cat([prompt_ids, comp_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, comp_mask], dim=1)

    device = next(model.parameters()).device
    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    out    = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # (1, seq_len, vocab)

    # log-softmax over vocab → log-probs for next-token prediction
    logprobs = F.log_softmax(logits, dim=-1)

    # Shift: logprobs[:, t, :] predicts token at position t+1
    shift_logprobs = logprobs[:, :-1, :]   # (1, seq_len-1, vocab)
    shift_labels   = input_ids[:, 1:]      # (1, seq_len-1)

    prompt_len = prompt_ids.shape[1]
    comp_len   = comp_ids.shape[1]

    # Completion token positions in the shifted sequence
    start = prompt_len - 1
    end   = start + comp_len
    comp_positions = torch.arange(start, end, device=device)

    token_logprobs = shift_logprobs[0, comp_positions, shift_labels[0, comp_positions]]

    total_logprob      = token_logprobs.sum().item()
    # FIX 2: Return length-normalized log-prob alongside the raw total.
    normalized_logprob = total_logprob / comp_len

    return total_logprob, normalized_logprob, token_logprobs.detach().cpu().tolist(), comp_ids[0].cpu().tolist()


# ---------------------------------------------------------------------------
# IFS metric (replaces compute_deviation)
# ---------------------------------------------------------------------------

def compute_rdd_and_ifs(logprob_matrix: np.ndarray):
    """
    Compute per-variant Relative Diagonal Dominance (RDD) and the aggregate
    Invariance Failure Score (IFS) for a single prompt group.

    RDD(i) = log p(c_i | p_i) - mean_{j != i} log p(c_j | p_i)

    IFS = mean_i RDD(i)

    A positive IFS means the model systematically prefers its own completion
    given its own prompt variant — i.e., invariance failure.
    IFS = 0 would indicate perfect invariance.

    Args:
        logprob_matrix: np.ndarray of shape (N, N).
                        Entry [i, j] = (length-normalized) log p(c_j | p_i).

    Returns:
        ifs: float — aggregate invariance failure score for this group.
        rdd_scores: list[float] — per-variant RDD values.
    """
    assert logprob_matrix.ndim == 2, "Expected a 2-D matrix."
    assert logprob_matrix.shape[0] == logprob_matrix.shape[1], (
        f"Expected square matrix, got shape {logprob_matrix.shape}. "
        "Number of prompt variants and completion variants must match."
    )

    n = logprob_matrix.shape[0]
    rdd_scores = []
    for i in range(n):
        diag_val      = logprob_matrix[i, i]
        off_diag_vals = np.concatenate(
            [logprob_matrix[i, :i], logprob_matrix[i, i + 1:]]
        )
        rdd_scores.append(abs(diag_val - np.mean(off_diag_vals)))

    ifs = float(np.mean(rdd_scores))
    return ifs, rdd_scores


def run_prompts(prompts: list[str], pipeline, tokenizer, prompt_formatter, bsize=30) -> tuple[list[str], list[str]]:
    """
    Run a set of prompts on the pipeline, produce the final list of prompt strs and completion strs.
    """
    all_prompts, all_completions = [], []
    for i in range(0, len(prompts), bsize):
        full_prompts = [prompt_formatter(p) for p in prompts[i: i + bsize]]
        outputs = pipeline(full_prompts, max_new_tokens=1000, do_sample=False)
        comps = [outputs[j][0]["generated_text"][-1] for j in range(len(outputs))]
        prompt_strs = [tokenizer.decode(tokenizer.apply_chat_template(fp)) for fp in full_prompts]
        completion_strs = [tokenizer.decode(tokenizer.apply_chat_template([c])) for c in comps]
        all_prompts.extend(prompt_strs)
        all_completions.extend(completion_strs)
    return all_prompts, all_completions


def run_prompts_vllm(prompts: list[str], llm, tokenizer, prompt_formatter, max_new_tokens: int = 1000) -> tuple[list[str], list[str]]:
    """
    vllm-backed generation. Passes all prompts in one call (vllm handles batching internally).
    """
    from vllm import SamplingParams
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    full_prompts = [prompt_formatter(p) for p in prompts]
    outputs = llm.chat(full_prompts, sampling_params, use_tqdm=False)
    all_prompts, all_completions = [], []
    for fp, output in zip(full_prompts, outputs):
        comp_text = output.outputs[0].text
        prompt_str = tokenizer.decode(tokenizer.apply_chat_template(fp))
        completion_str = tokenizer.decode(tokenizer.apply_chat_template([{"role": "assistant", "content": comp_text}]))
        all_prompts.append(prompt_str)
        all_completions.append(completion_str)
    return all_prompts, all_completions


def logprob_matrix_vllm(llm, tokenizer, prompts: list[str], completions: list[str], batch_size: int = 32, names: list = []) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the N×N raw and length-normalized logprob matrices using vllm.
    N² pairs are processed in chunks of `batch_size` to avoid GPU OOM.
    Reduce batch_size if you hit OOM; increase it for throughput on larger GPUs.
    """
    from vllm import SamplingParams
    n = len(prompts)
    all_ids, prompt_lens, comp_lens = [], [], []
    for i, p in enumerate(prompts):
        for j, c in enumerate(completions):
            if names:
                c = c.replace(names[j], names[i])
            p_ids = tokenizer(p, add_special_tokens=True)["input_ids"]
            c_ids = tokenizer(c, add_special_tokens=False)["input_ids"]
            all_ids.append(p_ids + c_ids)
            prompt_lens.append(len(p_ids))
            comp_lens.append(len(c_ids))

    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=1)
    raw_matrix  = np.zeros((n, n))
    norm_matrix = np.zeros((n, n))

    for start in range(0, len(all_ids), batch_size):
        end = min(start + batch_size, len(all_ids))
        chunk_inputs = [{"prompt_token_ids": ids} for ids in all_ids[start:end]]
        chunk_outputs = llm.generate(chunk_inputs, sampling_params=sampling_params)
        for rel_idx, output in enumerate(chunk_outputs):
            idx = start + rel_idx
            i, j = divmod(idx, n)
            pl, cl = prompt_lens[idx], comp_lens[idx]
            token_logprobs = [
                output.prompt_logprobs[pos][all_ids[idx][pos]].logprob
                for pos in range(pl, pl + cl)
            ]
            total = sum(token_logprobs)
            raw_matrix[i, j]  = total
            norm_matrix[i, j] = total / cl
    return raw_matrix, norm_matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--use_vllm", action="store_true", help="Use vllm for generation and logprob scoring.")
    parser.add_argument("--logprob_batch_size", type=int, default=32, help="Number of (prompt, completion) pairs per logprob scoring batch. Reduce if OOM.")
    parser.add_argument("--max_model_len", type=int, default=None, help="Override max sequence length (vllm). Required for models with large defaults like Qwen3.")
    parser.add_argument("--enforce_eager", action="store_true", help="Disable CUDA graphs and torch.compile in vllm. Slower but uses less memory during init.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model dtype. Note: gemma3 and some models require bfloat16.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_vllm:
        from vllm import LLM
        llm_kwargs = dict(
            model=args.model_name,
            dtype=args.dtype,
            enable_prefix_caching=True,
            enforce_eager=args.enforce_eager,
        )
        if args.max_model_len is not None:
            llm_kwargs["max_model_len"] = args.max_model_len
        llm = LLM(**llm_kwargs)
        model = None
        pipeline = None
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
        )
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        torch.cuda.empty_cache()
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            max_new_tokens=args.max_new_tokens,
        )
        llm = None

    if args.task == "PAPILLON":
        from task_customizer import PAPILLONCustomizer
        task_customizer = PAPILLONCustomizer()
    elif args.task == "GSM8k":
        from task_customizer import GSM8kCustomizer
        task_customizer = GSM8kCustomizer()
    elif args.task == "TruthfulQA":
        from task_customizer import TruthfulQACustomizer
        task_customizer = TruthfulQACustomizer()
    else:
        raise NotImplementedError("Not supported dataset")
    
    FNAME = "DUMMY.json"

    import os
    if not os.path.exists(FNAME):

        import datetime
        curr_dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        model_name_safe = args.model_name.replace("/", "_")
        file_name = f"{args.task}_{model_name_safe}_{curr_dt}.json"
        FNAME = file_name
        
        
        full_prompts, full_completions = [], []
        for p in tqdm.tqdm(task_customizer.data):
            if args.use_vllm:
                curr_all_prompts, curr_all_completions = run_prompts_vllm(p, llm, tokenizer, task_customizer.format_prompt, args.max_new_tokens)
            else:
                curr_all_prompts, curr_all_completions = run_prompts(p, pipeline, tokenizer, task_customizer.format_prompt)
            full_prompts.append(curr_all_prompts)
            full_completions.append(curr_all_completions)

            final_prompt_completions = [{"prompts": p, "comps": c} for p, c in zip(full_prompts, full_completions)]
            json.dump(final_prompt_completions, open(file_name, "w+"))
    
        final_prompt_completions = [{"prompts": p, "comps": c} for p, c in zip(full_prompts, full_completions)]
        json.dump(final_prompt_completions, open(file_name, "w+"))
    else:
        final_prompt_completions = json.load(open(FNAME))
        full_prompts = [x["prompts"] for x in final_prompt_completions]
        full_completions = [x["comps"] for x in final_prompt_completions]

    
    ifs_results = {}
    

    # -----------------------------------------------------------------------
    # Compute IFS across all prompt groups
    # -----------------------------------------------------------------------
    ifs_scores      = []   # one per prompt group (example)
    all_rdd_scores  = []   # flattened per-variant RDD values

    for l in tqdm.tqdm(range(len(full_completions))):
        if l > 300:
            break
        n = len(full_prompts[l])

        # FIX 5: Warn and skip non-square groups rather than silently misbehaving.
        if len(full_completions[l]) != n:
            print(
                f"[WARNING] Group {l}: {n} prompts but "
                f"{len(full_completions[l])} completions — skipping."
            )
            continue

        # Build both raw and length-normalized matrices in a single pass.
        if args.use_vllm:
            raw_matrix, norm_matrix = logprob_matrix_vllm(llm, tokenizer, full_prompts[l], full_completions[l], batch_size=args.logprob_batch_size, names=task_customizer.names[l])
        else:
            raw_matrix  = np.zeros((n, n))
            norm_matrix = np.zeros((n, n))
            for i, p in enumerate(full_prompts[l]):
                for j, c in enumerate(full_completions[l]):
                    if args.task == "GSM8k":
                        names = task_customizer.names[l]
                        c = c.replace(names[j], names[i])
                    total_lp, norm_lp, per_tok_lp, tok_ids = logprob_completion_causal(
                        model, tokenizer, p, c
                    )
                    raw_matrix[i][j]  = total_lp
                    norm_matrix[i][j] = norm_lp

        # --- IFS on length-normalized matrix (primary metric) ---
        ifs, rdd_scores = compute_rdd_and_ifs(norm_matrix)
        ifs_scores.append(ifs)
        all_rdd_scores.extend(rdd_scores)

        # --- Optional: also compute on raw matrix as a sanity check ---
        ifs_raw, _ = compute_rdd_and_ifs(raw_matrix)

        # --- Heatmap (normalized) ---
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(
        #     norm_matrix,
        #     annot=True,
        #     fmt=".2f",
        #     cmap="viridis",
        #     cbar_kws={"label": "Avg log-prob per token"},
        # )
        # plt.title(f"Length-Normalized Log-Probability Heatmap (group {l})\nIFS = {ifs:.3f}")
        # plt.xlabel("Completion index")
        # plt.ylabel("Prompt index")
        # plt.tight_layout()
        # plt.savefig(f"logprob_heatmap_{l}.png")
        # plt.close()

        # print(f"Group {l:3d} | IFS (normalized) = {ifs:+.4f} | IFS (raw) = {ifs_raw:+.4f}")
        
        ifs_results.update({
            l: {
                "ifs_normalized": ifs,
                "ifs_raw": ifs_raw,
                "rdd_scores": rdd_scores
            }
        })
        json.dump(ifs_results, open("ifs_" + FNAME + ".json", "w+"))

    # -----------------------------------------------------------------------
    # Dataset-level summary
    # -----------------------------------------------------------------------
    mean_ifs = float(np.mean(ifs_scores))
    std_ifs  = float(np.std(ifs_scores))
    pct_positive = 100.0 * np.mean(np.array(ifs_scores) > 0)

    print("\n=== Dataset-level IFS summary ===")
    print(f"  Mean IFS : {mean_ifs:+.4f}")
    print(f"  Std  IFS : {std_ifs:.4f}")
    print(f"  % groups with IFS > 0 (invariance failure) : {pct_positive:.1f}%")