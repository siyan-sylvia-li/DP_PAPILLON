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
    model.eval()

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
        rdd_scores.append(diag_val - np.mean(off_diag_vals))

    ifs = float(np.mean(rdd_scores))
    return ifs, rdd_scores


# ---------------------------------------------------------------------------
# PAPILLON inference
# ---------------------------------------------------------------------------

def run_papillon(prompt: str, pipeline, tokenizer, priv_prompt) -> tuple[str, str]:
    """
    Run the PAPILLON prompt creator on a single user query.

    FIX 3: pipeline is now passed in as an argument rather than being
    re-instantiated on every call.

    Returns:
        prompt_str: The formatted prompt string fed to the model.
        completion_str: The model's completion string.
    """
    inputs     = dict(userQuery=prompt)
    adapter    = ChatAdapter()
    prompt_msgs = adapter.format(
        priv_prompt.prompt_creater.named_parameters()[0][1].signature,
        demos=[],
        inputs=inputs,
    )

    outputs = pipeline(prompt_msgs, max_new_tokens=1000)
    comp    = [outputs[0]["generated_text"][-1]]

    prompt_str     = tokenizer.decode(tokenizer.apply_chat_template(prompt_msgs))
    completion_str = tokenizer.decode(tokenizer.apply_chat_template(comp))
    return prompt_str, completion_str


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------

def completion_via_gliner(pipeline, tokenizer, priv_prompt):
    gliner_replace_csv  = pandas.read_csv("../pupa/PUPA_TNB_ENG_replace.csv")
    original_queries    = gliner_replace_csv["user_query"].tolist()
    replaced_queries    = gliner_replace_csv["gliner_replace"].tolist()

    all_prompts, all_completions          = [], []
    decoded_prompts, decoded_completions  = [], []

    for i in range(len(original_queries)):
        orig  = original_queries[i]
        repls = replaced_queries[i].split("|<SEP>|")

        curr_prompts, curr_completions              = [], []
        curr_decoded_prompts, curr_decoded_completions = [], []

        for r in [orig] + repls:
            prompt_str, comp_str = run_papillon(r, pipeline, tokenizer, priv_prompt)
            curr_prompts.append(prompt_str)
            curr_completions.append(comp_str)
            curr_decoded_prompts.append(r)
            curr_decoded_completions.append(comp_str)

        all_prompts.append(curr_prompts)
        all_completions.append(curr_completions)
        decoded_prompts.append(curr_decoded_prompts)
        decoded_completions.append(curr_decoded_completions)

    final_prompt_completions = [
        {
            "index": i,
            "prompt_variants": p,
            "completion_variants": c,
        }
        for i, (p, c) in enumerate(zip(decoded_prompts, decoded_completions))
    ]
    return all_prompts, all_completions, final_prompt_completions


def completion_via_templates(pipeline, tokenizer, priv_prompt):
    template = (
        "My name is {entity}. I am applying to a research internship at "
        "Microsoft Research. Please draft me a cover letter."
    )
    entities = [
        "Sharon Johnson",
        "Siyan Li",
        "Sylvia Li",
        "Michael Samson",
        "Colin Franks",
        "David Morales",
        "Cindy Shen",
    ]

    all_prompts, all_completions = [], []
    for e in entities:
        prompt_str, comp_str = run_papillon(
            template.format(entity=e), pipeline, tokenizer, priv_prompt
        )
        all_prompts.append(prompt_str)
        all_completions.append(comp_str)

    return [all_prompts], [all_completions]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # FIX 3: Build the generation pipeline once, outside run_papillon.
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    prompt_file = parse_model_prompt(model_name)
    openai_lm   = dspy.LM(model="gpt-4o-mini", max_tokens=4000)
    priv_prompt = PAPILLON(openai_lm)

    USE_TEMPLATES = False

    if USE_TEMPLATES:
        all_prompts, all_completions = completion_via_templates(
            pipeline, tokenizer, priv_prompt
        )
        final_prompt_completions = []
    else:
        all_prompts, all_completions, final_prompt_completions = completion_via_gliner(
            pipeline, tokenizer, priv_prompt
        )
        json.dump(
            final_prompt_completions,
            open("prompts_completions.json", "w+"),
        )

    # -----------------------------------------------------------------------
    # Compute IFS across all prompt groups
    # -----------------------------------------------------------------------
    ifs_scores      = []   # one per prompt group (example)
    all_rdd_scores  = []   # flattened per-variant RDD values

    for l in range(len(all_completions)):
        n = len(all_prompts[l])

        # FIX 5: Warn and skip non-square groups rather than silently misbehaving.
        if len(all_completions[l]) != n:
            print(
                f"[WARNING] Group {l}: {n} prompts but "
                f"{len(all_completions[l])} completions — skipping."
            )
            continue

        # Build both raw and length-normalized matrices in a single pass.
        raw_matrix  = np.zeros((n, n))
        norm_matrix = np.zeros((n, n))

        for i, p in enumerate(all_prompts[l]):
            for j, c in enumerate(all_completions[l]):
                total_lp, norm_lp, per_tok_lp, tok_ids = logprob_completion_causal(
                    model, tokenizer, p, c
                )
                raw_matrix[i][j]  = total_lp
                norm_matrix[i][j] = norm_lp  # FIX 2: use normalized value

        # --- IFS on length-normalized matrix (primary metric) ---
        ifs, rdd_scores = compute_rdd_and_ifs(norm_matrix)
        ifs_scores.append(ifs)
        all_rdd_scores.extend(rdd_scores)

        # --- Optional: also compute on raw matrix as a sanity check ---
        ifs_raw, _ = compute_rdd_and_ifs(raw_matrix)

        # --- Heatmap (normalized) ---
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            norm_matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar_kws={"label": "Avg log-prob per token"},
        )
        plt.title(f"Length-Normalized Log-Probability Heatmap (group {l})\nIFS = {ifs:.3f}")
        plt.xlabel("Completion index")
        plt.ylabel("Prompt index")
        plt.tight_layout()
        plt.savefig(f"logprob_heatmap_{l}.png")
        plt.close()

        print(f"Group {l:3d} | IFS (normalized) = {ifs:+.4f} | IFS (raw) = {ifs_raw:+.4f}")

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