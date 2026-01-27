import pandas
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate_papillon import parse_model_prompt
import dspy
from run_llama_dspy import PAPILLON
from dspy.adapters import ChatAdapter
import transformers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

torch.cuda.empty_cache()

@torch.no_grad()
def logprob_completion_causal(
    model,
    tokenizer,
    prompt: list[int],
    completion: list[int],
    *,
    add_special_tokens_to_prompt: bool = True,
):
    """
    Returns:
      total_logprob: float (sum of log p(token_i | prompt, previous completion tokens))
      token_logprobs: list[float] (per completion token)
      completion_token_ids: list[int]
    """
    model.eval()
    
    prompt = tokenizer.decode(prompt)
    completion = tokenizer.decode(completion)
    # print(prompt, completion)

    # Tokenize prompt and completion separately to precisely locate the boundary.
    prompt_enc = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=add_special_tokens_to_prompt,
    )
    comp_enc = tokenizer(
        completion,
        return_tensors="pt",
        add_special_tokens=False,   # IMPORTANT: don't add BOS/EOS to the completion
    )

    prompt_ids = prompt_enc["input_ids"]
    prompt_mask = prompt_enc["attention_mask"]
    comp_ids = comp_enc["input_ids"]
    comp_mask = comp_enc["attention_mask"]

    # Concatenate into one sequence: [prompt][completion]
    input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, comp_mask], dim=1)

    # Move to the model device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    # out.logits: (batch=1, seq_len, vocab)
    logits = out.logits

    # Convert logits -> log-probabilities for next token prediction
    logprobs = F.log_softmax(logits, dim=-1)

    # Shift so that positions align as: logprobs_at_pos[t] predicts token_id[t+1]
    shift_logprobs = logprobs[:, :-1, :]      # (1, seq_len-1, vocab)
    shift_labels   = input_ids[:, 1:]         # (1, seq_len-1)

    prompt_len = prompt_ids.shape[1]
    comp_len   = comp_ids.shape[1]

    # Completion tokens live in shift_labels at indices:
    # (prompt_len-1) ... (prompt_len-1 + comp_len - 1)
    start = prompt_len - 1
    end = start + comp_len
    comp_positions = torch.arange(start, end, device=device)

    # Gather log p(correct_token) for each completion token
    # shape after gather: (1, comp_len)
    token_logprobs = shift_logprobs[0, comp_positions, shift_labels[0, comp_positions]]

    total_logprob = token_logprobs.sum().item()

    return total_logprob, token_logprobs.detach().cpu().tolist(), comp_ids[0].cpu().tolist()


def run_papillon(prompt):
    inputs = dict(userQuery=prompt)
    adapter = ChatAdapter()
    prompt_msgs = adapter.format(priv_prompt.prompt_creater.named_parameters()[0][1].signature, demos=[], inputs=inputs)
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    outputs = pipeline(
        prompt_msgs,
        max_new_tokens=1000,
    )
    comp = [outputs[0]["generated_text"][-1]]
    return tokenizer.apply_chat_template(prompt_msgs), tokenizer.apply_chat_template(comp)

def completion_via_gliner():
    gliner_replace_csv = pandas.read_csv("../pupa/PUPA_TNB_ENG_replace.csv")
    gliner_replace_csv = gliner_replace_csv.sample(n=20)
    original_queries = gliner_replace_csv["user_query"].tolist()
    replaced_queries = gliner_replace_csv["gliner_replace"].tolist()
    all_prompts = []
    all_completions = []
    decoded_prompts, decoded_completions = [], []
    for i in range(len(original_queries)):
        orig = original_queries[i]
        repls = replaced_queries[i].split("|<SEP>|")
        curr_prompts, curr_completions = [], []
        curr_decoded_prompts, curr_decoded_completions = [], []
        for r in [orig] + repls:
            prompt, comp = run_papillon(r)
            curr_prompts.append(prompt)
            curr_completions.append(comp)
            curr_decoded_prompts.append(r)
            curr_decoded_completions.append(tokenizer.decode(comp))
        all_prompts.append(curr_prompts)
        all_completions.append(curr_completions)
        decoded_prompts.append(curr_decoded_prompts)
        decoded_completions.append(curr_decoded_completions)
    return all_prompts, all_completions, decoded_prompts, decoded_completions

def completion_via_templates():
    template = "My name is {entity}. I am applying to a research internship at Microsoft Research. Please draft me a cover letter."
    entities = [
        "Sharon Johnson",
        "Siyan Li",
        "Sylvia Li",
        "Michael Samson",
        "Colin Franks",
        "David Morales",
        "Cindy Shen"
    ]
    
    all_prompts = []
    all_completions = []
    for e in entities:
        prompt = template.format(entity=e)
        prompt, comp = run_papillon(prompt)
        all_completions.append(comp)
        all_prompts.append(prompt)
        
    return [all_prompts], [all_completions]

# --- Example usage ---
if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"  # replace with your model
    # model_name = "meta-llama/Llama-3.2-1B-Instruct"  # replace with your model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config
    )
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
    prompt_file = parse_model_prompt(model_name)

    openai_lm = dspy.LM(model="gpt-4o-mini", max_tokens=4000)

    priv_prompt = PAPILLON(openai_lm)
    # print(priv_prompt.prompt_creater.named_parameters()[0][1])
    # print(priv_prompt.prompt_creater.named_parameters()[0][1].signature)
    
    # priv_prompt.load(prompt_file, use_legacy_loading=True)
    # priv_prompt.load(prompt_file)

    USE_TEMPLATES = False
    
    if USE_TEMPLATES:
        all_prompts, all_completions = completion_via_templates()
    else:
        all_prompts, all_completions, decoded_prompt, decoded_completions = completion_via_gliner()

    import json
    json.dump({"prompts": decoded_prompt, "completions": decoded_completions}, open("prompts_completions.json", "w+"))
        
    
    for l in range(len(all_completions)):
        prob_matrix = np.zeros((len(all_prompts[l]), len(all_completions[l])))
        for i, p in enumerate(all_prompts[l]):
            for j, c in enumerate(all_completions[l]):
                # print(i, j)
                total_lp, per_tok_lp, tok_ids = logprob_completion_causal(model, tokenizer, p, c)
                prob_matrix[i][j] = total_lp
                # print("avg logprob per token      =", total_lp / len(per_tok_lp))

        plt.figure(figsize=(8, 6))
        sns.heatmap(prob_matrix, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": "Total log-prob"})
        plt.title("Log-Probability Heatmap")
        plt.xlabel("Completion index")
        plt.ylabel("Prompt index")
        plt.tight_layout()
        plt.savefig(f"logprob_heatmap_{l}.png")
        plt.close()
                
        
        # print("Average log prob: ", prob_matrix.mean())
        # print("Max log prob: ", prob_matrix.max())
        # print("Min log prob:", prob_matrix.min())
        

        
