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
import tqdm
import copy

torch.cuda.empty_cache()

prompt_buffer = []
final_prompts, final_completions, final_correctness = [], [], []

@torch.no_grad()
def run_gsm8k(prompts: list[str], ogs: list[str], must_run: bool = False):
    global prompt_buffer, final_prompts, final_completions, final_correctness, pipeline
    prompt_msgs = [
        [{"role": "user", "content": prompt + "\n\nReturn only the final numeric answer."}] for prompt in prompts
    ]

    for og, pm in zip(ogs, prompt_msgs):
        prompt_buffer.append((og, pm))
    
    if len(prompt_buffer) >= 500 or must_run:
        torch.cuda.empty_cache()
        prompt_msgs = [pm for og, pm in prompt_buffer]
        final_prompts.extend(prompt_msgs)
        outputs = pipeline(
            prompt_msgs,
            max_new_tokens=3000,
        )
        comps = [outputs[i][0]["generated_text"][-1] for i in range(len(outputs))]
        ogs = [og for og, pm in prompt_buffer]
        for og, comp in zip(ogs, comps):
            final_completions.append((og, comp))
        all_last_numbers = []
        print(len(comps), len(final_completions), len(ogs))
        for comp in comps:
            # if "**Answer" not in comp["content"] or "**Final Answer" not in comp["content"]:
            if "</think" not in comp["content"]:
                all_last_numbers.append(None)
                continue
            # Split comp into words, take the last number
            comp_words = comp["content"].split()
            last_number = None
            for w in reversed(comp_words):
                try:
                    # Remove all non-numeric characters from w first
                    w = ''.join(filter(lambda x: x.isdigit() or x == '.' or x == '-', w))
                    last_number = int(w)
                    break
                except:
                    continue
            all_last_numbers.append(last_number)
        for last_num, og in zip(all_last_numbers, ogs):
            answer = int(og.split("||")[-1].strip().replace(".", "").replace(",", ""))
            final_correctness.append(int(last_num == answer))
        prompt_buffer = []



def completion_via_templates(df: pandas.DataFrame):
    for i, row in tqdm.tqdm(df.iterrows()):
        og_question = row["original_question"]
        pii = row["pii"]
        substitutes = row["substitutes"].split("||")
        new_questions = [og_question.replace(pii, sub) for sub in substitutes]
        answer_key = int(row["original_answer"].split("####")[-1].strip().replace(".", "").replace(",", ""))
        # curr_prompts, curr_completions = [], []
        run_gsm8k([og_question] + new_questions, [og_question + "||" + str(answer_key)] * (1 + len(new_questions)))
    run_gsm8k([], [], must_run=True)
    return copy.copy(final_prompts), copy.copy(final_completions), copy.copy(final_correctness)

# --- Example usage ---
if __name__ == "__main__":
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"  # replace with your model
    # model_name = "meta-llama/Llama-3.2-1B-Instruct"  # replace with your model
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    pipeline.model.to("cuda:0")
    pipeline.model.config.use_cache = True

    # from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    # quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     quantization_config=quantization_config
    # )
    # # model = AutoModelForCausalLM.from_pretrained(model_name)
    # model.to("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.empty_cache()
    
    # prompt_file = parse_model_prompt(model_name)

    # openai_lm = dspy.LM(model="gpt-4o-mini", max_tokens=4000)

    # priv_prompt = PAPILLON(openai_lm)
    # print(priv_prompt.prompt_creater.named_parameters()[0][1])
    # print(priv_prompt.prompt_creater.named_parameters()[0][1].signature)
    
    # priv_prompt.load(prompt_file, use_legacy_loading=True)
    # priv_prompt.load(prompt_file)

    train_file = pandas.read_csv("../pupa_nemotron/gsm8k_train_gliner_pii.csv")
    train_file = train_file.sample(n=100)
    test_file = pandas.read_csv("../pupa_nemotron/gsm8k_test_gliner_pii.csv")
    test_file = test_file.sample(n=100)

    all_prompts_train, all_completions_train, all_correctness_train = completion_via_templates(train_file)
    final_prompts, final_completions, final_correctness = [], [], []
    import json
    json.dump({"prompts": all_prompts_train, "completions": all_completions_train, "correctness": all_correctness_train}, open("prompts_completions_gsm8k_1b_train.json", "w+"))
    all_prompts_test, all_completions_test, all_correctness_test = completion_via_templates(test_file)
    json.dump({"prompts": all_prompts_test, "completions": all_completions_test, "correctness": all_correctness_test}, open("prompts_completions_gsm8k_1b_test.json", "w+"))

    # Print the correctness of train vs test
    train_correct = sum(all_correctness_train)/len(all_correctness_train)
    test_correct = sum(all_correctness_test)/len(all_correctness_test)
    print(f"Train accuracy: {train_correct}")
    print(f"Test accuracy: {test_correct}")
    


        
