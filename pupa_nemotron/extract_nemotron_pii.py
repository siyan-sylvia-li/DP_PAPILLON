from datasets import load_dataset
import json
import ast

if __name__ == "__main__":
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("nvidia/Nemotron-PII", split="train")
    # print(ds["spans"])
    tags = {}

    for i, example in enumerate(ds):
        spans = ast.literal_eval(example["spans"])
        for s in spans:
            l = s['label']
            if l not in tags:
                tags[l] = {}
            if i not in tags[l]:
                tags[l][i] = []
            tags[l][i].append(s["text"])
        if i % 50 == 0:
            json.dump(tags, open("nemotron_tags.json", "w+"))

    print(f"Found {len(tags)} unique labels")
    for label, instances in tags.items():
        print(f"Label '{label}': {len(instances)} examples")
    

