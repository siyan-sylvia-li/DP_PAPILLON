import random
import pandas
from gliner import GLiNER
from datasets import load_dataset


labels = ["first_name"]

# 3. Load the PII model
model = GLiNER.from_pretrained("nvidia/gliner-pii")

male_name_list = [
    "Liam", "Noah", "Oliver", "Elijah", "James", "William", "Benjamin", "Lucas", "Henry", "Alexander",
    "Mason", "Michael", "Ethan", "Daniel", "Jacob", "Logan", "Jackson", "Levi", "Sebastian", "Mateo", "John"
]
female_name_list = [
    "Olivia", "Emma", "Ava", "Sophia", "Isabella", "Mia", "Charlotte", "Amelia", "Harper", "Evelyn",
    "Abigail", "Ella", "Scarlett", "Grace", "Chloe", "Victoria", "Riley", "Aria", "Lily", "Aurora", "Zoey"
]


def predict_gliner_entities(text: str):
    entities = model.predict_entities(text, labels, threshold=0.85)
    piis = []
    for e in entities:
        if e["score"] > 0.9:
            piis.append(e["text"])
    if len(piis) == 0:
        return None, None
    if len(set(piis)) > 1:
        # print(f"Multiple PIIs found: {piis} in text: {text}")
        return None, None
    if "she" in text.lower() or "her" in text.lower() or "hers" in text.lower():
        return piis[0], random.choices(female_name_list, k=5)
    elif "he" in text.lower() or "him" in text.lower() or "his" in text.lower():
        return piis[0], random.choices(male_name_list, k=5)
    return None, None

if __name__ == "__main__":
    # Load gsm8k dataset
    gsm8k_dataset_train = load_dataset("openai/gsm8k", "main", split="train")
    gsm8k_dataset_test = load_dataset("openai/gsm8k", "main", split="test")

    all_train_examples = []
    all_test_examples = []

    for i, example in enumerate(gsm8k_dataset_train):
        pii, substitutes = predict_gliner_entities(example["question"])
        if pii is not None:
            all_train_examples.append({
                "original_question": example["question"],
                "pii": pii,
                "substitutes": "||".join(substitutes),
                "original_answer": example["answer"]
            })
        if i > 2000:
            break  # Limit to first 2000 examples for faster processing 
    for i, example in enumerate(gsm8k_dataset_test):
        pii, substitutes = predict_gliner_entities(example["question"])
        if pii is not None:
            all_test_examples.append({
                "original_question": example["question"],
                "pii": pii,
                "substitutes": "||".join(substitutes),
                "original_answer": example["answer"]
            })
        if i > 500:
            break  # Limit to first 500 examples for faster processing
    
    train_df = pandas.DataFrame(all_train_examples)
    test_df = pandas.DataFrame(all_test_examples)
    train_df.to_csv("gsm8k_train_gliner_pii.csv", index=False)
    test_df.to_csv("gsm8k_test_gliner_pii.csv", index=False)
    print(f"Total train examples with PII: {len(train_df)}")
    print(f"Total test examples with PII: {len(test_df)}")
        