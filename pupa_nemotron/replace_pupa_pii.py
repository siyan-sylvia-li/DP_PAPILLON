import pandas
from gliner import GLiNER
import json
import random
import dspy

import dotenv
dotenv.load_dotenv("../.env")

class QAReplacedQuery(dspy.Signature):
    """Given a user query and a modified version by replacing one of the names in the original query, determine whether the names used in the modified query are consistent with the rest of the query context. Respond directly with yes or no. If there is a name in the modified query that is inconsistent with how it is referred to (e.g. inconsistent with pronouns), respond with no. If the original query references a historical figure or well known person, respond with no."""
    original_user_query = dspy.InputField(desc="The original user query before any modifications.")
    modified_user_query = dspy.InputField(desc="The modified user query to be evaluated.")
    judgment = dspy.OutputField(desc="The judgment of whether the names in the query are consistent with the context. Respond with 'yes' or 'no'.")

class DetermineQueryLanguage(dspy.Signature):
    """Given a user query, determine whether the query is written COMPLETELY in English. Respond directly with yes or no."""
    user_query = dspy.InputField(desc="The user query to be evaluated.")
    language_judgment = dspy.OutputField(desc="Whether the query is COMPLETELY in English. Respond with yes or no.")


labels = ["email", "phone_number", "user_name", "first_name", "last_name", "company_name", "url", "country", "city", "county"]

NEMOTRON_PII_LEN = 99900

# 3. Load the PII model
model = GLiNER.from_pretrained("nvidia/gliner-pii")

def swap_gliner_entities(text: str):
    if language_detector(user_query=text).language_judgment.lower().startswith("no"):
        return None
    entities = model.predict_entities(text, labels, threshold=0.85)
    entities = [e for e in entities if e["score"] > 0.9]
    all_labels = [e["label"] for e in entities]
    all_swaps = []
    original_text = text
    if "first_name" in all_labels and "last_name" in all_labels:
        first_name = [e["text"] for e in entities if e["label"] == "first_name"][0]
        last_name = [e["text"] for e in entities if e["label"] == "last_name"][0]
        for _ in range(5):
            found = False
            retry_attempts = 0
            while not found and retry_attempts < 10:
                random_label = str(random.randint(0, NEMOTRON_PII_LEN))
                if random_label in nemotron_labels["first_name"] and random_label in nemotron_labels["last_name"]:
                    found = True
                    text = original_text.replace(first_name, nemotron_labels["first_name"][random_label][0]).replace(last_name, nemotron_labels["last_name"][random_label][0])
                    if qa_judge(original_user_query=original_text, modified_user_query=text).judgment.lower().startswith("no"):
                        found = False
                        retry_attempts += 1
            if found:
                all_swaps.append(text)
    if all_swaps:
        return "|<SEP>|".join(all_swaps)
    return None
            
        

if __name__ == "__main__":
    nemotron_labels = json.load(open("nemotron_tags.json"))
    eng_queries = pandas.read_csv("../pupa/PUPA_TNB_ENG.csv")

    llm_5_nano = dspy.LM(model="openai/gpt-5-nano")
    dspy.configure(lm=llm_5_nano)
    language_detector = dspy.Predict(DetermineQueryLanguage)
    qa_judge = dspy.Predict(QAReplacedQuery)
    
    
    eng_queries = eng_queries.sample(n=200)
    eng_queries["gliner_replace"] = eng_queries["user_query"].map(swap_gliner_entities)
    
    eng_queries = eng_queries.loc[pandas.notna(eng_queries["gliner_replace"])]
    eng_queries.to_csv("../pupa/PUPA_TNB_ENG_replace.csv")