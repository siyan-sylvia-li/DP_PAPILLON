import pandas
from gliner import GLiNER
import json
import tqdm
labels = ["email", "phone_number", "user_name", "first_name", "last_name", "company_name", "url", "country", "city", "county"]

# 3. Load the PII model
model = GLiNER.from_pretrained("nvidia/gliner-pii")


def predict_gliner_entities(text: str):
    entities = model.predict_entities(text, labels, threshold=0.85)
    piis = []
    pii_types = []
    for e in entities:
        if e["score"] > 0.9:
            piis.append(e["text"].lower())
            pii_types.append(e["label"])
    if len(piis) == 0:
        return None
    return "||".join(list(set(piis))), pii_types


if __name__ == "__main__":
    processed_data = json.load(open("/local-storage/interaction/siyanli/DP_PAPILLON/pupa_nemotron/final_data_filtered_by_topic.json"))
    # processed_data = [x for x in processed_data if x["categorys_analysis"] in ["4. Job, visa, and other applications", "2. Copy-and-pasted emails or messaging transcripts", "10. Medical and healthcare information"]]
    new_processed_data = []

    for data in tqdm.tqdm(processed_data):
        for c in ["4. Job, visa, and other applications", "2. Copy-and-pasted emails or messaging transcripts", "10. Medical and healthcare information"]:
            if data["categorys_analysis"].startswith(c):
                data["categorys_analysis"] = c
                user_query = data["user_query"]
                pii_gliner, pii_types = predict_gliner_entities(user_query)
                data["pii_gliner"] = pii_gliner
                data["pii_gliner_types"] = pii_types
                if pii_gliner is not None:
                    new_processed_data.append(data)
                continue
    json.dump(new_processed_data, open("/local-storage/interaction/siyanli/DP_PAPILLON/pupa_nemotron/final_data_with_gliner.json", "w+"), indent=2)

    # eng_queries = pandas.read_csv("../pupa/PUPA_TNB_ENG.csv")
    # eng_queries = eng_queries.sample(n=20)
    # eng_queries["pii_gliner"] = eng_queries["user_query"].map(predict_gliner_entities)
    
    # eng_queries = eng_queries.loc[pandas.notna(eng_queries["pii_gliner"])]
    # eng_queries.to_csv("../pupa/PUPA_TNB_ENG_GLiNER.csv")
    