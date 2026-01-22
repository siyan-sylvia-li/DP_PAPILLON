import pandas
from gliner import GLiNER

# 1. Define our new text
# text = "Hi support, I can't log in! My account username is 'johndoe88'. Every time I try, it says 'invalid credentials'. Please reset my password. You can reach me at (555) 123-4567 or johnd@example.com"
text = "rite an 8-10 page business plan for a new non profit org. The non profit org will address the digital divide in Underserved Populations/Communities in Seattle/King county area. Write in great detail about Executive summary, Nonprofit description, Need analysis, Products, programs, and services descriptions. The non profit will offer free tech training to qualifying individuals. Outline the goals and objectives to achieve our mission, Operational plan, Marketing plan, Impact plan,Financial plan. How to build awareness for the cause. How to raise funds from donors. Funding sources: List out grants and significant funds you’ve received. Fundraising plan: Outline how you plan to raise additional funds. The organization plans to go from local, to. international once fully established. Be very detailed in all aspects. Each description should be very detailed."
# 2. Define the labels we're hunting for.
labels = ["email", "phone_number", "user_name", "first_name", "last_name", "company_name", "url", "country", "city", "county"]

# 3. Load the PII model
model = GLiNER.from_pretrained("nvidia/gliner-pii")

# 4. Run the prediction at given threshold
entities = model.predict_entities(text, labels, threshold=0.85)

print(entities)
print(type(entities))

def predict_gliner_entities(text: str):
    entities = model.predict_entities(text, labels, threshold=0.85)
    piis = []
    for e in entities:
        if e["score"] > 0.9:
            piis.append(e["text"].lower())
    if len(piis) == 0:
        return None
    return "||".join(list(set(piis)))


if __name__ == "__main__":
    eng_queries = pandas.read_csv("../pupa/PUPA_TNB_ENG.csv")
    eng_queries = eng_queries.sample(n=20)
    eng_queries["pii_gliner"] = eng_queries["user_query"].map(predict_gliner_entities)
    
    eng_queries = eng_queries.loc[pandas.notna(eng_queries["pii_gliner"])]
    eng_queries.to_csv("../pupa/PUPA_TNB_ENG_GLiNER.csv")
    