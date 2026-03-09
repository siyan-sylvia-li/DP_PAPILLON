GPT_PROMPT = """In the following messages, does the user reveal any of these categories of information? For each category that is included, list the category, whether or not it is explicit or implicit, who if anyone is explicitly involved, your confidence, and your reasoning. If a category is not present, do not list it.

Example:
2. Quoted email or messaging transcripts; explicit; the user, the user's coworker; very confident; the query contains the text of an email message.

Categories:
1. Sexually explicit or sexually suggestive content
2. Copy-and-pasted emails or messaging transcripts
3. Copy-and-pasted programming code longer than a single line
4. Job, visa, and other applications
5. Financial or corporate information
6. Academic and educational information, including homework problems and essay writing
7. Personal expressions about the user's emotions, mental health, etc.
8. Personal expressions about the user's politics and religion
9. Mention or discussion of the user's personal relationships
10. Medical and healthcare information
11. Engagement with a specific fandom, including character development, story writing, and discussions related to the fandom
12. Mention or discussion of the user's hobbies and habits
Messages: <USER_CHATBOT_INTERACTIONS>
Answer:"""

import json
import os
import dotenv
import openai
import tqdm

dotenv.load_dotenv()

def determine_categorys(user_query: str):
    prompt = GPT_PROMPT.replace("<USER_CHATBOT_INTERACTIONS>", user_query)
    response = openai.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=500
    )
    answer = response.choices[0].message.content.strip()
    selected_categories = [
        "2. Copy-and-pasted emails or messaging transcripts",
        "4. Job, visa, and other applications",
        "5. Financial or corporate information",
        "6. Academic and educational information, including homework problems and essay writing",
        "10. Medical and healthcare information"
    ]
    for s in selected_categories:
        if answer.startswith(s):
            return True, answer
    return False, answer

if __name__ == "__main__":
    curr_data = json.load(open("final_data_with_openai_eng.json", "r"))
    final_filtered_data = json.load(open("final_data_filtered_by_topic.json", "r")) if os.path.exists("final_data_filtered_by_topic.json") else []
    existing_data = set(entry["user_query"] for entry in final_filtered_data)
    for entry in tqdm.tqdm(curr_data):
        if entry["user_query"] in existing_data:
            continue        
        try:
            has_category, answer = determine_categorys("USER: " + entry["user_query"] + "\nCHATBOT: " + entry["target_response"])
        except Exception as e:
            print(f"Error processing entry: {e}")
            continue
        entry["categorys_analysis"] = answer
        final_filtered_data.append(entry)
        json.dump(final_filtered_data, open("final_data_filtered_by_topic.json", "w"))
    print(f"Original data size: {len(curr_data)}")
    print(f"Filtered data size: {len(final_filtered_data)}")
    json.dump(final_filtered_data, open("final_data_filtered_by_topic.json", "w"))