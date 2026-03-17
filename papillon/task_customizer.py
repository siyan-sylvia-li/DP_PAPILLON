from run_llama_dspy import CreateOnePrompt
from dspy.adapters import ChatAdapter
import pandas
from datasets import load_dataset
from constants import MALE_NAME_LIST, FEMALE_NAME_LIST

class TaskCustomizer:
    def __init__(self):
        self.data = []

    def format_prompt(self, text:str):
        pass

    def create_data(self):
        pass


class PAPILLONCustomizer(TaskCustomizer):
    def __init__(self):
        super().__init__()
        self.adapter = ChatAdapter()
        self.create_data()
    
    def format_prompt(self, text):
        inputs     = dict(userQuery=text)
        prompt_msgs = self.adapter.format(
            CreateOnePrompt,
            demos=[],
            inputs=inputs,
        )
        return prompt_msgs

    def create_data(self):
        gliner_replace_csv = pandas.read_csv("../pupa/PUPA_New_ENG_replace_full.csv")
        original_queries    = gliner_replace_csv["user_query"].tolist()
        replaced_queries    = gliner_replace_csv["gliner_replace"].tolist()
        for i in range(len(original_queries)):
            curr_item = [original_queries[i]] + replaced_queries[i].split("|<SEP>|")
            self.data.append(curr_item)
        print("PAPILLON Customizer Data Size:", len(self.data))


class GSM8kCustomizer(TaskCustomizer):
    def __init__(self):
        super().__init__()
        self.answers = []
        self.names = []
        self.prompt_prefix = """As an expert problem solver, solve step by step the following mathematical questions.

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. The final answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. The final answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. The final answer is 39

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8. The final answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9. The final answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29. The final answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33. The final answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8. The final answer is 8.

Q: <TARGET_QUESTION>
A: Let's think step by step."""
        self.create_data()
    
    def format_prompt(self, text):
        return [{"role": "user", "content": self.prompt_prefix.replace("<TARGET_QUESTION>", text)}]
    
    def create_data(self):
        # original_question,pii,substitutes,original_answer
        gsm8k_train = pandas.read_csv("../pupa_nemotron/gsm8k_train_gliner_pii.csv")
        for i, row in gsm8k_train.iterrows():
            curr_item = [row["original_question"]]
            curr_pii = row["pii"]
            subs = row["substitutes"].split("||")
            curr_item = curr_item + [row["original_question"].replace(curr_pii, s) for s in subs]
            curr_names = [curr_pii] + subs
            self.names.append(curr_names)
            self.data.append(curr_item)
            self.answers.append(row["original_answer"])
        print("GSM8k Customizer Data Size:", len(self.data))
        
    

class TruthfulQACustomizer(TaskCustomizer):
    def __init__(self):
        super().__init__()
        self.best_answers = []
        self.create_data()

    def format_prompt(self, text):
        return [
            {"role": "user", "content": "Answer the following question as a helpful, honest and harmless AI assistant.\n\n" + text}
        ]
    
    def create_data(self):
        ds = load_dataset("domenicrosati/TruthfulQA", split="train")
        for instance in ds:
            curr_item = [instance["Question"]] + [x + "asks: " + instance["Question"] for x in MALE_NAME_LIST + FEMALE_NAME_LIST]
            self.data.append(curr_item)
        print("Truthful QA Data Size:", len(self.data))


