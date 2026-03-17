import dspy


import os; os.environ['LITELLM_LOG'] = 'ERROR'


class CreateOnePrompt(dspy.Signature):
    """
    You are a privacy-conscious assistant utilizing an advanced language model. When presented with a user request, your task is to craft a well-structured, privacy-preserving prompt for the language model. Focus on abstracting specific details from the user's query while capturing the essence of the request. Ensure that the prompt is general enough to avoid any personally identifiable information while still allowing the language model to generate useful and relevant responses. After creating the prompt, provide it directly without any additional commentary. Do not attempt to complete the user's original request; your sole responsibility is to present the generated prompt.
    """
    userQuery = dspy.InputField(desc="The user's request to be fulfilled.")
    createdPrompt = dspy.OutputField()

class InfoAggregator(dspy.Signature):
    """
    You are a highly skilled assistant capable of generating engaging and innovative responses. Analyze the user's request thoroughly and provide a detailed response that not only answers their query but also showcases creativity and understanding of the context. Your output should include well-structured content, relevant examples, and an engaging tone appropriate for the task at hand. Always aim to surprise the user with insightful perspectives that go beyond basic information, ensuring clarity and emotional resonance in your communication.
    """

    userQuery = dspy.InputField(desc="The user's request to be fulfilled.")
    modelExampleResponses = dspy.InputField(desc="Information from a more powerful language model responding to related queries. Complete the user query by referencing this information. Only you have access to this information.")
    finalOutput = dspy.OutputField()


class PAPILLON(dspy.Module):
    def __init__(self, untrusted_model):
        self.prompt_creater = dspy.ChainOfThought(CreateOnePrompt)
        self.info_aggregator = dspy.Predict(InfoAggregator)
        self.untrusted_model = untrusted_model

    def forward(self, user_query):
        try:
            prompt = self.prompt_creater(userQuery=user_query).createdPrompt
            response = self.untrusted_model(prompt)[0]
            output = self.info_aggregator(userQuery=user_query, modelExampleResponses=response)
        except Exception:
            return dspy.Prediction(prompt="", output="", gptResponse="")

        return dspy.Prediction(prompt=prompt, output=output.finalOutput, gptResponse=response)