from json import loads
from decouple import config
from groq import Groq

class LLM:

    def __init__(self, api='GROQ', groq_model="llama3-70b-8192"):
        self.prompt = ""
        self.max_tokens = 512  # Default max tokens for response
        if api == 'GROQ':
            api_key = config('GROQ_API_KEY')
            self.groq_client = Groq(api_key=api_key)
            self.groq_model = groq_model

    def set_prompt(self, system_template=None, user_query=None, context=None):
        prompt = f"""
        Consider the user query below:

        ------ USER QUERY -----

        {user_query}

        Consider the following relevant context:

        ------ CONTEXT ------

        {context}

        Your role is as follows:

        {system_template}

        Given the context and your role, respond to the user query.
        Make sure to respond in JSON format as follows

        {{"Response": "your response"}}
        """
        self.prompt = prompt

    def set_max_tokens(self, max_tokens):
        """
        Set maximum number of tokens for LLM completion
        """
        self.max_tokens = max_tokens

    def respond_to_prompt(self):
        """
        Returns LLM response based on prompt
        """
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": self.prompt,
                    }
                ],
                temperature=0.0,
                model=self.groq_model,
                max_tokens=self.max_tokens  # ✅ apply token cap
            )
            return str(chat_completion.choices[0].message.content)

        except Exception as e:
            print(e)
            print("Unsupported LLM API or JSON parsing error ...")
            exit()
