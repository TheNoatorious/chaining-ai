import os
from dotenv import load_dotenv
from langchain.llms import OpenAI

load_dotenv()  # Load environment variables from .env file
API_KEY = os.getenv("API_KEY")

llm = OpenAI(openai_api_key=API_KEY)

result = llm("Write a very very short poem")

print(result)