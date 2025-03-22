import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import warnings
warnings.filterwarnings("ignore")
import argparse # Command line argument parsing

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default="Python")

args = parser.parse_args()

load_dotenv() # Load vars from env file
API_KEY = os.getenv("API_KEY")

llm = OpenAI(openai_api_key=API_KEY) # Chain to use

# Prompt template and variables
code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
    )

# Language model and prompt
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt
)

# Input variables for the template
result = code_chain({
    "language": args.language,
    "task": args.task
})

print(result)