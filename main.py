from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import warnings
warnings.filterwarnings("ignore")
import argparse # Command line argument parsing

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default="Python")

args = parser.parse_args()

load_dotenv() # Load vars from env file

llm = OpenAI() # Chain to use, incl. OpenAI API key

# Prompt template and variables
code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
    )

test_prompt = PromptTemplate(
    template="Write a test for the following {language} code:\n{code}",
    input_variables=["language", "code"]
)

# Language model and prompts
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

# Chaining models together

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["test", "code"]
)

# Input variables for the template
result = chain({
    "language": args.language,
    "task": args.task
})

print(">>>>> GENERATED CODE:")
print(result["code"])
print(">>>>> GENERATED TEST:")
print(result["test"])