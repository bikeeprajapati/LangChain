from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv
load_dotenv()
import os

#Creation of a  Tool
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b
print(multiply.invoke({"a": 3, "b": 4}))  # Example usage of the tool

#Binding the tool to a language model
llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", 
                            task="chat-completion", 
                            temperature=0, 
                            max_new_tokens=512, 
)
model = ChatHuggingFace(llm=llm)

llm_with_tools = model.bind_tools([multiply])

response = llm_with_tools.invoke("What is 3 multiplied by 4?")
print(response)