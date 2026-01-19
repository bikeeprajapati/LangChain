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

query = HumanMessage(content="What is 3 multiplied by 4?")
messages = [query]

#Tool Calling via the language model

response = llm_with_tools.invoke(messages)
messages.append(response)
print(response.tool_calls[0]["args"])

#Tool execution
tool_result = multiply.invoke(response.tool_calls[0])
messages.append(tool_result)

print("Tool result:", tool_result)
print(messages)

final_response = llm_with_tools.invoke(messages)
print("Final response from LLM:", final_response.content)