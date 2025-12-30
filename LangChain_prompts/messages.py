from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm  = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation",
    max_new_tokens = 200,
    do_sample = False,
    repetition_penalty = 1.03,
    temperature = 0.7

)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of Nepal?")
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)