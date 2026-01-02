from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

#1st Prompt -detailed report
template1 = PromptTemplate(
    template="Provide a detailed report on the following topic: {topic}",
    input_variables=["topic"]
)

#2nd Prompt -detailed report
template2 = PromptTemplate(
    template="Write a five line summary on the following text. /n: {text}",
    input_variables=["text"]
)

prompt1 = template1.format(topic="Artificial Intelligence")
result1 = model.invoke(prompt1)

prompt2 = template2.format(text=result1.content)
final_result = model.invoke(prompt2)
print(final_result.content)