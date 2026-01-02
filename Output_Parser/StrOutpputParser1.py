from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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


parser = StrOutputParser()

chain  = template1 | model | template2 | model | parser
result = chain.invoke({'topic':"Artificial Intelligence"})
print(result)