from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm  = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)    

prompt1 = PromptTemplate(
    template  = 'Generate a creative title for a story about a {subject} .',
    input_variables= ['subject']
)

prompt2 = PromptTemplate(
    template  = 'Generate a  5 pointer summary for a story with the title: {title} .',
    input_variables= ['title']
)

parser = StrOutputParser()

chain =  prompt1 | model | prompt2 | model | parser
result  = chain.invoke({'subject':'dragon'})
print(result)