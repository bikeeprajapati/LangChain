from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)
app = PromptTemplate(
    template  = 'Write a summary o f the  following poem \n {poem} .',
    input_variables= ['poem']
)

loader = TextLoader("/home/bikee-prajapati/LangChain_Models/Documents_Loader/cricket.txt", encoding="utf-8")
parser = StrOutputParser()

docs = loader.load()
print(docs)
print(docs[0].page_content)

chain = app | model | parser
result = chain.invoke({'poem': docs[0].page_content})
print(result)