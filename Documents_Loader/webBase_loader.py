from langchain_community.document_loaders import WebBaseLoader
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
    template  = 'Answer the following question {question} from the following content \n {content} .',
    input_variables= [ 'question', 'content' ]
)
parser = StrOutputParser()

url = "https://www.macrumors.com/"
loader = WebBaseLoader(url)
docs = loader.load()
chain = app | model  | parser 
result = chain.invoke({'question': "What is MacRumors?", 'content': docs[0].page_content})
print(result)