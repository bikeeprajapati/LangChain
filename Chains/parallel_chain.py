from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import  RunnableParallel

load_dotenv()


llm1 = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.7",
    task = "text-generation"
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template  = 'Generate a short and simple  notes on {text} .',
    input_variables= ['text']
)
prompt2 = PromptTemplate(
    template  = 'Generate a  5 pointer quiz  on: {text} .',
    input_variables= ['text']
)

prompt3 = PromptTemplate(
    template  = 'Merge the provided notes and quiz in a single document  \n : {notes}  and {quiz}.',
    input_variables= ['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model2 | parser
    }
)


chain  = parallel_chain | prompt3 | model1 | parser

text  = """Python programming language is widely used for web development, data analysis, artificial intelligence, scientific computing, and more.  Its simplicity and readability make it a favorite among both beginners and experienced developers. and the quiz title is Python Basics Quiz """

result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()