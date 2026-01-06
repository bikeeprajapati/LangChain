from langchain_huggingface import  ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence   
from dotenv import load_dotenv

load_dotenv()


llm  = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a joke on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='explain t his joke {text} in brief',
    input_variables=['text']
)


chain = RunnableSequence(prompt1 | model | prompt2 | model | parser)

result = chain.invoke({'topic':'technology'})
print(result)
