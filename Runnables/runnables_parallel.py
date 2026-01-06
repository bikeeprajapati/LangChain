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
    template  = 'Generate a tweeter post on {topic} .',
    input_variables= ['topic']
)
prompt2 = PromptTemplate(
    template  = 'Generate a LinkedIn post on: {topic} .',
    input_variables= ['topic']
)


parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "tweet": prompt1 | model1 | parser,
        "linkedin": prompt2 | model2 | parser
    }
)


result = parallel_chain.invoke({'topic': "Artificial Intelligence in Healthcare"})
print(result)

