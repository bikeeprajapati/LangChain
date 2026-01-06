from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import  RunnablePassthrough , RunnableParallel

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
    template  = 'Create a Joke{topic} .',
    input_variables= ['topic']
)
prompt2 = PromptTemplate(
    template="Explain the following joke in simple terms:\n\n{joke}",
    input_variables=["joke"]
)

parser = StrOutputParser()
joke_chain = prompt1 | model1 | parser
explain_chain = prompt2 | model2 | parser

parallel_chain = RunnableParallel(
    {
        "topic": RunnablePassthrough(),
        "joke": joke_chain,
        "explanation":  joke_chain | explain_chain
    }
)


result = parallel_chain.invoke({'topic': "technology"})
print(result)