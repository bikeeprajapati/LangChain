from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import  RunnablePassthrough , RunnableParallel , RunnableLambda , RunnableSequence
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)



model1 = ChatHuggingFace(llm=llm)

app = PromptTemplate(
    template  = 'Create a Joke about {topic} .',
    input_variables= ['topic']

)
parser = StrOutputParser() 

joke_chain = RunnableSequence(app | model1 | parser)



parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "word_count": RunnableLambda(lambda joke: len(joke.split())),
    }
)

final_chain = RunnableSequence(joke_chain | parallel_chain)
result = final_chain.invoke({'topic': "technology"})
print(result)