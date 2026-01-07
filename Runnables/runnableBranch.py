from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import  RunnablePassthrough , RunnableBranch  , RunnableSequence

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)



model1 = ChatHuggingFace(llm=llm)

app = PromptTemplate(
    template  = 'Write a detailed  report on {topic} .',
    input_variables= ['topic']

)
app2 = PromptTemplate(
    template  = 'Summarize the following text \n: {text}',
    input_variables= ['text']
)

parser = StrOutputParser() 

report_gen_chain = RunnableSequence(app | model1 | parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 50,RunnableSequence(app2 | model1 | parser)),
    RunnablePassthrough()

)

final_chain = RunnableSequence(report_gen_chain | branch_chain)
result = final_chain.invoke({'topic': "Beauty of Nepal"})
print(result)

