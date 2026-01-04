from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"        
)

model = ChatHuggingFace(llm=llm)

tempalte = PromptTemplate(
    template  = 'Give me a short creative story about a {subject} in {place}.',
    input_variables=['subject','place']
)

parser = StrOutputParser()
chain =  tempalte | model | parser
result  = chain.invoke({'subject':'dragon','place':'Kathmandu'})
print(result)

chain.get_graph().print_ascii()