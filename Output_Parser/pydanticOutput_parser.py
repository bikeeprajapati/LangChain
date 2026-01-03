from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(..., description="The name of the fictional person")
    age: int = Field(..., description="The age of the fictional person")
    city: str = Field(..., description="The city where the fictional person lives")

parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template  = 'Give me  name age and city of a fictional{place} person \n {format_instructions}\n',
    input_variables=['place'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


chain =  template | model | parser
result  = chain.invoke({'place':'Kathmandu'})
print(result)