from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()



llm1 = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm1)

parser = StrOutputParser()

class Feddback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(..., description="The sentiment of the review")

parser2 = PydanticOutputParser(pydantic_object=Feddback)

prompt1 = PromptTemplate(
    template  = 'Classify the  sentiments of the following review as Positive, Negative \n {feedback}. \n {format_instructions}',
    input_variables= ['feedback'],
    partial_variables={"format_instructions": parser2.get_format_instructions()}

)

prompt2 = PromptTemplate(
    template = "Write an appropreate response to the  positive feedback: {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template = "Write an appropreate response to the  negative feedback: {feedback}",
    input_variables=['feedback']
)
classifier_chain =  prompt1 | model | parser2


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'Positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'Negative', prompt3 | model | parser),
    RunnableLambda(lambda inputs:"could not find the sentiment")
)

chain = classifier_chain | branch_chain
feedback = "The product quality is worst ."
result = chain.invoke({'feedback': feedback})
print(result)