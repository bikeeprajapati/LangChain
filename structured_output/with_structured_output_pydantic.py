from pydantic import BaseModel, Field, EmailStr,Field
from typing import Optional
from langchain_openai import ChatOpenAI
from typing import  Literal
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI()

class Review(BaseModel):
    key_themes:list[str] = Field( description="The key themes discussed in the review")
    summary: str= Field( description="A brief summary of the review")
    sentiment: Literal["positive", "negative"]= Field( description="The sentiment of the review (positive, negative)")
    pros: Optional[list[str]] = Field( None, description="The pros of the review")
    cons: Optional[list[str]] = Field( None, description="The cons of the review")
    name: Optional[str] = Field( None, description="The name of the reviewer")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The movie was fantastic! The plot was engaging and the characters were well-developed. I especially loved the cinematography. By the way, my name is John.""")
print(result.name)
